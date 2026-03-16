#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <unitree_go/msg/go2_front_video_data.hpp>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

#include <memory>
#include <vector>
#include <string>
#include <array>
#include <chrono>

class TheiaTinyNode : public rclcpp::Node
{
public:
  TheiaTinyNode() : Node("theia_tiny_node")
  {
    // Parameters (all loaded from config/theia_tiny.yaml via launch file)
    this->declare_parameter<std::string>("model_path", "");
    this->declare_parameter<std::string>("image_topic", "/head_camera");
    this->declare_parameter<std::string>("embedding_topic", "/theia_embedding");
    this->declare_parameter<std::string>("resolution", "720p");
    this->declare_parameter<int>("cam_width", 640);
    this->declare_parameter<int>("cam_height", 480);
    this->declare_parameter<bool>("use_gpu", true);
    // Export-only params — declared so ROS2 doesn't warn about them
    this->declare_parameter<std::string>("hf_name", "");
    this->declare_parameter<int>("opset", 17);

    const auto model_path = this->get_parameter("model_path").as_string();
    const auto image_topic = this->get_parameter("image_topic").as_string();
    const auto embedding_topic = this->get_parameter("embedding_topic").as_string();
    resolution_ = this->get_parameter("resolution").as_string();
    cam_width_ = this->get_parameter("cam_width").as_int();
    cam_height_ = this->get_parameter("cam_height").as_int();
    const bool use_gpu = this->get_parameter("use_gpu").as_bool();

    raw_size_ = cam_width_ * cam_height_ * 3;
    input_size_ = 3 * cam_height_ * cam_width_;
    num_patches_ = (cam_height_ / 16) * (cam_width_ / 16);

    if (model_path.empty()) {
      RCLCPP_FATAL(this->get_logger(), "Parameter 'model_path' must be set");
      throw std::runtime_error("model_path not set");
    }

    // ONNX Runtime setup
    env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "theia_tiny");
    Ort::SessionOptions session_opts;
    session_opts.SetIntraOpNumThreads(1);
    session_opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    if (use_gpu) {
      try {
        OrtCUDAProviderOptions cuda_opts{};
        cuda_opts.device_id = 0;
        session_opts.AppendExecutionProvider_CUDA(cuda_opts);
        RCLCPP_INFO(this->get_logger(), "CUDA execution provider enabled");
      } catch (const Ort::Exception & e) {
        RCLCPP_WARN(this->get_logger(), "CUDA not available, falling back to CPU: %s", e.what());
      }
    }

    session_ = std::make_unique<Ort::Session>(*env_, model_path.c_str(), session_opts);
    RCLCPP_INFO(this->get_logger(), "Loaded ONNX model: %s", model_path.c_str());

    // Query hidden dim from output shape: [batch, num_patches, hidden_dim]
    auto out_info = session_->GetOutputTypeInfo(0);
    auto tensor_info = out_info.GetTensorTypeAndShapeInfo();
    auto out_shape = tensor_info.GetShape();
    if (out_shape.size() == 3 && out_shape[2] > 0) {
      hidden_dim_ = static_cast<size_t>(out_shape[2]);
    }
    output_size_ = num_patches_ * hidden_dim_;

    RCLCPP_INFO(this->get_logger(), "Camera: %dx%d → %zu patches x %zu dim = %zu floats",
                cam_width_, cam_height_, num_patches_, hidden_dim_, output_size_);

    // Pre-allocate input buffer
    input_buf_.resize(input_size_);

    // Publisher & subscriber
    pub_ = this->create_publisher<std_msgs::msg::Float32MultiArray>(embedding_topic, 10);
    sub_ = this->create_subscription<unitree_go::msg::Go2FrontVideoData>(
      image_topic, 10,
      std::bind(&TheiaTinyNode::image_callback, this, std::placeholders::_1));

    RCLCPP_INFO(this->get_logger(), "Subscribing: %s (%s)  |  Publishing: %s",
                image_topic.c_str(), resolution_.c_str(), embedding_topic.c_str());
  }

private:
  static constexpr size_t kChannels = 3;

  // ImageNet normalization constants
  static constexpr float kMean[] = {0.485f, 0.456f, 0.406f};
  static constexpr float kStd[] = {0.229f, 0.224f, 0.225f};

  const std::vector<uint8_t> & select_frame(
    const unitree_go::msg::Go2FrontVideoData::SharedPtr & msg) const
  {
    if (resolution_ == "360p") return msg->video360p;
    if (resolution_ == "180p") return msg->video180p;
    return msg->video720p;
  }

  void image_callback(const unitree_go::msg::Go2FrontVideoData::SharedPtr msg)
  {
    const auto & frame_data = select_frame(msg);
    if (frame_data.empty()) {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                           "Empty %s frame — skipping", resolution_.c_str());
      return;
    }

    if (frame_data.size() != raw_size_) {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                           "Unexpected frame size: %zu (expected %zu)", frame_data.size(), raw_size_);
      return;
    }

    auto t0 = std::chrono::steady_clock::now();

    // Wrap raw RGB data as cv::Mat (no copy) — feed at native resolution
    cv::Mat rgb(cam_height_, cam_width_, CV_8UC3, const_cast<uint8_t *>(frame_data.data()));

    // Normalize to float CHW with ImageNet stats
    const size_t h = cam_height_;
    const size_t w = cam_width_;
    const size_t hw = h * w;
    for (size_t y = 0; y < h; ++y) {
      const uint8_t * row = rgb.ptr<uint8_t>(y);
      for (size_t x = 0; x < w; ++x) {
        for (size_t c = 0; c < kChannels; ++c) {
          float val = static_cast<float>(row[x * 3 + c]) / 255.0f;
          input_buf_[c * hw + y * w + x] = (val - kMean[c]) / kStd[c];
        }
      }
    }

    // Create input tensor at native resolution
    std::array<int64_t, 4> input_shape = {
      1,
      static_cast<int64_t>(kChannels),
      static_cast<int64_t>(cam_height_),
      static_cast<int64_t>(cam_width_)
    };
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
      memory_info,
      input_buf_.data(),
      input_size_,
      input_shape.data(),
      input_shape.size());

    // Run inference
    const char * input_names[] = {"pixel_values"};
    const char * output_names[] = {"features"};

    auto output_tensors = session_->Run(
      Ort::RunOptions{nullptr},
      input_names, &input_tensor, 1,
      output_names, 1);

    // Publish patch token embeddings — shape (1, num_patches, hidden_dim)
    const float * features = output_tensors[0].GetTensorData<float>();

    auto out_msg = std_msgs::msg::Float32MultiArray();
    out_msg.layout.dim.resize(2);
    out_msg.layout.dim[0].label = "patches";
    out_msg.layout.dim[0].size = num_patches_;
    out_msg.layout.dim[0].stride = num_patches_ * hidden_dim_;
    out_msg.layout.dim[1].label = "hidden";
    out_msg.layout.dim[1].size = hidden_dim_;
    out_msg.layout.dim[1].stride = hidden_dim_;
    out_msg.data.assign(features, features + output_size_);
    pub_->publish(out_msg);

    auto dt = std::chrono::steady_clock::now() - t0;
    double ms = std::chrono::duration<double, std::milli>(dt).count();
    RCLCPP_DEBUG(this->get_logger(), "Inference: %.1f ms", ms);
  }

  std::unique_ptr<Ort::Env> env_;
  std::unique_ptr<Ort::Session> session_;
  int cam_width_ = 640;
  int cam_height_ = 480;
  size_t raw_size_ = 640 * 480 * 3;
  size_t input_size_ = 3 * 480 * 640;
  size_t num_patches_ = 30 * 40;   // (480/16) * (640/16) = 1200
  size_t hidden_dim_ = 192;
  size_t output_size_ = 1200 * 192;
  std::string resolution_;
  std::vector<float> input_buf_;

  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr pub_;
  rclcpp::Subscription<unitree_go::msg::Go2FrontVideoData>::SharedPtr sub_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<TheiaTinyNode>());
  rclcpp::shutdown();
  return 0;
}
