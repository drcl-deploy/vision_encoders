/**
 * Theia-Tiny inference node.
 *
 * Reads JPEG frames from a TCP stream (4-byte LE length + JPEG payload) and
 * publishes patch-token embeddings on a ROS 2 topic.
 *
 * Producers on the other end of the TCP wire:
 *   - real robot: camera_streamer.py on the Orin (pyrealsense2 + cv2.imencode)
 *   - sim:        unitree_mujoco's head-camera TCP server in unitree_sdk2_bridge.h
 */

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>

#include <array>
#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

class TheiaTinyNode : public rclcpp::Node
{
public:
  TheiaTinyNode() : Node("theia_tiny_node")
  {
    this->declare_parameter<std::string>("model_path", "");
    this->declare_parameter<std::string>("embedding_topic", "/theia_embedding");
    this->declare_parameter<std::string>("camera_ip", "127.0.0.1");
    this->declare_parameter<int>("camera_port", 5555);
    this->declare_parameter<int>("raw_frame_width", 640);
    this->declare_parameter<int>("raw_frame_height", 480);
    this->declare_parameter<int>("model_input_width", 224);
    this->declare_parameter<int>("model_input_height", 224);
    this->declare_parameter<double>("inference_rate_hz", 30.0);
    // Export-only params — declared so ROS 2 doesn't warn.
    this->declare_parameter<std::string>("hf_name", "");
    this->declare_parameter<int>("opset", 17);

    const auto model_path = this->get_parameter("model_path").as_string();
    const auto embedding_topic = this->get_parameter("embedding_topic").as_string();
    camera_ip_ = this->get_parameter("camera_ip").as_string();
    camera_port_ = static_cast<uint16_t>(this->get_parameter("camera_port").as_int());
    raw_frame_width_ = this->get_parameter("raw_frame_width").as_int();
    raw_frame_height_ = this->get_parameter("raw_frame_height").as_int();
    model_input_width_ = this->get_parameter("model_input_width").as_int();
    model_input_height_ = this->get_parameter("model_input_height").as_int();
    const double inference_rate_hz = this->get_parameter("inference_rate_hz").as_double();

    if (model_path.empty()) {
      RCLCPP_FATAL(this->get_logger(), "Parameter 'model_path' must be set");
      throw std::runtime_error("model_path not set");
    }

    input_size_ = 3 * model_input_height_ * model_input_width_;
    num_patches_ = (model_input_height_ / 16) * (model_input_width_ / 16);

    // ONNX Runtime — CUDA required, fail loud otherwise.
    auto providers = Ort::GetAvailableProviders();
    bool has_cuda = std::find(providers.begin(), providers.end(),
                              "CUDAExecutionProvider") != providers.end();
    if (!has_cuda) {
      RCLCPP_FATAL(this->get_logger(),
                   "CUDAExecutionProvider not available in this onnxruntime build. Aborting.");
      throw std::runtime_error("CUDA EP missing");
    }

    env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "theia_tiny");
    Ort::SessionOptions session_opts;
    session_opts.SetIntraOpNumThreads(1);
    session_opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    OrtCUDAProviderOptions cuda_opts{};
    cuda_opts.device_id = 0;
    session_opts.AppendExecutionProvider_CUDA(cuda_opts);

    session_ = std::make_unique<Ort::Session>(*env_, model_path.c_str(), session_opts);
    RCLCPP_INFO(this->get_logger(), "Loaded ONNX model (CUDA): %s", model_path.c_str());

    // Query hidden_dim from the ONNX output shape: [B, num_patches, hidden_dim].
    auto out_shape = session_->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    if (out_shape.size() == 3 && out_shape[2] > 0) {
      hidden_dim_ = static_cast<size_t>(out_shape[2]);
    }
    output_size_ = num_patches_ * hidden_dim_;

    RCLCPP_INFO(this->get_logger(),
                "raw %dx%d -> model %dx%d -> %zu patches x %zu dim = %zu floats",
                raw_frame_width_, raw_frame_height_,
                model_input_width_, model_input_height_,
                num_patches_, hidden_dim_, output_size_);

    input_buf_.resize(input_size_);
    pub_ = this->create_publisher<std_msgs::msg::Float32MultiArray>(embedding_topic, 10);
    stats_pub_ = this->create_publisher<std_msgs::msg::Float32MultiArray>("/theia_stats", 10);

    running_.store(true);
    tcp_thread_ = std::thread(&TheiaTinyNode::tcp_loop, this);
    const auto period = std::chrono::duration<double>(1.0 / inference_rate_hz);
    timer_ = this->create_wall_timer(
      std::chrono::duration_cast<std::chrono::nanoseconds>(period),
      std::bind(&TheiaTinyNode::infer_latest, this));

    RCLCPP_INFO(this->get_logger(),
                "Connecting to %s:%u  |  Publishing: %s  |  Inference tick: %.1f Hz",
                camera_ip_.c_str(), camera_port_, embedding_topic.c_str(), inference_rate_hz);
  }

  ~TheiaTinyNode() override
  {
    running_.store(false);
    if (tcp_thread_.joinable()) tcp_thread_.join();
  }

private:
  static constexpr size_t kChannels = 3;
  static constexpr float kMean[] = {0.485f, 0.456f, 0.406f};
  static constexpr float kStd[]  = {0.229f, 0.224f, 0.225f};

  static bool recv_exact(int fd, uint8_t * buf, size_t n)
  {
    size_t got = 0;
    while (got < n) {
      ssize_t r = ::recv(fd, buf + got, n - got, 0);
      if (r <= 0) return false;
      got += static_cast<size_t>(r);
    }
    return true;
  }

  void tcp_loop()
  {
    while (running_.load()) {
      int fd = ::socket(AF_INET, SOCK_STREAM, 0);
      if (fd < 0) {
        std::this_thread::sleep_for(std::chrono::seconds(2));
        continue;
      }
      sockaddr_in addr{};
      addr.sin_family = AF_INET;
      addr.sin_port = htons(camera_port_);
      inet_pton(AF_INET, camera_ip_.c_str(), &addr.sin_addr);

      if (::connect(fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) < 0) {
        RCLCPP_WARN(this->get_logger(), "Cannot connect to %s:%u, retrying...",
                    camera_ip_.c_str(), camera_port_);
        ::close(fd);
        std::this_thread::sleep_for(std::chrono::seconds(2));
        continue;
      }
      RCLCPP_INFO(this->get_logger(), "Connected to %s:%u", camera_ip_.c_str(), camera_port_);

      while (running_.load()) {
        uint32_t length = 0;
        if (!recv_exact(fd, reinterpret_cast<uint8_t *>(&length), 4)) break;

        std::vector<uint8_t> jpg_buf(length);
        if (!recv_exact(fd, jpg_buf.data(), length)) break;

        auto t_decode_start = std::chrono::steady_clock::now();
        cv::Mat img = cv::imdecode(jpg_buf, cv::IMREAD_COLOR);  // BGR
        if (img.empty()) continue;

        // Stage 1 resize: wire -> raw_frame_*. Sanity step in case the producer
        // doesn't honor the agreed-upon hardware-mirror resolution.
        if (img.cols != raw_frame_width_ || img.rows != raw_frame_height_) {
          cv::resize(img, img, cv::Size(raw_frame_width_, raw_frame_height_));
        }
        double tcp_decode_ms = std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - t_decode_start).count();

        std::lock_guard<std::mutex> lock(frame_mutex_);
        latest_frame_ = std::move(img);
        latest_tcp_decode_ms_ = tcp_decode_ms;
        has_new_frame_ = true;
      }

      ::close(fd);
      RCLCPP_WARN(this->get_logger(), "Disconnected, reconnecting...");
    }
  }

  void infer_latest()
  {
    cv::Mat frame;
    double tcp_decode_ms = 0.0;
    {
      std::lock_guard<std::mutex> lock(frame_mutex_);
      if (!has_new_frame_) return;
      frame = std::move(latest_frame_);
      tcp_decode_ms = latest_tcp_decode_ms_;
      has_new_frame_ = false;
    }

    auto t0 = std::chrono::steady_clock::now();

    // Stage 2 resize: raw_frame_* -> model_input_*. This is the resolution the
    // ONNX was exported at and the policy was trained against.
    if (frame.cols != model_input_width_ || frame.rows != model_input_height_) {
      cv::resize(frame, frame, cv::Size(model_input_width_, model_input_height_));
    }

    // BGR -> normalized RGB, CHW layout. ImageNet stats, matches Isaac Lab's
    // image_features / _prepare_theia_transformer_model.
    const size_t h = static_cast<size_t>(frame.rows);
    const size_t w = static_cast<size_t>(frame.cols);
    const size_t hw = h * w;
    for (size_t y = 0; y < h; ++y) {
      const uint8_t * row = frame.ptr<uint8_t>(y);
      for (size_t x = 0; x < w; ++x) {
        input_buf_[0 * hw + y * w + x] = (row[x * 3 + 2] / 255.0f - kMean[0]) / kStd[0]; // R
        input_buf_[1 * hw + y * w + x] = (row[x * 3 + 1] / 255.0f - kMean[1]) / kStd[1]; // G
        input_buf_[2 * hw + y * w + x] = (row[x * 3 + 0] / 255.0f - kMean[2]) / kStd[2]; // B
      }
    }

    std::array<int64_t, 4> input_shape = {
      1, static_cast<int64_t>(kChannels),
      static_cast<int64_t>(model_input_height_),
      static_cast<int64_t>(model_input_width_)
    };
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
      memory_info, input_buf_.data(), input_size_,
      input_shape.data(), input_shape.size());

    const char * input_names[]  = {"pixel_values"};
    const char * output_names[] = {"features"};
    auto output_tensors = session_->Run(
      Ort::RunOptions{nullptr},
      input_names, &input_tensor, 1,
      output_names, 1);

    const float * features = output_tensors[0].GetTensorData<float>();
    std_msgs::msg::Float32MultiArray out_msg;
    out_msg.layout.dim.resize(2);
    out_msg.layout.dim[0].label = "patches";
    out_msg.layout.dim[0].size = num_patches_;
    out_msg.layout.dim[0].stride = num_patches_ * hidden_dim_;
    out_msg.layout.dim[1].label = "hidden";
    out_msg.layout.dim[1].size = hidden_dim_;
    out_msg.layout.dim[1].stride = hidden_dim_;
    out_msg.data.assign(features, features + output_size_);
    pub_->publish(out_msg);

    double infer_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - t0).count();

    std_msgs::msg::Float32MultiArray stats_msg;
    stats_msg.data = {static_cast<float>(tcp_decode_ms), static_cast<float>(infer_ms)};
    stats_pub_->publish(stats_msg);

    RCLCPP_DEBUG(this->get_logger(), "TCP decode: %.1f ms | Inference: %.1f ms",
                 tcp_decode_ms, infer_ms);
  }

  std::unique_ptr<Ort::Env> env_;
  std::unique_ptr<Ort::Session> session_;

  int raw_frame_width_ = 640;
  int raw_frame_height_ = 480;
  int model_input_width_ = 224;
  int model_input_height_ = 224;
  size_t input_size_ = 0;
  size_t num_patches_ = 0;
  size_t hidden_dim_ = 192;
  size_t output_size_ = 0;
  std::vector<float> input_buf_;

  std::string camera_ip_;
  uint16_t camera_port_ = 5555;

  std::thread tcp_thread_;
  std::atomic<bool> running_{false};
  std::mutex frame_mutex_;
  cv::Mat latest_frame_;
  double latest_tcp_decode_ms_ = 0.0;
  bool has_new_frame_ = false;

  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr pub_;
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr stats_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<TheiaTinyNode>());
  rclcpp::shutdown();
  return 0;
}
