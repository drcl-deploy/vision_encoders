/**
 * Theia Tiny inference node that reads camera frames from the Orin's TCP
 * streamer (camera_streamer.py) instead of the DDS /frontvideostream topic.
 *
 * This bypasses the CycloneDDS 0.7 (Foxy/Orin) <-> 0.10 (Humble/desktop)
 * interop bug that truncates large uint8[] sequences.
 *
 * Publishes the same /theia_embedding topic as theia_tiny_node.
 */

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

#include <memory>
#include <vector>
#include <string>
#include <array>
#include <chrono>
#include <thread>
#include <atomic>
#include <mutex>
#include <cstring>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>

class TheiaTinyTcpNode : public rclcpp::Node
{
public:
  TheiaTinyTcpNode() : Node("theia_tiny_tcp_node")
  {
    // Parameters
    this->declare_parameter<std::string>("model_path", "");
    this->declare_parameter<std::string>("embedding_topic", "/theia_embedding");
    this->declare_parameter<std::string>("orin_ip", "192.168.123.164");
    this->declare_parameter<int>("orin_port", 5555);
    this->declare_parameter<int>("cam_width", 640);
    this->declare_parameter<int>("cam_height", 480);
    this->declare_parameter<bool>("use_gpu", true);
    // Export-only params — declared so ROS2 doesn't warn about them
    this->declare_parameter<std::string>("hf_name", "");
    this->declare_parameter<int>("opset", 17);

    const auto model_path = this->get_parameter("model_path").as_string();
    const auto embedding_topic = this->get_parameter("embedding_topic").as_string();
    orin_ip_ = this->get_parameter("orin_ip").as_string();
    orin_port_ = static_cast<uint16_t>(this->get_parameter("orin_port").as_int());
    cam_width_ = this->get_parameter("cam_width").as_int();
    cam_height_ = this->get_parameter("cam_height").as_int();
    const bool use_gpu = this->get_parameter("use_gpu").as_bool();

    input_size_ = 3 * cam_height_ * cam_width_;
    num_patches_ = (cam_height_ / 16) * (cam_width_ / 16);

    if (model_path.empty()) {
      RCLCPP_FATAL(this->get_logger(), "Parameter 'model_path' must be set");
      throw std::runtime_error("model_path not set");
    }

    // ONNX Runtime setup
    env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "theia_tiny_tcp");
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

    RCLCPP_INFO(this->get_logger(), "Camera: %dx%d -> %zu patches x %zu dim = %zu floats",
                cam_width_, cam_height_, num_patches_, hidden_dim_, output_size_);

    // Pre-allocate input buffer
    input_buf_.resize(input_size_);

    // Publisher
    pub_ = this->create_publisher<std_msgs::msg::Float32MultiArray>(embedding_topic, 10);

    // Start TCP receiver thread
    running_.store(true);
    tcp_thread_ = std::thread(&TheiaTinyTcpNode::tcp_loop, this);

    // Timer to process latest frame
    timer_ = this->create_wall_timer(
      std::chrono::milliseconds(33),  // ~30 fps
      std::bind(&TheiaTinyTcpNode::infer_latest, this));

    RCLCPP_INFO(this->get_logger(), "Connecting to %s:%d  |  Publishing: %s",
                orin_ip_.c_str(), orin_port_, embedding_topic.c_str());
  }

  ~TheiaTinyTcpNode() override
  {
    running_.store(false);
    if (tcp_thread_.joinable()) {
      tcp_thread_.join();
    }
  }

private:
  static constexpr size_t kChannels = 3;
  static constexpr float kMean[] = {0.485f, 0.456f, 0.406f};
  static constexpr float kStd[] = {0.229f, 0.224f, 0.225f};

  /** Read exactly n bytes from a socket. Returns false on disconnect. */
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

  /** Background thread: connect to Orin, receive JPEG frames, decode to BGR. */
  void tcp_loop()
  {
    while (running_.load()) {
      int fd = ::socket(AF_INET, SOCK_STREAM, 0);
      if (fd < 0) {
        RCLCPP_ERROR(this->get_logger(), "socket() failed");
        std::this_thread::sleep_for(std::chrono::seconds(2));
        continue;
      }

      sockaddr_in addr{};
      addr.sin_family = AF_INET;
      addr.sin_port = htons(orin_port_);
      inet_pton(AF_INET, orin_ip_.c_str(), &addr.sin_addr);

      if (::connect(fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) < 0) {
        RCLCPP_WARN(this->get_logger(), "Cannot connect to %s:%d, retrying...",
                    orin_ip_.c_str(), orin_port_);
        ::close(fd);
        std::this_thread::sleep_for(std::chrono::seconds(2));
        continue;
      }
      RCLCPP_INFO(this->get_logger(), "Connected to %s:%d", orin_ip_.c_str(), orin_port_);

      while (running_.load()) {
        // Read 4-byte little-endian length
        uint32_t length = 0;
        if (!recv_exact(fd, reinterpret_cast<uint8_t *>(&length), 4)) break;

        // Read JPEG payload
        std::vector<uint8_t> jpg_buf(length);
        if (!recv_exact(fd, jpg_buf.data(), length)) break;

        // Decode JPEG -> BGR cv::Mat
        cv::Mat img = cv::imdecode(jpg_buf, cv::IMREAD_COLOR);
        if (img.empty()) continue;

        // Resize to expected dimensions if needed
        if (img.cols != cam_width_ || img.rows != cam_height_) {
          cv::resize(img, img, cv::Size(cam_width_, cam_height_));
        }

        {
          std::lock_guard<std::mutex> lock(frame_mutex_);
          latest_frame_ = std::move(img);
          has_new_frame_ = true;
        }
      }

      ::close(fd);
      RCLCPP_WARN(this->get_logger(), "Disconnected, reconnecting...");
    }
  }

  /** Timer callback: grab latest frame and run ONNX inference. */
  void infer_latest()
  {
    cv::Mat frame;
    {
      std::lock_guard<std::mutex> lock(frame_mutex_);
      if (!has_new_frame_) return;
      frame = std::move(latest_frame_);
      has_new_frame_ = false;
    }

    auto t0 = std::chrono::steady_clock::now();

    // Normalize BGR -> float CHW with ImageNet stats
    const size_t h = static_cast<size_t>(frame.rows);
    const size_t w = static_cast<size_t>(frame.cols);
    const size_t hw = h * w;
    for (size_t y = 0; y < h; ++y) {
      const uint8_t * row = frame.ptr<uint8_t>(y);
      for (size_t x = 0; x < w; ++x) {
        // OpenCV BGR -> RGB reorder: channel 0=R, 1=G, 2=B
        input_buf_[0 * hw + y * w + x] = (static_cast<float>(row[x * 3 + 2]) / 255.0f - kMean[0]) / kStd[0];
        input_buf_[1 * hw + y * w + x] = (static_cast<float>(row[x * 3 + 1]) / 255.0f - kMean[1]) / kStd[1];
        input_buf_[2 * hw + y * w + x] = (static_cast<float>(row[x * 3 + 0]) / 255.0f - kMean[2]) / kStd[2];
      }
    }

    // Create input tensor
    std::array<int64_t, 4> input_shape = {
      1,
      static_cast<int64_t>(kChannels),
      static_cast<int64_t>(cam_height_),
      static_cast<int64_t>(cam_width_)
    };
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
      memory_info, input_buf_.data(), input_size_,
      input_shape.data(), input_shape.size());

    // Run inference
    const char * input_names[] = {"pixel_values"};
    const char * output_names[] = {"features"};
    auto output_tensors = session_->Run(
      Ort::RunOptions{nullptr},
      input_names, &input_tensor, 1,
      output_names, 1);

    // Publish patch token embeddings — shape (num_patches, hidden_dim)
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

  // ONNX
  std::unique_ptr<Ort::Env> env_;
  std::unique_ptr<Ort::Session> session_;

  // Camera / model dimensions
  int cam_width_ = 640;
  int cam_height_ = 480;
  size_t input_size_ = 3 * 480 * 640;
  size_t num_patches_ = 30 * 40;
  size_t hidden_dim_ = 192;
  size_t output_size_ = 1200 * 192;
  std::vector<float> input_buf_;

  // TCP connection
  std::string orin_ip_;
  uint16_t orin_port_ = 5555;

  // Threading
  std::thread tcp_thread_;
  std::atomic<bool> running_{false};
  std::mutex frame_mutex_;
  cv::Mat latest_frame_;
  bool has_new_frame_ = false;

  // ROS
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr pub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<TheiaTinyTcpNode>());
  rclcpp::shutdown();
  return 0;
}
