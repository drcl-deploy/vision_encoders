/**
 * Generic frozen-encoder inference node — one binary, n backbones.
 *
 * The artifact pair <tag>.onnx + <tag>.manifest.json (vision.onnx.v1, exported
 * by vibe's `export-encoder`) carries ALL model knowledge: preprocessing (cast,
 * /255, BGR->RGB, layout, normalization) is baked into the graph, grid/dims are
 * measured facts in the manifest. What remains here: TCP frame in -> resize ->
 * memcpy BGR HWC uint8 -> run -> publish ImageTokens. Zero per-model code.
 *
 * Producers on the other end of the TCP wire (4-byte LE length + JPEG/PNG):
 *   - real robot: camera_streamer.py on the Orin
 *   - sim:        unitree_mujoco's head-camera TCP server in unitree_sdk2_bridge.h
 */

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <vision_encoders/msg/image_tokens.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>

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

namespace
{

/// vision.onnx.v1 sidecar (JSON ⊂ YAML, so yaml-cpp parses it — no new dep).
struct EncoderManifest
{
  std::string tag;
  std::string hf_id;
  int height = 0, width = 0;      // model input (rows, cols)
  int grid_h = 0, grid_w = 0;
  int num_patches = 0, dim = 0;
  bool has_cls = false;
  std::string compute_dtype;

  static EncoderManifest load(const std::string & path)
  {
    YAML::Node root = YAML::LoadFile(path);
    const auto schema = root["schema"].as<std::string>();
    if (schema != "vision.onnx.v1") {
      throw std::runtime_error("manifest schema '" + schema + "' != vision.onnx.v1");
    }
    EncoderManifest m;
    m.tag = root["tag"].as<std::string>();
    m.hf_id = root["hf_id"].as<std::string>();
    m.height = root["input"]["height"].as<int>();
    m.width = root["input"]["width"].as<int>();
    const auto layout = root["input"]["layout"].as<std::string>();
    if (layout != "bgr_hwc_uint8") {
      throw std::runtime_error("unsupported input layout '" + layout + "'");
    }
    m.grid_h = root["patch"]["grid_h"].as<int>();
    m.grid_w = root["patch"]["grid_w"].as<int>();
    m.num_patches = root["outputs"]["patch_tokens"]["num"].as<int>();
    m.dim = root["outputs"]["patch_tokens"]["dim"].as<int>();
    m.has_cls = root["outputs"]["cls_token"].IsMap();
    m.compute_dtype = root["compute_dtype"].as<std::string>();
    if (m.grid_h * m.grid_w != m.num_patches) {
      throw std::runtime_error("manifest grid != num_patches");
    }
    return m;
  }
};

}  // namespace

class EncoderNode : public rclcpp::Node
{
public:
  EncoderNode() : Node("encoder_node")
  {
    // `model` = artifact tag -> <share>/models/<tag>.{onnx,manifest.json};
    // absolute overrides for artifacts living elsewhere.
    this->declare_parameter<std::string>("model", "");
    this->declare_parameter<std::string>("model_path", "");
    this->declare_parameter<std::string>("manifest_path", "");
    this->declare_parameter<std::string>("camera_ip", "127.0.0.1");
    this->declare_parameter<int>("camera_port", 5555);
    this->declare_parameter<int>("raw_frame_width", 640);
    this->declare_parameter<int>("raw_frame_height", 480);
    this->declare_parameter<double>("inference_rate_hz", 50.0);
    this->declare_parameter<std::string>("tokens_topic", "/enc/tokens");
    this->declare_parameter<bool>("publish_frame", true);
    this->declare_parameter<std::string>("frame_topic", "/enc/frame");

    const auto model = this->get_parameter("model").as_string();
    auto model_path = this->get_parameter("model_path").as_string();
    auto manifest_path = this->get_parameter("manifest_path").as_string();
    if (model_path.empty()) {
      if (model.empty()) {
        throw std::runtime_error("set `model` (artifact tag) or `model_path`");
      }
      const auto share = ament_index_cpp::get_package_share_directory("vision_encoders");
      model_path = share + "/models/" + model + ".onnx";
    }
    if (manifest_path.empty()) {
      manifest_path = model_path.substr(0, model_path.rfind(".onnx")) + ".manifest.json";
    }

    camera_ip_ = this->get_parameter("camera_ip").as_string();
    camera_port_ = static_cast<uint16_t>(this->get_parameter("camera_port").as_int());
    raw_frame_width_ = this->get_parameter("raw_frame_width").as_int();
    raw_frame_height_ = this->get_parameter("raw_frame_height").as_int();
    const double rate_hz = this->get_parameter("inference_rate_hz").as_double();
    publish_frame_ = this->get_parameter("publish_frame").as_bool();

    manifest_ = EncoderManifest::load(manifest_path);

    // ONNX Runtime — CUDA required, fail loud otherwise.
    auto providers = Ort::GetAvailableProviders();
    if (std::find(providers.begin(), providers.end(), "CUDAExecutionProvider") ==
        providers.end()) {
      throw std::runtime_error("CUDAExecutionProvider not available in this onnxruntime build");
    }
    env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "encoder_node");
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(1);
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    OrtCUDAProviderOptions cuda_opts{};
    opts.AppendExecutionProvider_CUDA(cuda_opts);
    session_ = std::make_unique<Ort::Session>(*env_, model_path.c_str(), opts);

    validate_graph();
    build_tensors();

    const auto qos = rclcpp::QoS(rclcpp::KeepLast(1));
    tokens_pub_ = this->create_publisher<vision_encoders::msg::ImageTokens>(
      this->get_parameter("tokens_topic").as_string(), qos);
    if (publish_frame_) {
      frame_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
        this->get_parameter("frame_topic").as_string(), qos);
    }

    running_.store(true);
    tcp_thread_ = std::thread(&EncoderNode::tcp_loop, this);
    const auto period = std::chrono::duration<double>(1.0 / rate_hz);
    timer_ = this->create_wall_timer(
      std::chrono::duration_cast<std::chrono::nanoseconds>(period),
      std::bind(&EncoderNode::infer_latest, this));

    RCLCPP_INFO(this->get_logger(),
                "%s (%s) | %dx%d -> %dx%d grid (%d patches x %d dim, cls=%s, %s) | "
                "%s:%u @ %.0f Hz",
                manifest_.tag.c_str(), manifest_.hf_id.c_str(),
                manifest_.width, manifest_.height, manifest_.grid_w, manifest_.grid_h,
                manifest_.num_patches, manifest_.dim, manifest_.has_cls ? "y" : "n",
                manifest_.compute_dtype.c_str(),
                camera_ip_.c_str(), camera_port_, rate_hz);
  }

  ~EncoderNode() override
  {
    running_.store(false);
    if (tcp_thread_.joinable()) tcp_thread_.join();
  }

private:
  /// Every manifest fact must match the graph — fail at startup, never mid-run.
  void validate_graph()
  {
    Ort::AllocatorWithDefaultOptions alloc;
    const auto in_name = session_->GetInputNameAllocated(0, alloc);
    if (std::string(in_name.get()) != "image_bgr_hwc") {
      throw std::runtime_error("graph input 0 is not image_bgr_hwc");
    }
    auto in_info = session_->GetInputTypeInfo(0);
    auto in_shape = in_info.GetTensorTypeAndShapeInfo().GetShape();
    const std::vector<int64_t> want = {1, manifest_.height, manifest_.width, 3};
    if (in_shape != want) {
      throw std::runtime_error("graph input shape != manifest (1, H, W, 3)");
    }
    const size_t n_out = session_->GetOutputCount();
    const size_t want_out = manifest_.has_cls ? 2 : 1;
    if (n_out != want_out) {
      throw std::runtime_error("graph output count != manifest");
    }
    auto out_info = session_->GetOutputTypeInfo(0);
    auto out_shape = out_info.GetTensorTypeAndShapeInfo().GetShape();
    if (out_shape.size() != 3 || out_shape[1] != manifest_.num_patches ||
        out_shape[2] != manifest_.dim) {
      throw std::runtime_error("graph patch_tokens shape != manifest (1, P, D)");
    }
  }

  /// Buffers + Ort::Values built once — zero alloc per tick.
  void build_tensors()
  {
    input_buf_.resize(static_cast<size_t>(manifest_.height) * manifest_.width * 3);
    patches_buf_.resize(static_cast<size_t>(manifest_.num_patches) * manifest_.dim);
    cls_buf_.resize(manifest_.has_cls ? manifest_.dim : 0);

    auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    const std::array<int64_t, 4> in_shape = {1, manifest_.height, manifest_.width, 3};
    input_value_ = Ort::Value::CreateTensor<uint8_t>(
      mem, input_buf_.data(), input_buf_.size(), in_shape.data(), in_shape.size());

    const std::array<int64_t, 3> tok_shape = {1, manifest_.num_patches, manifest_.dim};
    output_values_.emplace_back(Ort::Value::CreateTensor<float>(
      mem, patches_buf_.data(), patches_buf_.size(), tok_shape.data(), tok_shape.size()));
    output_names_ = {"patch_tokens"};
    if (manifest_.has_cls) {
      const std::array<int64_t, 2> cls_shape = {1, manifest_.dim};
      output_values_.emplace_back(Ort::Value::CreateTensor<float>(
        mem, cls_buf_.data(), cls_buf_.size(), cls_shape.data(), cls_shape.size()));
      output_names_.push_back("cls_token");
    }
  }

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
        std::vector<uint8_t> payload(length);
        if (!recv_exact(fd, payload.data(), length)) break;

        const auto stamp = this->now();
        const auto t0 = std::chrono::steady_clock::now();
        cv::Mat img = cv::imdecode(payload, cv::IMREAD_COLOR);  // BGR
        if (img.empty()) continue;
        // Stage-1 resize: wire -> raw_frame_* (hardware-mirror sanity step).
        if (img.cols != raw_frame_width_ || img.rows != raw_frame_height_) {
          cv::resize(img, img, cv::Size(raw_frame_width_, raw_frame_height_));
        }
        const double decode_ms = std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - t0).count();

        std::lock_guard<std::mutex> lock(frame_mutex_);
        latest_frame_ = std::move(img);
        latest_stamp_ = stamp;
        latest_decode_ms_ = decode_ms;
        has_new_frame_ = true;
      }
      ::close(fd);
      RCLCPP_WARN(this->get_logger(), "Disconnected, reconnecting...");
    }
  }

  void infer_latest()
  {
    cv::Mat frame;
    rclcpp::Time stamp;
    double decode_ms = 0.0;
    {
      std::lock_guard<std::mutex> lock(frame_mutex_);
      if (!has_new_frame_) return;
      frame = std::move(latest_frame_);
      stamp = latest_stamp_;
      decode_ms = latest_decode_ms_;
      has_new_frame_ = false;
    }

    const auto t0 = std::chrono::steady_clock::now();

    // Stage-2 resize: raw_frame_* -> manifest input. All remaining preprocessing
    // (cast, /255, BGR->RGB, layout, norm) lives inside the graph.
    if (frame.cols != manifest_.width || frame.rows != manifest_.height) {
      cv::resize(frame, frame, cv::Size(manifest_.width, manifest_.height));
    }
    if (!frame.isContinuous()) frame = frame.clone();
    std::memcpy(input_buf_.data(), frame.data, input_buf_.size());

    const char * in_names[] = {"image_bgr_hwc"};
    session_->Run(Ort::RunOptions{nullptr}, in_names, &input_value_, 1,
                  output_names_.data(), output_values_.data(), output_names_.size());

    const double infer_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - t0).count();

    vision_encoders::msg::ImageTokens msg;
    msg.header.stamp = stamp;
    msg.header.frame_id = manifest_.tag;
    msg.grid_h = static_cast<uint16_t>(manifest_.grid_h);
    msg.grid_w = static_cast<uint16_t>(manifest_.grid_w);
    msg.dim = static_cast<uint16_t>(manifest_.dim);
    msg.patches.assign(patches_buf_.begin(), patches_buf_.end());
    msg.cls.assign(cls_buf_.begin(), cls_buf_.end());
    msg.decode_ms = static_cast<float>(decode_ms);
    msg.infer_ms = static_cast<float>(infer_ms);
    tokens_pub_->publish(msg);

    // The model-input frame, as the graph sees it (bgr8, pre-graph-normalize).
    if (publish_frame_ && frame_pub_) {
      sensor_msgs::msg::Image img;
      img.header.stamp = stamp;
      img.header.frame_id = manifest_.tag;
      img.height = static_cast<uint32_t>(frame.rows);
      img.width = static_cast<uint32_t>(frame.cols);
      img.encoding = "bgr8";
      img.is_bigendian = 0;
      img.step = static_cast<uint32_t>(frame.cols * 3);
      img.data.assign(frame.datastart, frame.dataend);
      frame_pub_->publish(img);
    }
  }

  EncoderManifest manifest_;
  std::unique_ptr<Ort::Env> env_;
  std::unique_ptr<Ort::Session> session_;
  std::vector<uint8_t> input_buf_;
  std::vector<float> patches_buf_, cls_buf_;
  Ort::Value input_value_{nullptr};
  std::vector<Ort::Value> output_values_;
  std::vector<const char *> output_names_;

  std::string camera_ip_;
  uint16_t camera_port_ = 5555;
  int raw_frame_width_ = 640, raw_frame_height_ = 480;
  bool publish_frame_ = true;

  std::thread tcp_thread_;
  std::atomic<bool> running_{false};
  std::mutex frame_mutex_;
  cv::Mat latest_frame_;
  rclcpp::Time latest_stamp_;
  double latest_decode_ms_ = 0.0;
  bool has_new_frame_ = false;

  rclcpp::Publisher<vision_encoders::msg::ImageTokens>::SharedPtr tokens_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr frame_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<EncoderNode>());
  rclcpp::shutdown();
  return 0;
}
