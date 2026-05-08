cmake_minimum_required(VERSION 3.8)
project(vision_encoders)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ROS2 dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(OpenCV REQUIRED)

# ONNX Runtime GPU — symlink pre-built release into thirdparty/
# (same pattern as cpp_control; see readme.md)
set(ONNXRUNTIME_VERSION "1.22.0")
set(ONNXRUNTIME_ROOT
  "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/onnxruntime-linux-x64-gpu-${ONNXRUNTIME_VERSION}")

if(NOT EXISTS "${ONNXRUNTIME_ROOT}/include")
  message(FATAL_ERROR
    "ONNX Runtime GPU not found at ${ONNXRUNTIME_ROOT}\n"
    "Download onnxruntime-linux-x64-gpu-${ONNXRUNTIME_VERSION}.tgz from\n"
    "  https://github.com/microsoft/onnxruntime/releases/tag/v${ONNXRUNTIME_VERSION}\n"
    "extract it, and symlink it into thirdparty/:\n"
    "  ln -s /PATH/TO/onnxruntime-linux-x64-gpu-${ONNXRUNTIME_VERSION} "
    "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/")
endif()

add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES
  IMPORTED_LOCATION "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so"
  INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_ROOT}/include"
)

# Theia Tiny Node — unified TCP receiver for both sim and real.
add_executable(theia_tiny_node src/theia_tiny_node.cpp)
target_compile_features(theia_tiny_node PUBLIC cxx_std_17)
target_link_libraries(theia_tiny_node onnxruntime ${OpenCV_LIBS})
ament_target_dependencies(theia_tiny_node rclcpp std_msgs sensor_msgs)

install(TARGETS theia_tiny_node
  DESTINATION lib/${PROJECT_NAME})

# Install ONNX Runtime shared libs alongside the node
install(
  DIRECTORY "${ONNXRUNTIME_ROOT}/lib/"
  DESTINATION lib
  FILES_MATCHING PATTERN "*.so*"
)

# Install config
install(
  DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
)

# Install launch files
install(
  DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

# Install scripts
install(
  DIRECTORY scripts/
  DESTINATION share/${PROJECT_NAME}/scripts
  USE_SOURCE_PERMISSIONS
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
