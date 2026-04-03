cmake_minimum_required(VERSION 3.8)
project(vision_encoders)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ROS2 dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(unitree_go REQUIRED)
find_package(OpenCV REQUIRED)

# ONNX Runtime GPU — pre-built in thirdparty/
set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/onnxruntime-gpu")

if(NOT EXISTS "${ONNXRUNTIME_ROOT}/include")
  message(FATAL_ERROR
    "ONNX Runtime not found at ${ONNXRUNTIME_ROOT}\n"
    "Run: bash scripts/setup_onnxruntime.sh")
endif()

add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES
  IMPORTED_LOCATION "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so"
  INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_ROOT}/include"
)

# Theia Tiny Node (DDS subscriber — requires matching CycloneDDS versions)
add_executable(theia_tiny_node src/theia_tiny_node.cpp)
target_compile_features(theia_tiny_node PUBLIC cxx_std_17)
target_link_libraries(theia_tiny_node onnxruntime ${OpenCV_LIBS})
ament_target_dependencies(theia_tiny_node rclcpp std_msgs unitree_go)

# Theia Tiny TCP Node (TCP receiver — bypasses DDS for camera frames)
add_executable(theia_tiny_tcp_node src/theia_tiny_tcp_node.cpp)
target_compile_features(theia_tiny_tcp_node PUBLIC cxx_std_17)
target_link_libraries(theia_tiny_tcp_node onnxruntime ${OpenCV_LIBS})
ament_target_dependencies(theia_tiny_tcp_node rclcpp std_msgs)

install(TARGETS theia_tiny_node theia_tiny_tcp_node
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
