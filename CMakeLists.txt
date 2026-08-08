cmake_minimum_required(VERSION 3.8)
project(vision_encoders)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ROS2 dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)

# Package msgs
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/ImageTokens.msg"
  DEPENDENCIES std_msgs
)

# ONNX Runtime GPU — symlink pre-built (or source-built) release into thirdparty/
# Auto-detects architecture and version. See readme.md.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64")
    set(_ort_arch "x64")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    set(_ort_arch "aarch64")
endif()
file(GLOB _ort_candidates "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/onnxruntime-linux-${_ort_arch}-gpu-*")
list(SORT _ort_candidates ORDER DESCENDING)
list(LENGTH _ort_candidates _ort_count)
if(_ort_count GREATER 0)
    list(GET _ort_candidates 0 ONNXRUNTIME_ROOT)
else()
    set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/NOTFOUND")
endif()

if(NOT EXISTS "${ONNXRUNTIME_ROOT}/include")
  message(FATAL_ERROR
    "ONNX Runtime GPU not found in ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/\n"
    "Expected: onnxruntime-linux-${_ort_arch}-gpu-<version>\n"
    "Download or build onnxruntime with CUDA, then symlink into thirdparty/:\n"
    "  ln -s /PATH/TO/onnxruntime-linux-${_ort_arch}-gpu-<version> "
    "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/")
endif()
message(STATUS "Found ONNX Runtime GPU at ${ONNXRUNTIME_ROOT}")

add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES
  IMPORTED_LOCATION "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so"
  INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_ROOT}/include"
)

# The generic encoder node — one binary, model knowledge lives in the artifact.
add_executable(encoder_node src/encoder_node.cpp)
target_compile_features(encoder_node PUBLIC cxx_std_17)
target_link_libraries(encoder_node onnxruntime ${OpenCV_LIBS} yaml-cpp)
ament_target_dependencies(encoder_node rclcpp std_msgs sensor_msgs ament_index_cpp)
if(COMMAND rosidl_get_typesupport_target)
  # Galactic and newer.
  rosidl_get_typesupport_target(cpp_typesupport_target
    ${PROJECT_NAME} "rosidl_typesupport_cpp")
  target_link_libraries(encoder_node "${cpp_typesupport_target}")
else()
  # Foxy compatibility: rosidl_get_typesupport_target() is not available.
  rosidl_target_interfaces(encoder_node
    ${PROJECT_NAME} "rosidl_typesupport_cpp")
endif()

# The node lives in lib/<PROJECT_NAME>/ but ORT + CUDA libs install to lib/.
# Set RPATH so the dynamic linker finds them relative to the binary.
set_target_properties(encoder_node PROPERTIES
  INSTALL_RPATH "$ORIGIN/../"
  BUILD_WITH_INSTALL_RPATH FALSE
)

install(TARGETS encoder_node
  DESTINATION lib/${PROJECT_NAME})

# Install ONNX Runtime shared libs alongside the node
install(
  DIRECTORY "${ONNXRUNTIME_ROOT}/lib/"
  DESTINATION lib
  FILES_MATCHING PATTERN "*.so*"
)

install(DIRECTORY config/ DESTINATION share/${PROJECT_NAME}/config)
install(DIRECTORY launch/ DESTINATION share/${PROJECT_NAME}/launch)
install(DIRECTORY scripts/ DESTINATION share/${PROJECT_NAME}/scripts
        USE_SOURCE_PERMISSIONS)
install(DIRECTORY models/ DESTINATION share/${PROJECT_NAME}/models)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
