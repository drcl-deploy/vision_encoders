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

# Package msgs
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/ImageTokens.msg"
  DEPENDENCIES std_msgs
)

option(VISION_ENCODERS_BUILD_RUNTIME "Build the CUDA encoder (OFF for offboard interfaces)" ON)
if(VISION_ENCODERS_BUILD_RUNTIME)
  find_package(OpenCV REQUIRED)
  find_package(yaml-cpp REQUIRED)

  # ONNX Runtime is provisioned explicitly by scripts/setup_onnxruntime.sh. An
  # explicit CMake value wins, followed by the environment, then the stable link
  # managed by that script. Do not scan versioned directories: it makes clean
  # builds depend on whatever stale archive happens to sort last.
  set(ONNXRUNTIME_ROOT "" CACHE PATH "ONNX Runtime SDK root")
  if(NOT ONNXRUNTIME_ROOT AND DEFINED ENV{ONNXRUNTIME_ROOT})
    set(ONNXRUNTIME_ROOT "$ENV{ONNXRUNTIME_ROOT}")
  endif()
  if(NOT ONNXRUNTIME_ROOT)
    set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/onnxruntime")
  endif()
  get_filename_component(ONNXRUNTIME_ROOT "${ONNXRUNTIME_ROOT}" REALPATH)

  unset(ONNXRUNTIME_INCLUDE_DIR CACHE)
  unset(ONNXRUNTIME_LIBRARY CACHE)
  find_path(ONNXRUNTIME_INCLUDE_DIR
    NAMES onnxruntime_cxx_api.h
    PATHS "${ONNXRUNTIME_ROOT}/include"
    NO_DEFAULT_PATH)
  find_library(ONNXRUNTIME_LIBRARY
    NAMES onnxruntime
    PATHS "${ONNXRUNTIME_ROOT}/lib"
    NO_DEFAULT_PATH)
  set(ONNXRUNTIME_CUDA_PROVIDER
    "${ONNXRUNTIME_ROOT}/lib/libonnxruntime_providers_cuda.so")
  set(ONNXRUNTIME_SHARED_PROVIDER
    "${ONNXRUNTIME_ROOT}/lib/libonnxruntime_providers_shared.so")

  if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY OR
    NOT EXISTS "${ONNXRUNTIME_CUDA_PROVIDER}" OR
    NOT EXISTS "${ONNXRUNTIME_SHARED_PROVIDER}")
    message(FATAL_ERROR
      "A CUDA-enabled ONNX Runtime SDK was not found at:\n"
      "  ${ONNXRUNTIME_ROOT}\n"
      "Run scripts/setup_onnxruntime.sh, or configure with:\n"
      "  -DONNXRUNTIME_ROOT=/path/to/onnxruntime")
  endif()
  message(STATUS "ONNX Runtime root: ${ONNXRUNTIME_ROOT}")

  add_library(onnxruntime SHARED IMPORTED GLOBAL)
  set_target_properties(onnxruntime PROPERTIES
    IMPORTED_LOCATION "${ONNXRUNTIME_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE_DIR}"
  )

  # The generic encoder node — one binary, model knowledge lives in the artifact.
  add_executable(encoder_node src/encoder_node.cpp)
  target_compile_features(encoder_node PUBLIC cxx_std_17)
  target_include_directories(encoder_node PRIVATE include)
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

  # Small deployment check: loading the CUDA provider catches missing or ABI-
  # incompatible CUDA/cuDNN libraries without requiring a camera or model.
  add_executable(cuda_provider_check src/cuda_provider_check.cpp)
  target_compile_features(cuda_provider_check PUBLIC cxx_std_17)
  target_link_libraries(cuda_provider_check onnxruntime)

  # The node lives in lib/<PROJECT_NAME>/ but ORT + CUDA libs install to lib/.
  # Set RPATH so the dynamic linker finds them relative to the binary.
  set_target_properties(encoder_node cuda_provider_check PROPERTIES
    INSTALL_RPATH "$ORIGIN/../"
    BUILD_WITH_INSTALL_RPATH FALSE
  )

  install(TARGETS encoder_node cuda_provider_check
    DESTINATION lib/${PROJECT_NAME})

  # Install only ONNX Runtime itself and its CUDA provider. CUDA and cuDNN remain
  # platform dependencies supplied by the workstation or JetPack.
  file(GLOB ONNXRUNTIME_CORE_LIBRARIES
    "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so*")
  install(FILES
    ${ONNXRUNTIME_CORE_LIBRARIES}
    "${ONNXRUNTIME_CUDA_PROVIDER}"
    "${ONNXRUNTIME_SHARED_PROVIDER}"
    DESTINATION lib)

  # setup_onnxruntime.sh writes an ignored, machine-local hook when CUDA runtime
  # libraries live outside the system loader paths (for example NVIDIA wheels).
  set(VISION_ENCODERS_CUDA_HOOK
    "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/vision_encoders_cuda_runtime.dsv")
  if(EXISTS "${VISION_ENCODERS_CUDA_HOOK}")
    ament_environment_hooks("${VISION_ENCODERS_CUDA_HOOK}")
  endif()
endif()  # VISION_ENCODERS_BUILD_RUNTIME

# frame_wire.hpp is header-only and consumed by other packages (cpp_control's
# the repose planner reads the depth plane off the same wire the encoder reads colour from).
install(DIRECTORY include/ DESTINATION include)
ament_export_include_directories(include)
ament_export_dependencies(rosidl_default_runtime)

install(DIRECTORY config/ DESTINATION share/${PROJECT_NAME}/config)
install(DIRECTORY launch/ DESTINATION share/${PROJECT_NAME}/launch)
install(DIRECTORY scripts/ DESTINATION share/${PROJECT_NAME}/scripts
        USE_SOURCE_PERMISSIONS)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/models")
  install(DIRECTORY models/ DESTINATION share/${PROJECT_NAME}/models)
else()
  message(STATUS "No local encoder artifacts found; skipping models install")
endif()

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
