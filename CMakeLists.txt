cmake_minimum_required(VERSION 3.22)
project(action_go)

include(CTest)
find_package(Python3 REQUIRED COMPONENTS Interpreter)

option(ACTION_GO_BUILD_OMNIXTREME "Build generic_composite_policy_player; TensorRT remains a runtime provider option" ON)
option(ACTION_GO_CODE_ONLY "Do not register tests that require private/BYOW assets" OFF)
option(ACTION_GO_REQUIRE_REALSENSE
  "Fail configuration when librealsense2 is unavailable (recommended for real-robot Instinct builds)"
  OFF
)
set(ACTION_GO_REALSENSE_ROOT "" CACHE PATH
  "librealsense install prefix containing lib/cmake/realsense2/realsense2Config.cmake"
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# For single-config generators (Makefiles/Ninja), make a plain `cmake ..`
# configure a Release build by default. This keeps board-side one-command
# builds fast without needing `-DCMAKE_BUILD_TYPE=Release` every time.
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(unitree_sdk2 REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(ZLIB REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(nlohmann_json 3.10 QUIET CONFIG)
if(TARGET nlohmann_json::nlohmann_json)
  set(ACTION_GO_NLOHMANN_JSON_TARGET nlohmann_json::nlohmann_json)
else()
  find_path(ACTION_GO_NLOHMANN_JSON_INCLUDE_DIR
    NAMES nlohmann/json.hpp
    DOC "Directory containing nlohmann/json.hpp"
  )
  if(NOT ACTION_GO_NLOHMANN_JSON_INCLUDE_DIR)
    message(FATAL_ERROR
      "nlohmann/json.hpp is required. Install nlohmann-json3-dev "
      "(for example: sudo apt install nlohmann-json3-dev), or set "
      "ACTION_GO_NLOHMANN_JSON_INCLUDE_DIR to its include directory."
    )
  endif()
  add_library(action_go_nlohmann_json INTERFACE)
  target_include_directories(action_go_nlohmann_json SYSTEM INTERFACE
    ${ACTION_GO_NLOHMANN_JSON_INCLUDE_DIR}
  )
  set(ACTION_GO_NLOHMANN_JSON_TARGET action_go_nlohmann_json)
endif()
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
  pkg_check_modules(LIBZMQ QUIET libzmq)
  pkg_check_modules(APRILTAG QUIET apriltag)
endif()
find_package(OpenCV QUIET COMPONENTS core imgproc photo)

# A source-built librealsense commonly lives under ~/librealsense/install.
# Discover that install tree, not ~/librealsense/build: an uninstalled build
# config can contain source-machine-relative paths such as /home/include and
# then fail after the repository is moved or rebuilt.
if(NOT realsense2_DIR)
  set(_action_go_realsense_roots)
  if(ACTION_GO_REALSENSE_ROOT)
    list(APPEND _action_go_realsense_roots "${ACTION_GO_REALSENSE_ROOT}")
  endif()
  if(DEFINED ENV{HOME} AND NOT "$ENV{HOME}" STREQUAL "")
    list(APPEND _action_go_realsense_roots "$ENV{HOME}/librealsense/install")
  endif()
  foreach(_action_go_realsense_root IN LISTS _action_go_realsense_roots)
    set(_action_go_realsense_config_dir
      "${_action_go_realsense_root}/lib/cmake/realsense2")
    if(EXISTS "${_action_go_realsense_config_dir}/realsense2Config.cmake")
      set(realsense2_DIR "${_action_go_realsense_config_dir}" CACHE PATH
        "Directory containing realsense2Config.cmake" FORCE)
      message(STATUS
        "Auto-detected librealsense2 install: ${_action_go_realsense_root}")
      break()
    endif()
  endforeach()
  unset(_action_go_realsense_roots)
  unset(_action_go_realsense_root)
  unset(_action_go_realsense_config_dir)
endif()
find_package(realsense2 QUIET)
if(NOT realsense2_FOUND)
  if(PkgConfig_FOUND)
    pkg_check_modules(REALSENSE2 QUIET realsense2)
  endif()
endif()
find_path(APRILTAG_INCLUDE_DIR
  NAMES apriltag.h apriltag/apriltag.h
  PATHS ${APRILTAG_INCLUDE_DIRS}
  PATH_SUFFIXES apriltag
)
find_library(APRILTAG_LIBRARY
  NAMES apriltag
  PATHS ${APRILTAG_LIBRARY_DIRS}
)

add_library(action_go STATIC
  runtime/src/activation_blend.cpp
  runtime/src/async_command_publisher.cpp
  runtime/src/motion_action_controller.cpp
  runtime/src/event_logger.cpp
  runtime/src/supervisor_platform.cpp
  runtime/src/candidate_command_arbiter.cpp
  runtime/src/handoff_protocol.cpp
  runtime/src/network_command.cpp
  runtime/src/reference_endpoint.cpp
)
target_include_directories(action_go
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/include
)
target_link_libraries(action_go
  PUBLIC
    unitree_sdk2
    ddsc
    ddscxx
    rt
    pthread
    OpenSSL::Crypto
    ${ACTION_GO_NLOHMANN_JSON_TARGET}
)


add_executable(action_go_test runtime/apps/action_go_test.cpp)
target_include_directories(action_go_test PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/runtime/include
)
target_link_libraries(action_go_test
  action_go
)

add_executable(action_go_remote_test runtime/apps/action_go_remote_test.cpp)
target_include_directories(action_go_remote_test PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/runtime/include
)
target_link_libraries(action_go_remote_test
  action_go
)

if(BUILD_TESTING)
  add_test(
    NAME simulation_recovery_test
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/runtime/tests/simulation_recovery_test.py
  )
  add_test(
    NAME model_assets_check_test
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/runtime/tests/model_assets_check_test.py
  )
  add_executable(depth_monitor_publisher_test runtime/tests/depth_monitor_publisher_test.cpp)
  target_include_directories(depth_monitor_publisher_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/runtime/include)
  target_compile_options(depth_monitor_publisher_test PRIVATE -UNDEBUG)
  target_link_libraries(depth_monitor_publisher_test PRIVATE pthread)
  add_test(NAME depth_monitor_publisher_test COMMAND depth_monitor_publisher_test)
  add_executable(async_trace_writer_test runtime/tests/async_trace_writer_test.cpp)
  target_include_directories(async_trace_writer_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/runtime/include)
  target_compile_options(async_trace_writer_test PRIVATE -UNDEBUG)
  target_link_libraries(async_trace_writer_test PRIVATE pthread)
  add_test(NAME async_trace_writer_test COMMAND async_trace_writer_test)
  add_executable(bridge_gate_test runtime/tests/bridge_gate_test.cpp)
  target_include_directories(bridge_gate_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/runtime/include)
  add_test(NAME bridge_gate_test COMMAND bridge_gate_test)
  add_test(
    NAME remote_udp_nonloopback_requires_key
    COMMAND action_go_remote_test --udp-listen --udp-bind 0.0.0.0
  )
  set_tests_properties(remote_udp_nonloopback_requires_key PROPERTIES WILL_FAIL TRUE)
endif()

add_executable(action_go_trace_recorder runtime/tools/action_go_trace_recorder.cpp)
target_include_directories(action_go_trace_recorder PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/runtime/include
)
target_link_libraries(action_go_trace_recorder
  action_go
)

add_executable(action_go_command_arbiter runtime/tools/action_go_command_arbiter.cpp)
target_include_directories(action_go_command_arbiter PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/runtime/include
)
target_link_libraries(action_go_command_arbiter
  action_go
  yaml-cpp
)

set(ACTION_GO_CORE_TARGETS
  action_go
  action_go_test
  action_go_remote_test
  action_go_trace_recorder
  action_go_command_arbiter
)
set(ACTION_GO_ONNX_TARGETS)

if(BUILD_TESTING)
  add_executable(fake_policy_payload runtime/tests/fake_policy_payload.cpp)
  add_executable(motion_action_controller_test
    runtime/tests/motion_action_controller_test.cpp
  )
  add_executable(candidate_command_arbiter_test
    runtime/tests/candidate_command_arbiter_test.cpp
  )
  target_include_directories(motion_action_controller_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/include
  )
  target_compile_definitions(motion_action_controller_test PRIVATE
    ACTION_GO_FAKE_PAYLOAD="$<TARGET_FILE:fake_policy_payload>"
  )
  target_link_libraries(motion_action_controller_test action_go)
  target_include_directories(candidate_command_arbiter_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/include
  )
  target_link_libraries(candidate_command_arbiter_test action_go)
  add_dependencies(motion_action_controller_test fake_policy_payload)
  add_test(NAME motion_action_controller_test COMMAND motion_action_controller_test)
  add_test(NAME candidate_command_arbiter_test COMMAND candidate_command_arbiter_test)
  if(Python3_Interpreter_FOUND AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tools/experiments/tests")
    add_test(
      NAME experiment_tools_test
      COMMAND ${Python3_EXECUTABLE} -m unittest discover
              -s ${CMAKE_CURRENT_SOURCE_DIR}/tools/experiments/tests
              -p test_*.py
    )
  endif()
endif()

if(NOT DEFINED ACTION_GO_ONNXRUNTIME_DIR)
  if(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
    set(ACTION_GO_ONNXRUNTIME_DIR "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/onnxruntime-linux-x64-1.22.0")
  elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
    set(ACTION_GO_ONNXRUNTIME_DIR "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/onnxruntime-linux-aarch64-1.22.0")
  endif()
endif()

if(DEFINED ACTION_GO_ONNXRUNTIME_DIR AND EXISTS "${ACTION_GO_ONNXRUNTIME_DIR}/lib/libonnxruntime.so")
  set(ACTION_GO_ONNXRUNTIME_LIB "${ACTION_GO_ONNXRUNTIME_DIR}/lib/libonnxruntime.so")
elseif(DEFINED ACTION_GO_ONNXRUNTIME_DIR)
  file(GLOB ACTION_GO_ONNXRUNTIME_LIB_CANDIDATES
    LIST_DIRECTORIES false
    "${ACTION_GO_ONNXRUNTIME_DIR}/lib/libonnxruntime.so*"
  )
  list(FILTER ACTION_GO_ONNXRUNTIME_LIB_CANDIDATES EXCLUDE REGEX "libonnxruntime_providers_.*\\.so$")
  if(ACTION_GO_ONNXRUNTIME_LIB_CANDIDATES)
    list(SORT ACTION_GO_ONNXRUNTIME_LIB_CANDIDATES COMPARE NATURAL ORDER DESCENDING)
    list(GET ACTION_GO_ONNXRUNTIME_LIB_CANDIDATES 0 ACTION_GO_ONNXRUNTIME_LIB)
  endif()
endif()

if(DEFINED ACTION_GO_ONNXRUNTIME_DIR)
  if(EXISTS "${ACTION_GO_ONNXRUNTIME_DIR}/include/onnxruntime_cxx_api.h")
    set(ACTION_GO_ONNXRUNTIME_INCLUDE_DIR "${ACTION_GO_ONNXRUNTIME_DIR}/include")
  elseif(EXISTS "${ACTION_GO_ONNXRUNTIME_DIR}/include/onnxruntime/core/session/onnxruntime_cxx_api.h")
    set(ACTION_GO_ONNXRUNTIME_INCLUDE_DIR "${ACTION_GO_ONNXRUNTIME_DIR}/include/onnxruntime/core/session")
  endif()
endif()

# The motion player may use a board-specific CUDA/TensorRT ORT build without
# changing the ORT linked by the other generic players and test binaries.
if(NOT DEFINED ACTION_GO_MOTION_ONNXRUNTIME_DIR AND DEFINED ACTION_GO_ONNXRUNTIME_DIR)
  set(ACTION_GO_MOTION_ONNXRUNTIME_DIR "${ACTION_GO_ONNXRUNTIME_DIR}")
endif()
if(DEFINED ACTION_GO_MOTION_ONNXRUNTIME_DIR AND EXISTS "${ACTION_GO_MOTION_ONNXRUNTIME_DIR}/lib/libonnxruntime.so")
  set(ACTION_GO_MOTION_ONNXRUNTIME_LIB "${ACTION_GO_MOTION_ONNXRUNTIME_DIR}/lib/libonnxruntime.so")
elseif(DEFINED ACTION_GO_MOTION_ONNXRUNTIME_DIR)
  file(GLOB ACTION_GO_MOTION_ONNXRUNTIME_LIB_CANDIDATES
    LIST_DIRECTORIES false
    "${ACTION_GO_MOTION_ONNXRUNTIME_DIR}/lib/libonnxruntime.so*"
  )
  list(FILTER ACTION_GO_MOTION_ONNXRUNTIME_LIB_CANDIDATES EXCLUDE REGEX "libonnxruntime_providers_.*\\.so$")
  if(ACTION_GO_MOTION_ONNXRUNTIME_LIB_CANDIDATES)
    list(SORT ACTION_GO_MOTION_ONNXRUNTIME_LIB_CANDIDATES COMPARE NATURAL ORDER DESCENDING)
    list(GET ACTION_GO_MOTION_ONNXRUNTIME_LIB_CANDIDATES 0 ACTION_GO_MOTION_ONNXRUNTIME_LIB)
  endif()
endif()
if(DEFINED ACTION_GO_MOTION_ONNXRUNTIME_DIR)
  if(EXISTS "${ACTION_GO_MOTION_ONNXRUNTIME_DIR}/include/onnxruntime_cxx_api.h")
    set(ACTION_GO_MOTION_ONNXRUNTIME_INCLUDE_DIR "${ACTION_GO_MOTION_ONNXRUNTIME_DIR}/include")
  elseif(EXISTS "${ACTION_GO_MOTION_ONNXRUNTIME_DIR}/include/onnxruntime/core/session/onnxruntime_cxx_api.h")
    set(ACTION_GO_MOTION_ONNXRUNTIME_INCLUDE_DIR "${ACTION_GO_MOTION_ONNXRUNTIME_DIR}/include/onnxruntime/core/session")
  endif()
endif()

# Composite policies such as OmniXtreme may use a TensorRT-enabled ORT build
# without changing the regular and motion-policy executables in this build
# tree.  Keeping this as a separate imported library also prevents a later
# rebuild of generic_composite_policy_player from silently reverting it to the
# bundled CPU-only ORT.
if(NOT DEFINED ACTION_GO_COMPOSITE_ONNXRUNTIME_DIR AND DEFINED ACTION_GO_ONNXRUNTIME_DIR)
  set(ACTION_GO_COMPOSITE_ONNXRUNTIME_DIR "${ACTION_GO_ONNXRUNTIME_DIR}")
endif()
if(DEFINED ACTION_GO_COMPOSITE_ONNXRUNTIME_DIR AND EXISTS "${ACTION_GO_COMPOSITE_ONNXRUNTIME_DIR}/lib/libonnxruntime.so")
  set(ACTION_GO_COMPOSITE_ONNXRUNTIME_LIB "${ACTION_GO_COMPOSITE_ONNXRUNTIME_DIR}/lib/libonnxruntime.so")
elseif(DEFINED ACTION_GO_COMPOSITE_ONNXRUNTIME_DIR)
  file(GLOB ACTION_GO_COMPOSITE_ONNXRUNTIME_LIB_CANDIDATES
    LIST_DIRECTORIES false
    "${ACTION_GO_COMPOSITE_ONNXRUNTIME_DIR}/lib/libonnxruntime.so*"
  )
  list(FILTER ACTION_GO_COMPOSITE_ONNXRUNTIME_LIB_CANDIDATES EXCLUDE REGEX "libonnxruntime_providers_.*\\.so$")
  if(ACTION_GO_COMPOSITE_ONNXRUNTIME_LIB_CANDIDATES)
    list(SORT ACTION_GO_COMPOSITE_ONNXRUNTIME_LIB_CANDIDATES COMPARE NATURAL ORDER DESCENDING)
    list(GET ACTION_GO_COMPOSITE_ONNXRUNTIME_LIB_CANDIDATES 0 ACTION_GO_COMPOSITE_ONNXRUNTIME_LIB)
  endif()
endif()
if(DEFINED ACTION_GO_COMPOSITE_ONNXRUNTIME_DIR)
  if(EXISTS "${ACTION_GO_COMPOSITE_ONNXRUNTIME_DIR}/include/onnxruntime_cxx_api.h")
    set(ACTION_GO_COMPOSITE_ONNXRUNTIME_INCLUDE_DIR "${ACTION_GO_COMPOSITE_ONNXRUNTIME_DIR}/include")
  elseif(EXISTS "${ACTION_GO_COMPOSITE_ONNXRUNTIME_DIR}/include/onnxruntime/core/session/onnxruntime_cxx_api.h")
    set(ACTION_GO_COMPOSITE_ONNXRUNTIME_INCLUDE_DIR "${ACTION_GO_COMPOSITE_ONNXRUNTIME_DIR}/include/onnxruntime/core/session")
  endif()
endif()

set(ACTION_GO_MUJOCO_FOUND OFF)
set(ACTION_GO_MUJOCO_INCLUDE_DIR "")
set(ACTION_GO_MUJOCO_LIB "")
if(DEFINED ACTION_GO_MUJOCO_DIR)
  if(EXISTS "${ACTION_GO_MUJOCO_DIR}/include/mujoco/mujoco.h")
    set(ACTION_GO_MUJOCO_INCLUDE_DIR "${ACTION_GO_MUJOCO_DIR}/include")
  endif()
  file(GLOB ACTION_GO_MUJOCO_LIB_CANDIDATES
    LIST_DIRECTORIES false
    "${ACTION_GO_MUJOCO_DIR}/libmujoco.so*"
    "${ACTION_GO_MUJOCO_DIR}/lib/libmujoco.so*"
  )
else()
  execute_process(
    COMMAND python3 -c "import mujoco, pathlib; print(pathlib.Path(mujoco.__file__).resolve().parent)"
    OUTPUT_VARIABLE ACTION_GO_PY_MUJOCO_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
    RESULT_VARIABLE ACTION_GO_PY_MUJOCO_RESULT
  )
  if(ACTION_GO_PY_MUJOCO_RESULT EQUAL 0 AND EXISTS "${ACTION_GO_PY_MUJOCO_DIR}/include/mujoco/mujoco.h")
    set(ACTION_GO_MUJOCO_INCLUDE_DIR "${ACTION_GO_PY_MUJOCO_DIR}/include")
    file(GLOB ACTION_GO_MUJOCO_LIB_CANDIDATES
      LIST_DIRECTORIES false
      "${ACTION_GO_PY_MUJOCO_DIR}/libmujoco.so*"
      "${ACTION_GO_PY_MUJOCO_DIR}/lib/libmujoco.so*"
    )
  endif()
endif()
if(ACTION_GO_MUJOCO_INCLUDE_DIR AND ACTION_GO_MUJOCO_LIB_CANDIDATES)
  list(SORT ACTION_GO_MUJOCO_LIB_CANDIDATES COMPARE NATURAL ORDER DESCENDING)
  list(GET ACTION_GO_MUJOCO_LIB_CANDIDATES 0 ACTION_GO_MUJOCO_LIB)
  get_filename_component(ACTION_GO_MUJOCO_LIB_DIR "${ACTION_GO_MUJOCO_LIB}" DIRECTORY)
  set(ACTION_GO_MUJOCO_FOUND ON)
  message(STATUS "MuJoCo enabled for generic_motion_tracking_player: ${ACTION_GO_MUJOCO_LIB}")
else()
  message(STATUS "MuJoCo not found; Humanoid-GPT realtime qpos36 FK will report a runtime error, offline cache remains supported")
endif()

if(DEFINED ACTION_GO_ONNXRUNTIME_LIB AND DEFINED ACTION_GO_ONNXRUNTIME_INCLUDE_DIR)
  add_library(motion_action_onnxruntime SHARED IMPORTED GLOBAL)
  set_target_properties(motion_action_onnxruntime PROPERTIES
    IMPORTED_LOCATION "${ACTION_GO_ONNXRUNTIME_LIB}"
    INTERFACE_INCLUDE_DIRECTORIES "${ACTION_GO_ONNXRUNTIME_INCLUDE_DIR}"
  )
  message(STATUS "ACTION_GO_ONNXRUNTIME_DIR=${ACTION_GO_ONNXRUNTIME_DIR}")
  message(STATUS "ACTION_GO_ONNXRUNTIME_LIB=${ACTION_GO_ONNXRUNTIME_LIB}")
  message(STATUS "ACTION_GO_ONNXRUNTIME_INCLUDE_DIR=${ACTION_GO_ONNXRUNTIME_INCLUDE_DIR}")
  if(NOT DEFINED ACTION_GO_MOTION_ONNXRUNTIME_LIB OR NOT DEFINED ACTION_GO_MOTION_ONNXRUNTIME_INCLUDE_DIR)
    message(FATAL_ERROR "ACTION_GO_MOTION_ONNXRUNTIME_DIR does not contain a usable ONNX Runtime")
  endif()
  add_library(action_go_motion_onnxruntime SHARED IMPORTED GLOBAL)
  set_target_properties(action_go_motion_onnxruntime PROPERTIES
    IMPORTED_LOCATION "${ACTION_GO_MOTION_ONNXRUNTIME_LIB}"
    INTERFACE_INCLUDE_DIRECTORIES "${ACTION_GO_MOTION_ONNXRUNTIME_INCLUDE_DIR}"
  )
  message(STATUS "ACTION_GO_MOTION_ONNXRUNTIME_DIR=${ACTION_GO_MOTION_ONNXRUNTIME_DIR}")
  if(NOT DEFINED ACTION_GO_COMPOSITE_ONNXRUNTIME_LIB OR NOT DEFINED ACTION_GO_COMPOSITE_ONNXRUNTIME_INCLUDE_DIR)
    message(FATAL_ERROR "ACTION_GO_COMPOSITE_ONNXRUNTIME_DIR does not contain a usable ONNX Runtime")
  endif()
  add_library(action_go_composite_onnxruntime SHARED IMPORTED GLOBAL)
  set_target_properties(action_go_composite_onnxruntime PROPERTIES
    IMPORTED_LOCATION "${ACTION_GO_COMPOSITE_ONNXRUNTIME_LIB}"
    INTERFACE_INCLUDE_DIRECTORIES "${ACTION_GO_COMPOSITE_ONNXRUNTIME_INCLUDE_DIR}"
  )
  message(STATUS "ACTION_GO_COMPOSITE_ONNXRUNTIME_DIR=${ACTION_GO_COMPOSITE_ONNXRUNTIME_DIR}")

  target_sources(action_go_remote_test PRIVATE runtime/src/onnx_runner.cpp)
  target_sources(action_go_remote_test PRIVATE runtime/src/bridge_risk_monitor.cpp)
  target_include_directories(action_go_remote_test PRIVATE ${ACTION_GO_ONNXRUNTIME_INCLUDE_DIR})
  target_compile_definitions(action_go_remote_test PRIVATE ACTION_GO_HAS_ONNX=1)
  target_link_libraries(action_go_remote_test motion_action_onnxruntime)
  set_target_properties(action_go_remote_test PROPERTIES
    BUILD_RPATH "${ACTION_GO_ONNXRUNTIME_DIR}/lib"
    INSTALL_RPATH "$ORIGIN;$ORIGIN/../lib;${ACTION_GO_ONNXRUNTIME_DIR}/lib")

  target_compile_definitions(action_go_command_arbiter PRIVATE ACTION_GO_HAS_ONNX=1)
  target_include_directories(action_go_command_arbiter PRIVATE
    ${ACTION_GO_ONNXRUNTIME_INCLUDE_DIR}
  )
  target_link_libraries(action_go_command_arbiter motion_action_onnxruntime)
  set_target_properties(action_go_command_arbiter PROPERTIES
    BUILD_RPATH "${ACTION_GO_ONNXRUNTIME_DIR}/lib"
    INSTALL_RPATH "$ORIGIN;$ORIGIN/../lib;${ACTION_GO_ONNXRUNTIME_DIR}/lib"
  )

  set(ACTION_GO_BFM_SOURCES
    runtime/src/bfm_context_stream.cpp
    runtime/src/bfm_observation.cpp
    runtime/src/bfm_action.cpp
    runtime/src/bfm_context.cpp
    runtime/src/pbfm_policy.cpp
    runtime/src/depth_image_preprocessor.cpp
    runtime/src/realsense_depth_stream.cpp
  )

  add_executable(generic_policy_player
    runtime/players/generic_policy_player.cpp
    ${ACTION_GO_BFM_SOURCES}
    runtime/src/generic_policy_module_pbfm.cpp
    runtime/src/generic_policy_runtime.cpp
    runtime/src/generic_policy_module_self.cpp
    runtime/src/generic_policy_module_mimic.cpp
    runtime/src/generic_policy_module_motion.cpp
    runtime/src/generic_policy_module_supermimic.cpp
    runtime/src/policy_contract.cpp
    runtime/src/handoff_protocol.cpp
    runtime/src/policy_protocol.cpp
    runtime/src/event_logger.cpp
    runtime/src/command_safety_filter.cpp
    runtime/src/activation_blend.cpp
    runtime/src/async_command_publisher.cpp
    runtime/src/input_freshness.cpp
    runtime/src/onnx_runner.cpp
    runtime/src/reference_endpoint.cpp
    runtime/src/sonic_policy.cpp
    runtime/src/sonic_planner.cpp
  )
  target_include_directories(generic_policy_player PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/include
    ${ACTION_GO_ONNXRUNTIME_INCLUDE_DIR}
  )
  target_link_libraries(generic_policy_player
    unitree_sdk2
    ddsc
    ddscxx
    rt
    pthread
    yaml-cpp
    ZLIB::ZLIB
    motion_action_onnxruntime
    ${ACTION_GO_NLOHMANN_JSON_TARGET}
  )
  set_target_properties(generic_policy_player PROPERTIES
    BUILD_RPATH "${ACTION_GO_ONNXRUNTIME_DIR}/lib"
    INSTALL_RPATH "$ORIGIN;$ORIGIN/../lib;${ACTION_GO_ONNXRUNTIME_DIR}/lib"
  )
  list(APPEND ACTION_GO_ONNX_TARGETS generic_policy_player)

  add_executable(generic_motion_tracking_player
    runtime/players/generic_motion_tracking_player.cpp
    ${ACTION_GO_BFM_SOURCES}
    runtime/src/generic_policy_module_pbfm.cpp
    runtime/src/generic_policy_runtime.cpp
    runtime/src/generic_policy_module_self.cpp
    runtime/src/generic_policy_module_mimic.cpp
    runtime/src/generic_policy_module_motion.cpp
    runtime/src/generic_policy_module_supermimic.cpp
    runtime/src/policy_contract.cpp
    runtime/src/handoff_protocol.cpp
    runtime/src/policy_protocol.cpp
    runtime/src/event_logger.cpp
    runtime/src/command_safety_filter.cpp
    runtime/src/activation_blend.cpp
    runtime/src/async_command_publisher.cpp
    runtime/src/input_freshness.cpp
    runtime/src/onnx_runner.cpp
    runtime/src/reference_endpoint.cpp
    runtime/src/sonic_policy.cpp
    runtime/src/sonic_planner.cpp
  )
  target_include_directories(generic_motion_tracking_player PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/include
    ${ACTION_GO_MOTION_ONNXRUNTIME_INCLUDE_DIR}
  )
  target_link_libraries(generic_motion_tracking_player
    unitree_sdk2
    ddsc
    ddscxx
    rt
    pthread
    yaml-cpp
    ZLIB::ZLIB
    action_go_motion_onnxruntime
    ${ACTION_GO_NLOHMANN_JSON_TARGET}
  )
  if(LIBZMQ_FOUND)
    target_sources(generic_policy_player PRIVATE runtime/src/shared_memory_depth_stream.cpp)
    target_compile_definitions(generic_policy_player PRIVATE ACTION_GO_HAS_ZMQ=1)
    target_include_directories(generic_policy_player PRIVATE ${LIBZMQ_INCLUDE_DIRS})
    target_link_libraries(generic_policy_player ${LIBZMQ_LIBRARIES})
    target_sources(generic_motion_tracking_player PRIVATE runtime/src/shared_memory_depth_stream.cpp)
    target_compile_definitions(generic_motion_tracking_player PRIVATE ACTION_GO_HAS_ZMQ=1)
    target_include_directories(generic_motion_tracking_player PRIVATE ${LIBZMQ_INCLUDE_DIRS})
    target_link_libraries(generic_motion_tracking_player ${LIBZMQ_LIBRARIES})
  endif()
  if(ACTION_GO_MUJOCO_FOUND)
    target_compile_definitions(generic_motion_tracking_player PRIVATE ACTION_GO_HAS_MUJOCO=1)
    target_include_directories(generic_motion_tracking_player PRIVATE ${ACTION_GO_MUJOCO_INCLUDE_DIR})
    target_link_libraries(generic_motion_tracking_player ${ACTION_GO_MUJOCO_LIB})
  endif()
  set_target_properties(generic_motion_tracking_player PROPERTIES
    BUILD_RPATH "${ACTION_GO_MOTION_ONNXRUNTIME_DIR}/lib"
    INSTALL_RPATH "$ORIGIN;$ORIGIN/../lib;${ACTION_GO_MOTION_ONNXRUNTIME_DIR}/lib"
  )
  if(ACTION_GO_MUJOCO_FOUND)
    set_property(TARGET generic_motion_tracking_player APPEND PROPERTY BUILD_RPATH "${ACTION_GO_MUJOCO_LIB_DIR}")
    set_property(TARGET generic_motion_tracking_player APPEND PROPERTY INSTALL_RPATH "${ACTION_GO_MUJOCO_LIB_DIR}")
  endif()
  list(APPEND ACTION_GO_ONNX_TARGETS generic_motion_tracking_player)

  add_executable(runtime_hardening_test
    runtime/tests/runtime_hardening_test.cpp
    runtime/src/activation_blend.cpp
    runtime/src/async_command_publisher.cpp
    runtime/src/policy_contract.cpp
    runtime/src/policy_protocol.cpp
    runtime/src/command_safety_filter.cpp
    runtime/src/input_freshness.cpp
    runtime/src/onnx_runner.cpp
    runtime/src/sonic_policy.cpp
    runtime/src/sonic_planner.cpp
    runtime/src/network_command.cpp
    runtime/src/reference_endpoint.cpp
  )
  target_include_directories(runtime_hardening_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/include
    ${ACTION_GO_ONNXRUNTIME_INCLUDE_DIR}
  )
  set(ACTION_GO_SAFE_TEST_MODEL "${CMAKE_CURRENT_BINARY_DIR}/fixtures/actiongo_safe_fixture.onnx")
  add_custom_command(
    OUTPUT "${ACTION_GO_SAFE_TEST_MODEL}"
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/runtime/tests/create_safe_onnx_fixture.py
            "${ACTION_GO_SAFE_TEST_MODEL}"
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/runtime/tests/create_safe_onnx_fixture.py
    VERBATIM
  )
  add_custom_target(action_go_safe_test_model DEPENDS "${ACTION_GO_SAFE_TEST_MODEL}")
  add_dependencies(runtime_hardening_test action_go_safe_test_model)
  target_compile_definitions(runtime_hardening_test PRIVATE
    ACTION_GO_TEST_ONNX_MODEL="${ACTION_GO_SAFE_TEST_MODEL}"
  )
  target_link_libraries(runtime_hardening_test
    pthread
    OpenSSL::Crypto
    motion_action_onnxruntime
    ${ACTION_GO_NLOHMANN_JSON_TARGET}
  )
  set_target_properties(runtime_hardening_test PROPERTIES
    BUILD_RPATH "${ACTION_GO_ONNXRUNTIME_DIR}/lib"
    INSTALL_RPATH "$ORIGIN;$ORIGIN/../lib;${ACTION_GO_ONNXRUNTIME_DIR}/lib"
  )
  if(BUILD_TESTING)
    add_test(NAME runtime_hardening_test COMMAND runtime_hardening_test)
  endif()

  add_executable(policy_spec_schema_check
    runtime/tools/policy_spec_schema_check.cpp
    runtime/src/bfm_context.cpp
    runtime/src/pbfm_policy.cpp
    runtime/src/depth_image_preprocessor.cpp
    runtime/src/policy_contract.cpp
    runtime/src/policy_protocol.cpp
    runtime/src/onnx_runner.cpp
  )
  target_include_directories(policy_spec_schema_check PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/include
    ${ACTION_GO_ONNXRUNTIME_INCLUDE_DIR}
  )
  target_link_libraries(policy_spec_schema_check
    yaml-cpp
    motion_action_onnxruntime
  )
  set_target_properties(policy_spec_schema_check PROPERTIES
    BUILD_RPATH "${ACTION_GO_ONNXRUNTIME_DIR}/lib"
    INSTALL_RPATH "$ORIGIN;$ORIGIN/../lib;${ACTION_GO_ONNXRUNTIME_DIR}/lib"
  )
  list(APPEND ACTION_GO_ONNX_TARGETS policy_spec_schema_check)
  if(BUILD_TESTING)
    add_test(
      NAME policy_spec_schema_check_adapters
      COMMAND policy_spec_schema_check --all --allow-missing-assets
    )
    set_tests_properties(policy_spec_schema_check_adapters PROPERTIES
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
    if(NOT ACTION_GO_CODE_ONLY AND
       EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/actions/fighting154k/model/supermimic_omni.onnx" AND
       EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/actions/fighting154k/model/supermimic_no_omni.onnx" AND
       EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/actions/fighting154k/model/allmimic_omni.onnx" AND
       EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/actions/fighting154k/model/allmimic_no_omni.onnx")
      add_test(NAME fighting154k_deployment_test COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/runtime/tests/fighting154k_deployment_test.py
        --player $<TARGET_FILE:generic_motion_tracking_player>)
      set_tests_properties(fighting154k_deployment_test PROPERTIES TIMEOUT 120)
    endif()
    if(NOT ACTION_GO_CODE_ONLY AND
       EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/actions/allmimic/model/policy.onnx" AND
       EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/actions/allmimic/model/policy_no_omni.onnx")
      add_test(NAME allmimic_dry_run_test COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/runtime/tests/allmimic_dry_run_test.py
        --player $<TARGET_FILE:generic_motion_tracking_player>)
      set_tests_properties(allmimic_dry_run_test PROPERTIES TIMEOUT 60)
    endif()
    if(NOT ACTION_GO_CODE_ONLY AND
       EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/actions/mjlab_combat/model/policy_92000.onnx")
      add_test(NAME mjlab_combat_deployment_test COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/runtime/tests/mjlab_combat_deployment_test.py
        --player $<TARGET_FILE:generic_motion_tracking_player>)
      set_tests_properties(mjlab_combat_deployment_test PROPERTIES TIMEOUT 90)
    endif()
    if(NOT ACTION_GO_CODE_ONLY AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/actions/bfm0/model/FBcprAuxModel.onnx")
      add_test(
        NAME policy_spec_schema_check_all
        COMMAND policy_spec_schema_check --all
      )
      set_tests_properties(policy_spec_schema_check_all PROPERTIES
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      )
      add_test(
        NAME generic_policy_offline_dry_run
        COMMAND generic_policy_player
                --spec actions/bfm0/config/generic_bfm_zero.yaml
                --duration-sec 0.05
                --dry-run
                --offline-dry-run-state
      )
      set_tests_properties(generic_policy_offline_dry_run PROPERTIES
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        TIMEOUT 30
      )
    endif()
    if(NOT ACTION_GO_CODE_ONLY AND
       EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/actions/bfm0/model/FBcprAuxModel.onnx" AND
       EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/actions/beyondmimic/model/kick_right.onnx" AND
       EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/actions/protomotions/model/unified_pipeline.onnx")
      add_test(
        NAME generic_preview_integration
        COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/runtime/tests/generic_preview_integration.py
                --repo ${CMAKE_CURRENT_SOURCE_DIR}
                --policy-player $<TARGET_FILE:generic_policy_player>
                --motion-player $<TARGET_FILE:generic_motion_tracking_player>
      )
      set_tests_properties(generic_preview_integration PROPERTIES TIMEOUT 90)
    endif()
    if(NOT ACTION_GO_CODE_ONLY AND
       EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tools/experiments/failure_injection.py" AND
       EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/actions/bfm0/model/FBcprAuxModel.onnx" AND
       EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/actions/twist/model/twist_general_motion_tracker.onnx" AND
       EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/actions/protomotions/model/unified_pipeline.onnx")
      add_test(
        NAME failure_injection_quick
        COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/tools/experiments/failure_injection.py
                --repo ${CMAKE_CURRENT_SOURCE_DIR}
                --build-dir ${CMAKE_CURRENT_BINARY_DIR}
                --quick
                --out ${CMAKE_CURRENT_BINARY_DIR}/failure_injection_test.jsonl
      )
      set_tests_properties(failure_injection_quick PROPERTIES TIMEOUT 180)
    endif()
  endif()

  if(NOT ACTION_GO_BUILD_OMNIXTREME)
    message(STATUS "ACTION_GO_BUILD_OMNIXTREME=OFF is deprecated; generic_composite_policy_player still builds by default")
  endif()
  add_executable(generic_composite_policy_player
    runtime/players/generic_composite_policy_player.cpp
    runtime/src/policy_contract.cpp
    runtime/src/policy_protocol.cpp
    runtime/src/event_logger.cpp
    runtime/src/omnixtreme_runtime.cpp
    runtime/src/command_safety_filter.cpp
    runtime/src/activation_blend.cpp
    runtime/src/async_command_publisher.cpp
  )
  target_include_directories(generic_composite_policy_player PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime/include
    ${ACTION_GO_COMPOSITE_ONNXRUNTIME_INCLUDE_DIR}
  )
  target_link_libraries(generic_composite_policy_player
    unitree_sdk2
    ddsc
    ddscxx
    rt
    pthread
    yaml-cpp
    ZLIB::ZLIB
    action_go_composite_onnxruntime
  )
  set_target_properties(generic_composite_policy_player PROPERTIES
    BUILD_RPATH "${ACTION_GO_COMPOSITE_ONNXRUNTIME_DIR}/lib"
    INSTALL_RPATH "$ORIGIN;$ORIGIN/../lib;${ACTION_GO_COMPOSITE_ONNXRUNTIME_DIR}/lib"
  )
  list(APPEND ACTION_GO_ONNX_TARGETS generic_composite_policy_player)

  add_executable(depth_image_preprocess runtime/tools/depth_image_preprocess.cpp runtime/src/depth_image_preprocessor.cpp)
  target_include_directories(depth_image_preprocess PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/runtime/include)

  if(OpenCV_FOUND)
    if(NOT LIBZMQ_FOUND)
      message(FATAL_ERROR "generic_sensor_policy_player requires libzmq (install libzmq3-dev and pkg-config)")
    endif()
  add_executable(generic_sensor_policy_player
      runtime/players/generic_sensor_policy_player.cpp
      ${ACTION_GO_BFM_SOURCES}
      runtime/src/generic_policy_runtime.cpp
      runtime/src/generic_policy_module_self.cpp
      runtime/src/generic_policy_module_pbfm.cpp
      runtime/src/generic_policy_module_mimic.cpp
      runtime/src/generic_policy_module_motion.cpp
      runtime/src/generic_policy_module_supermimic.cpp
      runtime/src/handoff_protocol.cpp
      runtime/src/input_freshness.cpp
      runtime/src/onnx_runner.cpp
      runtime/src/sonic_policy.cpp
      runtime/src/sonic_planner.cpp
      runtime/src/policy_contract.cpp
      runtime/src/policy_protocol.cpp
      runtime/src/event_logger.cpp
      runtime/src/instinct_parkour_runtime.cpp
      runtime/src/shared_memory_depth_stream.cpp
      runtime/src/steadytray_runtime.cpp
      runtime/src/command_safety_filter.cpp
      runtime/src/activation_blend.cpp
      runtime/src/async_command_publisher.cpp
      runtime/src/reference_endpoint.cpp
    )
    target_include_directories(generic_sensor_policy_player PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/runtime/include
      ${ACTION_GO_ONNXRUNTIME_INCLUDE_DIR}
      ${OpenCV_INCLUDE_DIRS}
      ${LIBZMQ_INCLUDE_DIRS}
    )
    target_link_libraries(generic_sensor_policy_player
      unitree_sdk2
      ddsc
      ddscxx
      rt
      pthread
      yaml-cpp
      motion_action_onnxruntime
      ${OpenCV_LIBS}
      ${LIBZMQ_LIBRARIES}
      ZLIB::ZLIB
      ${ACTION_GO_NLOHMANN_JSON_TARGET}
    )
    target_compile_definitions(generic_sensor_policy_player PRIVATE ACTION_GO_HAS_ZMQ=1)
    if(realsense2_FOUND)
      target_compile_definitions(generic_sensor_policy_player PRIVATE ACTION_GO_WITH_REALSENSE)
      if(TARGET realsense2::realsense2)
        target_link_libraries(generic_sensor_policy_player realsense2::realsense2)
      else()
        target_link_libraries(generic_sensor_policy_player realsense2)
      endif()
      message(STATUS "generic_sensor_policy_player: librealsense2 enabled")
    elseif(REALSENSE2_FOUND)
      target_compile_definitions(generic_sensor_policy_player PRIVATE ACTION_GO_WITH_REALSENSE)
      target_include_directories(generic_sensor_policy_player PRIVATE ${REALSENSE2_INCLUDE_DIRS})
      target_compile_options(generic_sensor_policy_player PRIVATE ${REALSENSE2_CFLAGS_OTHER})
      target_link_libraries(generic_sensor_policy_player ${REALSENSE2_LDFLAGS})
      message(STATUS "generic_sensor_policy_player: librealsense2 enabled via pkg-config")
    else()
      if(ACTION_GO_REQUIRE_REALSENSE)
        message(FATAL_ERROR
          "librealsense2 is required for this build but was not found. "
          "Install it under $ENV{HOME}/librealsense/install, or pass "
          "-DACTION_GO_REALSENSE_ROOT=/absolute/install/prefix (preferred) "
          "or -Drealsense2_DIR=/absolute/install/lib/cmake/realsense2.")
      else()
        message(WARNING "librealsense2 not found; generic_sensor_policy_player builds, but RealSense modes will fail until librealsense2-dev is installed. For a real-robot Instinct build, reconfigure with -DACTION_GO_REQUIRE_REALSENSE=ON so this cannot fail silently")
      endif()
    endif()
    if(APRILTAG_LIBRARY AND (APRILTAG_INCLUDE_DIR OR APRILTAG_INCLUDE_DIRS))
      target_compile_definitions(generic_sensor_policy_player PRIVATE ACTION_GO_WITH_APRILTAG)
      if(APRILTAG_INCLUDE_DIR)
        target_include_directories(generic_sensor_policy_player PRIVATE ${APRILTAG_INCLUDE_DIR})
      endif()
      if(APRILTAG_INCLUDE_DIRS)
        target_include_directories(generic_sensor_policy_player PRIVATE ${APRILTAG_INCLUDE_DIRS})
      endif()
      target_link_libraries(generic_sensor_policy_player ${APRILTAG_LIBRARY})
      message(STATUS "generic_sensor_policy_player: AprilTag enabled")
    else()
      message(WARNING "apriltag C library not found; generic_sensor_policy_player builds, but --object-source apriltag will fail. Use --object-source udp/zero or install AprilRobotics apriltag and rebuild")
    endif()
    set_target_properties(generic_sensor_policy_player PROPERTIES
      BUILD_RPATH "${ACTION_GO_ONNXRUNTIME_DIR}/lib"
      INSTALL_RPATH "$ORIGIN;$ORIGIN/../lib;${ACTION_GO_ONNXRUNTIME_DIR}/lib"
    )
    list(APPEND ACTION_GO_ONNX_TARGETS generic_sensor_policy_player)
    if(BUILD_TESTING AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/actions/pbfm/model/policy.onnx")
      add_test(NAME pbfm_lifecycle_test COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/runtime/tests/pbfm_lifecycle_test.py
        --player $<TARGET_FILE:generic_sensor_policy_player>)
      set_tests_properties(pbfm_lifecycle_test PROPERTIES TIMEOUT 90)
      add_test(NAME pbfm_realtime_test COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/runtime/tests/pbfm_realtime_test.py
        --player $<TARGET_FILE:generic_sensor_policy_player>)
      set_tests_properties(pbfm_realtime_test PROPERTIES TIMEOUT 45)
    endif()
  else()
    message(WARNING "OpenCV not found; generic_sensor_policy_player target is disabled")
  endif()
else()
  message(WARNING "ONNX Runtime not found; ONNX player targets are disabled. Run tools/deps/fetch_onnxruntime.sh or set -DACTION_GO_ONNXRUNTIME_DIR to a directory containing lib/libonnxruntime.so* and include/onnxruntime_cxx_api.h")
endif()

set(ACTION_GO_ALL_TARGETS ${ACTION_GO_CORE_TARGETS} ${ACTION_GO_ONNX_TARGETS})
add_custom_target(action_go_all ALL
  DEPENDS ${ACTION_GO_ALL_TARGETS}
)
message(STATUS "action_go default ALL targets: ${ACTION_GO_ALL_TARGETS}")

if(BUILD_TESTING AND TARGET generic_policy_player)
    add_executable(bfm_policy_test runtime/tests/bfm_policy_test.cpp
      ${ACTION_GO_BFM_SOURCES} runtime/src/policy_protocol.cpp runtime/src/onnx_runner.cpp)
    target_include_directories(bfm_policy_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/runtime/include ${ACTION_GO_ONNXRUNTIME_INCLUDE_DIR})
    target_link_libraries(bfm_policy_test yaml-cpp motion_action_onnxruntime ${ACTION_GO_NLOHMANN_JSON_TARGET})
    set_target_properties(bfm_policy_test PROPERTIES BUILD_RPATH "${ACTION_GO_ONNXRUNTIME_DIR}/lib")
    add_test(NAME bfm_policy_test COMMAND bfm_policy_test)
    add_test(NAME pbfm_depth_test COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/runtime/tests/pbfm_depth_test.py
      --preprocessor $<TARGET_FILE:depth_image_preprocess>)
    add_test(NAME pbfm_keyboard_test COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/runtime/tests/pbfm_keyboard_test.py)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/actions/pbfm/model/policy.onnx" AND
       EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/actions/bfm0/model/FBcprAuxModel.onnx" AND
       EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/actions/ufo/model/FBcprAuxModel.onnx")
      add_test(NAME pbfm_onnx_parity COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/runtime/tests/pbfm_onnx_parity.py --test-binary $<TARGET_FILE:bfm_policy_test>)
      set_tests_properties(pbfm_onnx_parity PROPERTIES TIMEOUT 90)
    endif()
endif()
