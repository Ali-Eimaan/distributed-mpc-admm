# Copyright (c) 2026, Ali-Eimaan. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

cmake_minimum_required(VERSION 3.22...3.31)
project(cpp_admm)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -Wshadow -Wconversion)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(osqp REQUIRED)

# ZeroMQ is optional: the ROS transport is the default and ZeroMqTransport is only needed
# for the standalone benchmark. Guard it so a machine without cppzmq can still build.
#
# Two separate things are required and they ship in different packages: the C library
# (libzmq3-dev) and the header-only C++ binding cppzmq (cppzmq-dev), which is what
# provides <zmq.hpp>. Testing only for libzmq defines CPP_ADMM_WITH_ZMQ on a machine that
# has the library but not the binding, and the build then fails on a missing zmq.hpp.
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
  pkg_check_modules(ZMQ QUIET libzmq)
endif()
find_path(CPPZMQ_INCLUDE_DIR zmq.hpp HINTS ${ZMQ_INCLUDE_DIRS})

if(ZMQ_FOUND AND CPPZMQ_INCLUDE_DIR)
  set(CPP_ADMM_ZMQ_AVAILABLE TRUE)
else()
  set(CPP_ADMM_ZMQ_AVAILABLE FALSE)
endif()

# ---------------------------------------------------------------------------- library

add_library(admm_kernel SHARED
  src/admm_kernel.cpp
  src/per_agent_qp.cpp
)
target_include_directories(admm_kernel PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(admm_kernel PUBLIC Eigen3::Eigen osqp::osqp)

if(CPP_ADMM_ZMQ_AVAILABLE)
  target_compile_definitions(admm_kernel PRIVATE CPP_ADMM_WITH_ZMQ=1)
  target_include_directories(admm_kernel PRIVATE ${ZMQ_INCLUDE_DIRS} ${CPPZMQ_INCLUDE_DIR})
  target_link_libraries(admm_kernel PRIVATE ${ZMQ_LIBRARIES})
  message(STATUS "cppzmq found: ZeroMqTransport enabled")
else()
  message(STATUS
    "cppzmq (<zmq.hpp>) not found: building without ZeroMqTransport; "
    "constructing one will throw. Install cppzmq-dev to enable it.")
endif()

# ------------------------------------------------------------------------------- node

add_executable(consensus_node src/consensus_node.cpp)
target_link_libraries(consensus_node PRIVATE admm_kernel)
target_link_libraries(consensus_node PRIVATE
  ${rclcpp_TARGETS} ${geometry_msgs_TARGETS} ${std_msgs_TARGETS})

# ------------------------------------------------------------------------------ tests

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # Copyright/licence linters are noise for a single-author research repo.
  set(ament_cmake_copyright_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()

  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_admm_kernel test/test_admm_kernel.cpp)
  if(TARGET test_admm_kernel)
    target_link_libraries(test_admm_kernel admm_kernel)
    target_compile_definitions(test_admm_kernel PRIVATE
      CPP_ADMM_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test/data")
    # The parity fixture is JSON, so the gtest needs nlohmann/json.hpp. This is a hard
    # requirement, not an optional extra: A7 (C++/Python parity) is a release gate, and a
    # silently missing header would drop the whole test target rather than fail visibly.
    find_package(nlohmann_json QUIET)
    if(nlohmann_json_FOUND)
      target_link_libraries(test_admm_kernel nlohmann_json::nlohmann_json)
    else()
      find_path(NLOHMANN_JSON_INCLUDE_DIR nlohmann/json.hpp)
      if(NOT NLOHMANN_JSON_INCLUDE_DIR)
        message(FATAL_ERROR
          "nlohmann/json.hpp not found. It is required to build test_admm_kernel, which "
          "carries the C++/Python parity test (acceptance criterion A7). Install it with "
          "`apt-get install nlohmann-json3-dev`, or configure with -DBUILD_TESTING=OFF to "
          "build the library alone.")
      endif()
      target_include_directories(test_admm_kernel PRIVATE ${NLOHMANN_JSON_INCLUDE_DIR})
    endif()
  endif()
endif()

# ---------------------------------------------------------------------------- install

install(
  TARGETS admm_kernel
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
install(TARGETS consensus_node DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})

ament_export_include_directories(include)
ament_export_libraries(admm_kernel)
ament_export_dependencies(rclcpp geometry_msgs std_msgs Eigen3 osqp)
ament_package()
