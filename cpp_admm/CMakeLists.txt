cmake_minimum_required(VERSION 3.16)
project(cpp_admm)

# TODO [GUIDE 6.1]: this file is complete enough to configure; verify it builds once the
# sources compile, and delete this comment.

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
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
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
  pkg_check_modules(ZMQ QUIET libzmq)
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

if(ZMQ_FOUND)
  target_compile_definitions(admm_kernel PRIVATE CPP_ADMM_WITH_ZMQ=1)
  target_include_directories(admm_kernel PRIVATE ${ZMQ_INCLUDE_DIRS})
  target_link_libraries(admm_kernel PRIVATE ${ZMQ_LIBRARIES})
else()
  message(STATUS "libzmq not found: ZeroMqTransport will throw at construction")
endif()

# ------------------------------------------------------------------------------- node

add_executable(consensus_node src/consensus_node.cpp)
target_link_libraries(consensus_node PRIVATE admm_kernel)
ament_target_dependencies(consensus_node rclcpp geometry_msgs std_msgs)

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
