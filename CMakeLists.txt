cmake_minimum_required(VERSION 3.16)

project(snakelib_estimation)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(pinocchio REQUIRED)
find_package(pybind11 REQUIRED)

# ---------------------------------------------------------------------------
# C++ estimator library
# ---------------------------------------------------------------------------

add_library(snakelib_estimation SHARED
  src/estimator.cpp
  src/kinematics.cpp
)

target_include_directories(snakelib_estimation
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(snakelib_estimation
  PUBLIC
    Eigen3::Eigen
    pinocchio::pinocchio
)

# ---------------------------------------------------------------------------
# Python bindings
# ---------------------------------------------------------------------------

pybind11_add_module(_snakelib_estimation
  src/bindings.cpp
)

target_link_libraries(_snakelib_estimation
  PRIVATE
    snakelib_estimation
    Eigen3::Eigen
    pinocchio::pinocchio
)

target_include_directories(_snakelib_estimation
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# ---------------------------------------------------------------------------
# Install C++ library
# ---------------------------------------------------------------------------

install(
  TARGETS snakelib_estimation
  EXPORT snakelib_estimation_targets
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(
  DIRECTORY include/
  DESTINATION include
)

# ---------------------------------------------------------------------------
# Install Python module
# ---------------------------------------------------------------------------

install(
  TARGETS _snakelib_estimation
  LIBRARY DESTINATION lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages
)

# Python package
install(
  DIRECTORY snakelib_estimation/
  DESTINATION lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages/snakelib_estimation
)

# ---------------------------------------------------------------------------
# PYTHON CONFIGURATION
# ---------------------------------------------------------------------------

# Install the Python Module Folder
ament_python_install_package(${PROJECT_NAME})

# Install Standalone Python Scripts/Executables
install(PROGRAMS
  ${PROJECT_NAME}/state_estimation_node.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_export_dependencies(
  Eigen3
  pinocchio
)

# ---------------------------------------------------------------------------
# TESTING
# ---------------------------------------------------------------------------

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)

  # Automatically runs cpplint, cppcheck, uncrustify, etc.
  ament_lint_auto_find_test_dependencies()
endif()


ament_package()
