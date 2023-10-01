cmake_minimum_required(VERSION 3.20)

project(FractalRendering)

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)

# Add external modules
set(EIGEN_ROOT_DIR ${CMAKE_SOURCE_DIR}/external/eigen)
set(PYBIND_ROOT_DIR ${CMAKE_SOURCE_DIR}/external/pybind11)

# Add external modules
add_subdirectory(${PYBIND_ROOT_DIR})
include_directories(${EIGEN_ROOT_DIR})

# Find OpenMP module
find_package(OpenMP REQUIRED)

# Set module name
set(MODULE_NAME mandelbrot_cpp)

# Specify the configuration file from which the header file will be generated
configure_file(config.h.in config.h @ONLY)
include_directories("${PROJECT_BINARY_DIR}")

# Create a python module 
set(HEADER_FILES mandelbrot.h)
set(SOURCE_FILES mandelbrot.cpp)

# Add a pybind11 module
pybind11_add_module(${MODULE_NAME} MODULE python_wrappers.cpp "${HEADER_FILES}" "${SOURCE_FILES}")
target_link_libraries(${MODULE_NAME} PRIVATE OpenMP::OpenMP_CXX)
#target_compile_options(${MODULE_NAME} PRIVATE -O3 -march=native)


# Create an executable binary file 
#add_executable("${PROJECT_NAME}" main.cpp "${H_FILES}" "${S_FILES}")