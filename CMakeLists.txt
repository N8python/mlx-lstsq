cmake_minimum_required(VERSION 3.27)

project(_ext LANGUAGES CXX OBJCXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OBJCXX_STANDARD 17)
set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(BUILD_SHARED_LIBS "Build extensions as a shared library" ON)

if(DEFINED Python_EXECUTABLE)
  set(Python_FIND_STRATEGY LOCATION)
endif()

find_package(
  Python
  COMPONENTS Interpreter Development.Module
  REQUIRED)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m mlx --cmake-dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE MLX_ROOT)
find_package(MLX CONFIG REQUIRED)

find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
find_library(ACCELERATE_FRAMEWORK Accelerate REQUIRED)
find_library(METAL_FRAMEWORK Metal REQUIRED)
find_library(MPS_FRAMEWORK MetalPerformanceShaders REQUIRED)

add_library(mlx_ext)
target_sources(mlx_ext PUBLIC ${CMAKE_CURRENT_LIST_DIR}/mps_cholesky/mps_cholesky.mm)
target_include_directories(mlx_ext PUBLIC ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(
  mlx_ext
  PUBLIC
  mlx
  ${ACCELERATE_FRAMEWORK}
  ${FOUNDATION_FRAMEWORK}
  ${METAL_FRAMEWORK}
  ${MPS_FRAMEWORK})
target_compile_options(mlx_ext PRIVATE "$<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>")

if(MLX_BUILD_METAL)
  mlx_build_metallib(
    TARGET mlx_ext_metallib
    TITLE mlx_ext
    SOURCES ${CMAKE_CURRENT_LIST_DIR}/mps_cholesky/blocked_cholesky.metal
    INCLUDE_DIRS ${CMAKE_CURRENT_LIST_DIR} ${MLX_INCLUDE_DIRS}
    OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})

  add_dependencies(mlx_ext mlx_ext_metallib)
endif()

nanobind_add_module(
  _ext
  NB_STATIC
  LTO
  NOMINSIZE
  NB_DOMAIN
  mlx
  ${CMAKE_CURRENT_LIST_DIR}/bindings.cpp)
target_link_libraries(_ext PRIVATE mlx_ext)

if(BUILD_SHARED_LIBS)
  target_link_options(_ext PRIVATE -Wl,-rpath,@loader_path)
endif()
