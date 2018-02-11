cmake_minimum_required(VERSION 3.1)
project(lapsolverc)

add_subdirectory(pybind11)

find_package(OpenMP)

pybind11_add_module(lapsolverc src/lapsolverc.cpp)

target_compile_options(lapsolverc PUBLIC "$<$<CONFIG:RELEASE>:-O2>")
target_compile_options(lapsolverc PUBLIC "$<$<CONFIG:RELEASE>:${OpenMP_CXX_FLAGS}>")