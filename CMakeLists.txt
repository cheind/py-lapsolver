cmake_minimum_required(VERSION 3.1)
project(lapsolverc)

add_subdirectory(pybind11)
pybind11_add_module(lapsolverc src/lapsolverc.cpp)
target_compile_options(lapsolverc PUBLIC "$<$<CONFIG:RELEASE>:-O2>")