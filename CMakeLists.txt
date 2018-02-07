cmake_minimum_required(VERSION 3.1)
project(fast_hungarian)

add_subdirectory(pybind11)
pybind11_add_module(fast_hungarian_ext src/hungarian.cpp)