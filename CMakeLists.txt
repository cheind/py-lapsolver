cmake_minimum_required(VERSION 3.1)
project(hungarian)

add_subdirectory(pybind11)
pybind11_add_module(hungarian src/hungarian.cpp)