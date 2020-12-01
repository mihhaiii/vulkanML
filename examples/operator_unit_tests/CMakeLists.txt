cmake_minimum_required(VERSION 3.8)
project(operator_unit_tests)

#find_package(vulkanML REQUIRED)

add_executable(operator_unit_tests main.cpp unit_tests.cpp unit_tests.h)
target_link_libraries(operator_unit_tests PUBLIC vulkanML)

MESSAGE("Creating unit tests project")
