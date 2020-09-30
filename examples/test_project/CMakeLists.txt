cmake_minimum_required(VERSION 3.8)
project(test_project)

#find_package(vulkanML REQUIRED)

add_executable(test_project main.cpp test_conv_operators.cpp test_conv_operators.h)
target_link_libraries(test_project PUBLIC vulkanML)

file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/trainingSample"
     DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")
	 
MESSAGE("Copying tre-trained weights to the binaries directory")
file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/conv_mnist"
     DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")	