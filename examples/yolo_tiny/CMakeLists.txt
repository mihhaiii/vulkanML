cmake_minimum_required(VERSION 3.8)
project(yolo_tiny)

#find_package(vulkanML REQUIRED)

add_executable(yolo_tiny main.cpp)
target_link_libraries(yolo_tiny PUBLIC vulkanML)


MESSAGE("Copying test images to the binaries directory")
file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/data"
     DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")	