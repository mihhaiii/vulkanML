cmake_minimum_required(VERSION 3.8)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

project(yolo_tiny)

#find_package(vulkanML REQUIRED)

add_executable(yolo_tiny main.cpp)
target_link_libraries(yolo_tiny PUBLIC vulkanML)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
target_link_libraries(yolo_tiny PUBLIC ${OpenCV_LIBS})


MESSAGE("Copying test images to the binaries directory")
file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/data"
     DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")	