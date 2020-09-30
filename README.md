# vulkanML

Dependencies: 
* C++ 14
* CMake
* Vulkan SDK
* vuh (vulkan compute library)
  * git clone https://github.com/mihhaiii/vuh 
  * cd vuh
  * mkdir build
  * cd build
  * cmake -DCMAKE_INSTALL_PREFIX="path/where/to/install/vuh" ..
  * cmake --build . --target install
  
Build & Run instructions
* git clone https://github.com/mihhaiii/vulkanML
* cd vulkanML
* mkdir build
* cd build
* cmake -DCMAKE_PREFIX_PATH="path/where/to/install/vuh" ..
* cmake --build .
* (windows) open vulkanML.sln, set test_project as startup project and hit F5

test_project is an example project that links the vulkanML static library and uses it to make a simple MNIST inference.

To create a new project:
* create a new folder in examples/
* add to the examples/CMakeLists.txt file the following line: add_subdirectory(your_folder_name)
* add a CMakeLists.txt file in the created folder (use examples/test_project/CMakeLists.txt as an example)
  * create the executable with your .cpp and .h files with add_executable(...)
  * link the vulkanML library with target_link_libraries(...) - this will also add the necessary include directories to your project
* if you need new shaders just add them to the /shaders folder. The root CMakeLists.txt will automatically compile them at build time.
* regenerate and rebuild the solution
* (windows) open VS, set your project as the startup project and run the program


