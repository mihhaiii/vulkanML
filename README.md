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
* open vulkanML.sln, set vulkanML as startup project and hit F5

