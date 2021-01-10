cmake_minimum_required(VERSION 3.8)
project(vulkanML)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Vuh REQUIRED)


file(GLOB SHADERS
    "shaders/*.comp"
)
	
file(GLOB SOURCE
    "src/*.cpp"
)

file (GLOB HEADERS
	"include/*.h"

)

MESSAGE("Creating static library vulkanML")
add_library(vulkanML STATIC ${SOURCE} ${HEADERS})
target_include_directories(vulkanML PUBLIC "include/")

target_link_libraries(vulkanML PUBLIC vuh::vuh)

if(WIN32)
    set_target_properties(vulkanML PROPERTIES COMPILE_DEFINITIONS NOMINMAX)
endif()

# compile shaders and add dependencies
FOREACH (shader_file ${SHADERS})
	get_filename_component(shader_name ${shader_file} NAME_WE)
	MESSAGE("Compiling shader ${shader_name}")
	vuh_compile_shader("${shader_name}_shader"
		SOURCE ${shader_file}
		TARGET "${CMAKE_CURRENT_BINARY_DIR}/compiled_shaders/${shader_name}.spv"
	)
	MESSAGE(${CMAKE_CURRENT_BINARY_DIR}/${shader_name}.spv)
	add_dependencies(vulkanML "${shader_name}_shader")
ENDFOREACH()

add_compile_definitions(SHADERS_LOCATION="${CMAKE_CURRENT_BINARY_DIR}/compiled_shaders/")
	 
add_subdirectory(examples)
	 

	 
	 