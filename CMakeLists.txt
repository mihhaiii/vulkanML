cmake_minimum_required(VERSION 3.8)
project(vulkanML)

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

MESSAGE("Creating the executable project vulkanML")
add_executable(vulkanML ${SOURCE} ${HEADERS})
target_include_directories(vulkanML PRIVATE "include/")

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
		TARGET ${CMAKE_CURRENT_BINARY_DIR}/${shader_name}.spv
	)
	add_dependencies(vulkanML "${shader_name}_shader")
ENDFOREACH()
	
MESSAGE("Copying tre-trained weights to the binaries directory")
file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/trainingSample"
     DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")

file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/model_weights"
     DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")	
	 
file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/conv_mnist"
     DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")	
	 
	 
	 
	 