cmake_minimum_required(VERSION 3.8)
project(vulkanML)

find_package(Vuh REQUIRED)

# compile shaders
vuh_compile_shader(FC_shader
   SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/FC.comp
   TARGET ${CMAKE_CURRENT_BINARY_DIR}/FC.spv
)
vuh_compile_shader(reduce_max_shader
   SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/reduce_max.comp
   TARGET ${CMAKE_CURRENT_BINARY_DIR}/reduce_max.spv
)
vuh_compile_shader(reduce_sum_shader
   SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/reduce_sum.comp
   TARGET ${CMAKE_CURRENT_BINARY_DIR}/reduce_sum.spv
)
vuh_compile_shader(map_add_shader
   SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/map_add_scalar.comp
   TARGET ${CMAKE_CURRENT_BINARY_DIR}/map_add_scalar.spv
)
vuh_compile_shader(map_divide_shader
   SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/map_divide_scalar.comp
   TARGET ${CMAKE_CURRENT_BINARY_DIR}/map_divide_scalar.spv
)
vuh_compile_shader(map_exp_shader
   SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/map_exp.comp
   TARGET ${CMAKE_CURRENT_BINARY_DIR}/map_exp.spv
)
vuh_compile_shader(relu_shader
   SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/relu.comp
   TARGET ${CMAKE_CURRENT_BINARY_DIR}/relu.spv
)
vuh_compile_shader(conv2d_shader
   SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/conv2d.comp
   TARGET ${CMAKE_CURRENT_BINARY_DIR}/conv2d.spv
)
vuh_compile_shader(max_pool_shader
   SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/max_pool.comp
   TARGET ${CMAKE_CURRENT_BINARY_DIR}/max_pool.spv
)

SET(SOURCE
	main.cpp
	vulkan_operator_FC.cpp
	operator_FC_cpu.cpp
	operator_softmax_cpu.cpp
	vulkan_operator_softmax.cpp
	vulkan_operator_reduce_max.cpp
	vulkan_operator_reduce_sum.cpp
	vulkan_operator_map_exp.cpp
	vulkan_operator_map_add.cpp
	vulkan_operator_map_div.cpp
	vulkan_operator_relu.cpp
	operator_conv2d_cpu.cpp
	operator_maxpool_cpu.cpp
	test_conv_operators.cpp
	vulkan_operator_conv2d.cpp
	vulkan_operator_max_pool.cpp
	vulkan_operator_maxpool.cpp)
	
SET(HEADERS
	vulkan_operator_FC.h
	operator_FC_cpu.h
	operator_softmax_cpu.h
	vulkan_operator_softmax.h
	vulkan_operator_reduce_max.h
	vulkan_operator_reduce_sum.h
	vulkan_operator_map_exp.h
	vulkan_operator_map_add.h
	vulkan_operator_map_div.h
	vulkan_operator_relu.h
	operator_conv2d_cpu.h
	operator_maxpool_cpu.h
	test_conv_operators.h
	vulkan_operator_conv2d.h
	vulkan_operator_max_pool.h
	vulkan_operator_maxpool.h
	InstanceManager.h
	InferenceFramework.h
	stb_image.h)

add_executable(vulkanML ${SOURCE} ${HEADERS})

target_link_libraries(vulkanML PUBLIC vuh::vuh)

# add shadere dependencies
add_dependencies(vulkanML FC_shader)
add_dependencies(vulkanML reduce_max_shader)
add_dependencies(vulkanML reduce_sum_shader)
add_dependencies(vulkanML map_add_shader)
add_dependencies(vulkanML map_divide_shader)
add_dependencies(vulkanML map_exp_shader)
add_dependencies(vulkanML relu_shader)
add_dependencies(vulkanML conv2d_shader)
add_dependencies(vulkanML max_pool_shader)


file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/Data"
     DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")
	 
file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/trainingSample"
     DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")

file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/model_weights"
     DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")	
	 
file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/conv_mnist"
     DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")	
	 
	 
	 
	 