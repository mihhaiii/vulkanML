#pragma once
#include "vuh/array.hpp"

void vulkan_operator_FC(float * inputs, float * weigths, float * biases, float * outputs, int numNeuronsLastLayer, int numNeurons, int batch_size = 1,
	bool applyRelu = false, float * outTime = nullptr);

void vulkan_operator_FC(vuh::Array<float>* inputs, vuh::Array<float>* weigths, vuh::Array<float>* biases, 
	vuh::Array<float>* outputs, int numNeuronsLastLayer, int numNeurons, int batch_size = 1,
	bool applyRelu = false, float * outTime = nullptr);

void vulkan_operator_FC_backprop(vuh::Array<float>* inputs, vuh::Array<float>* weigths, vuh::Array<float>* biases,
	vuh::Array<float>* derivatives, vuh::Array<float>* prev_derivatives, vuh::Array<float>* outputs, int numNeuronsLastLayer, int numNeurons, int batch_size,
	bool applyRelu, float learning_rate, float * outTime = nullptr);