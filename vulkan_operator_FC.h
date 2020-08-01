#pragma once
#include "vuh/array.hpp"

void vulkan_operator_FC(float * inputs, float * weigths, float * biases, float * outputs, int numInputs, int numOutputs,
	bool applyRelu = false, float * outTime = nullptr);

void vulkan_operator_FC(vuh::Array<float>* inputs, vuh::Array<float>* weigths, vuh::Array<float>* biases, 
	vuh::Array<float>* outputs, int numInputs, int numOutputs,
	bool applyRelu = false, float * outTime = nullptr);