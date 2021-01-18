#pragma once
#include "vuh/array.hpp"

void vulkan_operator_softmax(float * inputs,  float * outputs, int numInputs, float * outTime = nullptr);

// new version with parallelization based on batches
void vulkan_operator_softmax(vuh::Array<float>* inputs, vuh::Array<float>* outputs, int batch_size, int sample_size);
void vulkan_operator_softmax_backprop(vuh::Array<float>* derivatives, vuh::Array<float>* outputs, vuh::Array<float>* ground_truth, 
	int ground_truth_offset, int batch_size, int sample_size);


void vulkan_operator_argmax(vuh::Array<float>* inputs, vuh::Array<float>* outputs, int batch_size, int sample_size);