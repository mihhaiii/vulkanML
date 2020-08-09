#pragma once
#include "vuh/array.hpp"

void vulkan_operator_leaky_relu(float * inputs,  float * outputs, int numInputs, float * outTime = nullptr);

void vulkan_operator_leaky_relu(vuh::Array<float>* inputs, vuh::Array<float>* outputs, int numInputs, float * outTime = nullptr);