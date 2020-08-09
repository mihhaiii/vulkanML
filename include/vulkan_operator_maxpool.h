#pragma once
#include "vuh/array.hpp"

void vulkan_operator_maxpool(float * inputImage, int height, int width, int channels,
	int kernelHeight, int kernelWidth, float * outputImage,
	float * outTime = nullptr);

void vulkan_operator_maxpool(vuh::Array<float>* inputImage, int height, int width, int channels,
	int kernelHeight, int kernelWidth, vuh::Array<float>* outputImage,
	float * outTime = nullptr);