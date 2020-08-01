#pragma once
#include "vuh/array.hpp"

void vulkan_operator_conv2d(float * inputImage, int height, int width, int channels,
	float * kernels, int kernelHeight, int kernelWidth, float* kernelBiases, float * outputImage, int numOutputChannels,
	float * outTime = nullptr);

void vulkan_operator_conv2d(vuh::Array<float>* inputImage, int height, int width, int channels,
	vuh::Array<float>* kernels, int kernelHeight, int kernelWidth, vuh::Array<float>* kernelBiases, vuh::Array<float>* outputImage, 
	int numOutputChannels,
	float * outTime = nullptr);