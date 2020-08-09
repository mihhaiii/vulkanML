#pragma once

void operator_conv2d_cpu(float * inputImage,  int height, int width, int channels,
	float * kernels,  int kernelHeight, int kernelWidth, float* kernelBiases, float * outputImage, int numOutputChannels);