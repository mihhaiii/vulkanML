#include "test_conv_operators.h"
#include "operator_maxpool_cpu.h"
#include <vector>
#include <assert.h>
#include "operator_conv2d_cpu.h"
#include "vulkan_operator_conv2d.h"
#include "vulkan_operator_maxpool.h"


void test_conv2d()
{
	test_conv2d_1();
	test_conv2d_2();
	test_conv2d_3();
}


void test_conv2d_1()
{
	int height = 5, width = 5, channels = 1, kernelHeight = 3, kernelWidth = 3, numOutputChannels = 1;
	float inputImage[] = {
		1, 2, 3, 4, 5,
		1, 1, 1, 1, 1,
		1, 2, 3, 4, 5,
		1, 1, 1, 1, 1,
		1, 2, 3, 4, 5,
	};
	float kernels[] = {
		0, 1, 0,
		1, 1, 1,
		0, 1, 0
	};
	float kernelBiases[] = {
		1
	};
	std::vector<float> outputImage(9);
	std::vector<float> outputImageVk(9);

	operator_conv2d_cpu(inputImage, height, width, channels,
		kernels, kernelHeight, kernelWidth, kernelBiases, &outputImage[0], numOutputChannels);
	vulkan_operator_conv2d(inputImage, height, width, channels,
		kernels, kernelHeight, kernelWidth, kernelBiases, &outputImageVk[0], numOutputChannels);

	std::vector<float> expected{8, 10, 12, 9, 12, 15, 8, 10, 12};
	for (int i = 0; i < outputImage.size(); i++) {
		assert(outputImage[i] == expected[i]);
		assert(outputImageVk[i] == expected[i]);
	}
}


void test_conv2d_2()
{
	int height = 5, width = 5, channels = 1, kernelHeight = 3, kernelWidth = 3, numOutputChannels = 2;
	float inputImage[] = {
		1, 1, 1, 1, 1,
		1, 1, 1, 1, 1,
		1, 1, 1, 1, 1,
		1, 1, 1, 1, 1,
		1, 1, 1, 1, 1
	};
	// first kernel 
	// 010
	// 111
	// 010
	// second kernel
	// 000
	// 030
	// 000
	float kernels[] = {	
		0, 0,  1, 0,  0, 0,
		1, 0,  1, 3,  1, 0,
		0, 0,  1, 0,  0, 0
	};
	float kernelBiases[] = {
		1,-1
	};
	std::vector<float> outputImage(18);
	std::vector<float> outputImageVk(18);

	operator_conv2d_cpu(inputImage, height, width, channels,
		kernels, kernelHeight, kernelWidth, kernelBiases, &outputImage[0], numOutputChannels);
	vulkan_operator_conv2d(inputImage, height, width, channels,
		kernels, kernelHeight, kernelWidth, kernelBiases, &outputImageVk[0], numOutputChannels);

	std::vector<float> expected{ 6,2,6,2,6,2,6,2,6,2,6,2,6,2,6,2,6,2 };
	for (int i = 0; i < outputImage.size(); i++) {
		assert(outputImage[i] == expected[i]);
		assert(outputImageVk[i] == expected[i]);
	}
}

void test_conv2d_3()
{
	int height = 6, width = 5, channels = 3, kernelHeight = 2, kernelWidth = 2, numOutputChannels = 2;
	float inputImage[] = {
		1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3,
		1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3,
		1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3,
		1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3,
		1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3,
		1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3
	};
	// first kernel 
	// 1  0
	// 0  2 
	// second kernel 
	// 0 -1
	// 3  0
	float kernels[] = {
		1, 0, 1, 0, 1, 0, 0, -1, 0, -1, 0, -1,
		0, 3, 0, 3, 0, 3, 2, 0, 2, 0, 2, 0
	};
	float kernelBiases[] = {
		1,-1
	};
	std::vector<float> outputImage(40);
	std::vector<float> outputImageVk(40);

	operator_conv2d_cpu(inputImage, height, width, channels,
		kernels, kernelHeight, kernelWidth, kernelBiases, &outputImage[0], numOutputChannels);
	vulkan_operator_conv2d(inputImage, height, width, channels,
		kernels, kernelHeight, kernelWidth, kernelBiases, &outputImageVk[0], numOutputChannels);

	std::vector<float> expected;
	while (expected.size() < 40) {
		expected.push_back(19);
		expected.push_back(11);
	}
	for (int i = 0; i < outputImage.size(); i++) {
		assert(outputImage[i] == expected[i]);
		assert(outputImageVk[i] == expected[i]);
	}
}



void test_maxpool() {
	test_maxpool_1();
	test_maxpool_2();
	test_maxpool_3();
}

void test_maxpool_1()
{
	int w = 10, h = 10, c = 1;
	float inputImage[] = {
		0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
		0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
		0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
		0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
		0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
		0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
		0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
		0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
		0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
		0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
	};
	int size = 2, stride = 2;
	int padding = 0; // valid
	int out_w = (w + padding - size) / stride + 1;
	int out_h = (h + padding - size) / stride + 1;
	std::vector<float> outputImage(out_w * out_h * c);
	std::vector<float> outputImageVk(out_w * out_h * c);

	operator_maxpool_cpu(inputImage, h, w, c, size, stride, padding, out_h, out_w, &outputImage[0]);
	vulkan_operator_maxpool(inputImage, h, w, c, size, stride, padding, out_h, out_w, &outputImageVk[0]);

	std::vector<float> expected{ 1,3,5,7,9,1,3,5,7,9, 1,3,5,7,9, 1,3,5,7,9, 1,3,5,7,9 };
	for (int i = 0; i < outputImage.size(); i++) {
		assert(outputImage[i] == expected[i]);
		assert(outputImageVk[i] == expected[i]);
	}
}
void test_maxpool_2()
{
	int w = 5, h = 10, c = 2, size = 2, stride = 2, padding = 0;
	int out_w = (w + padding - size) / stride + 1;
	int out_h = (h + padding - size) / stride + 1;
	float inputImage[] = {
		0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
		0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
		0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
		0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
		0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
		0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
		0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
		0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
		0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
		0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
	};
	std::vector<float> outputImage(out_w * out_h * c);
	std::vector<float> outputImageVk(out_w * out_h * c);

	operator_maxpool_cpu(inputImage, h, w, c, size, stride, padding, out_h, out_w, &outputImage[0]);
	vulkan_operator_maxpool(inputImage, h, w, c, size, stride, padding, out_h, out_w, &outputImageVk[0]);

	std::vector<float> expected{ 2,3,6,7,2,3,6,7,2,3,6,7,2,3,6,7,2,3,6,7 };
	for (int i = 0; i < outputImage.size(); i++) {
		assert(outputImage[i] == expected[i]);
		assert(outputImageVk[i] == expected[i]);
	}
}

void test_maxpool_3()
{
	int w = 5, h = 10, c = 2, size = 2, stride = 2, padding = 1;
	int out_w = (w + padding - size) / stride + 1;
	int out_h = (h + padding - size) / stride + 1;
	float inputImage[] = {
		0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
		0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
		0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
		0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
		0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
		0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
		0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
		0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
		0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
		0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
	};
	std::vector<float> outputImage(out_w * out_h * c);
	std::vector<float> outputImageVk(out_w * out_h * c);

	operator_maxpool_cpu(inputImage, h, w, c, size, stride, padding, out_h, out_w, &outputImage[0]);
	vulkan_operator_maxpool(inputImage, h, w, c, size, stride, padding, out_h, out_w, &outputImageVk[0]);

	std::vector<float> expected{ 2,3,6,7,8,9, 2,3,6,7,8,9, 2,3,6,7,8,9, 2,3,6,7,8,9, 2,3,6,7,8,9};
	for (int i = 0; i < outputImage.size(); i++) {
		assert(outputImage[i] == expected[i]);
		assert(outputImageVk[i] == expected[i]);
	}
}