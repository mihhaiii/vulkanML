#include <vuh/array.hpp>
#include <vuh/vuh.h>
#include <vector>
#include <ctime>
#include "vulkan_operator_maxpool.h"
#include "InstanceManager.h"

void vulkan_operator_maxpool(float * inputImage, int height, int width, int channels, int kernelHeight, int kernelWidth, float * outputImage, float * outTime)
{
	auto device = InstanceManger::getInstance().getDefaultDevice();
	// input image is flattened version of   height x width x channels
	// output image is flattened version of (height / kernelHeight) x (width / kernelWidth) x channels,
	int outWidth = width / kernelWidth;
	int outHeight = height / kernelHeight;

	// create device buffers for all the input parameters
	auto d_inputImage = vuh::Array<float>(device, height * width * channels);
	d_inputImage.fromHost(inputImage, inputImage + height * width * channels);

	auto d_outputImage = vuh::Array<float>(device, outWidth * outHeight * channels);

	using Specs = vuh::typelist<uint32_t>;
	struct Params {
		int height; int width; int channels; int kernelHeight; int kernelWidth;
	};
	const int groupSize = 32;
	const int numGroups = (channels + groupSize - 1) / groupSize;

	auto program = vuh::Program<Specs, Params>(device, "max_pool.spv");

	clock_t start = clock(); // measure only execution time, without data transfer
	program.grid(numGroups).spec(groupSize);
	program({ height, width, channels, kernelHeight, kernelWidth }, d_inputImage, d_outputImage);
	if (outTime)
		*outTime = (float)(clock() - start) / CLOCKS_PER_SEC;

	d_outputImage.toHost(outputImage);
}

void vulkan_operator_maxpool(vuh::Array<float>* inputImage, int height, int width, int channels, int kernelHeight, int kernelWidth,
	vuh::Array<float>* outputImage, float * outTime)
{
	using Specs = vuh::typelist<uint32_t>;
	struct Params {
		int height; int width; int channels; int kernelHeight; int kernelWidth;
	};
	const int groupSize = 32;
	const int numGroups = (channels + groupSize - 1) / groupSize;

	auto program = vuh::Program<Specs, Params>(InstanceManger::getInstance().getDefaultDevice(), "max_pool.spv");

	clock_t start = clock(); // measure only execution time, without data transfer
	program.grid(numGroups).spec(groupSize);
	program({ height, width, channels, kernelHeight, kernelWidth }, *inputImage, *outputImage);
	if (outTime)
		*outTime = (float)(clock() - start) / CLOCKS_PER_SEC;
}
