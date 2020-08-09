#include "vulkan_operator_conv2d.h"
#include <vuh/array.hpp>
#include <vuh/vuh.h>
#include <vector>
#include <ctime>
#include "InstanceManager.h"

void vulkan_operator_conv2d(float * inputImage, int height, int width, int channels, 
	float * kernels, int kernelHeight, int kernelWidth, float * kernelBiases, float * outputImage, int numOutputChannels,
	float * outTime)
{
	auto device = InstanceManger::getInstance().getDefaultDevice();
	// input image is flattened version of   height x width x channels
	// kernels is flattened version of	kernelHeight x kernelWidth x channels x numOutputChannels
	// kernelBiases has a size of	numOutputChannels
	// output image is flattened version of		(height - kernelHeight + 1)	x (width - kernelWidth + 1) x numOutputChannels

	// create device buffers for all the input parameters
	auto d_inputImage = vuh::Array<float>(device, height * width * channels);
	d_inputImage.fromHost(inputImage, inputImage + height * width * channels);

	auto d_kernels = vuh::Array<float>(device, kernelHeight * kernelWidth * channels * numOutputChannels);
	d_kernels.fromHost(kernels, kernels + kernelHeight * kernelWidth * channels * numOutputChannels);

	auto d_kernelBiases = vuh::Array<float>(device, numOutputChannels);
	d_kernelBiases.fromHost(kernelBiases, kernelBiases + numOutputChannels);

	auto d_outputImage = vuh::Array<float>(device, (height - kernelHeight + 1) * (width - kernelWidth + 1) * numOutputChannels);

	using Specs = vuh::typelist<uint32_t>;
	struct Params {
		int height; int width; int channels; int kernelHeight; int kernelWidth; int numOutputChannels;
	};
	const int groupSizeX = 32, groupSizeY = 32, groupSizeZ = 32;
	const int xSize = height - kernelHeight + 1, ySize = width - kernelWidth + 1, zSize = numOutputChannels;
	const int numGroupsX = (xSize + groupSizeX - 1) / groupSizeX;
	const int numGroupsY = (ySize + groupSizeY - 1) / groupSizeY;
	const int numGroupsZ = (zSize + groupSizeZ - 1) / groupSizeZ;

	auto program = vuh::Program<Specs, Params>(device, "conv2d.spv");

	clock_t start = clock(); // measure only execution time, without data transfer
	program.grid(numGroupsZ).spec(groupSizeZ);
	program({ height,width,channels,kernelHeight,kernelWidth,numOutputChannels }, d_inputImage, d_kernels, d_kernelBiases, d_outputImage);
	if (outTime)
		*outTime = (float)(clock() - start) / CLOCKS_PER_SEC;

	d_outputImage.toHost(outputImage);
}

void vulkan_operator_conv2d(vuh::Array<float>* inputImage, int height, int width, int channels, vuh::Array<float>* kernels, 
	int kernelHeight, int kernelWidth, vuh::Array<float>* kernelBiases, vuh::Array<float>* outputImage, int numOutputChannels, float * outTime)
{
	using Specs = vuh::typelist<uint32_t>;
	struct Params {
		int height; int width; int channels; int kernelHeight; int kernelWidth; int numOutputChannels;
	};
	const int groupSizeX = 32, groupSizeY = 32, groupSizeZ = 32;
	const int xSize = height - kernelHeight + 1, ySize = width - kernelWidth + 1, zSize = numOutputChannels;
	const int numGroupsX = (xSize + groupSizeX - 1) / groupSizeX;
	const int numGroupsY = (ySize + groupSizeY - 1) / groupSizeY;
	const int numGroupsZ = (zSize + groupSizeZ - 1) / groupSizeZ;

	static auto program = vuh::Program<Specs, Params>(InstanceManger::getInstance().getDefaultDevice(), "conv2d.spv");

	clock_t start = clock(); // measure only execution time, without data transfer
	program.grid(numGroupsZ).spec(groupSizeZ);
	program({ height,width,channels,kernelHeight,kernelWidth,numOutputChannels }, *inputImage, *kernels, *kernelBiases, *outputImage);
	if (outTime)
		*outTime = (float)(clock() - start) / CLOCKS_PER_SEC;
}
