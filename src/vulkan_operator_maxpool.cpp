#include <vuh/array.hpp>
#include <vuh/vuh.h>
#include <vector>
#include <ctime>
#include "vulkan_operator_maxpool.h"
#include "InstanceManager.h"
#include "GlobalTimeTracker.h"

void vulkan_operator_maxpool(float* inputImage, int batch_size, int h, int w, int c, int size, int stride, int padding, int out_h, int out_w, float* outputImage, float* outTime)
{
	// create device buffers for all the input parameters
	auto d_inputImage = vuh::Array<float>(InstanceManger::getInstance().getDefaultDevice(), batch_size * h * w * c);
	d_inputImage.fromHost(inputImage, inputImage + batch_size * h * w * c);
	auto d_outputImage = vuh::Array<float>(InstanceManger::getInstance().getDefaultDevice(), batch_size * out_h * out_w * c);

	vulkan_operator_maxpool(&d_inputImage, batch_size, h, w, c, size, stride, padding, out_h, out_w, &d_outputImage, outTime);
	d_outputImage.toHost(outputImage);
}

void vulkan_operator_maxpool(vuh::Array<float>* inputImage, int batch_size, int h, int w, int c, int size, int stride, int padding, int out_h, int out_w, vuh::Array<float>* outputImage, float* outTime)
{
	using Specs = vuh::typelist<uint32_t, uint32_t, uint32_t>;
	struct Params {
		int batch_size, h, w, c, size, stride, padding, out_h, out_w;
	};
	const int groupSizeX = 8;
	const int groupSizeY = 8;
	const int groupSizeZ = 16;
	const int numGroupsX = (out_h + groupSizeX - 1) / groupSizeX;
	const int numGroupsY = (out_w + groupSizeY - 1) / groupSizeY;
	const int numGroupsZ = (c + groupSizeZ - 1) / groupSizeZ;

	static auto program = vuh::Program<Specs, Params>(InstanceManger::getInstance().getDefaultDevice(), SHADERS_LOCATION "max_pool.spv");

	clock_t start = clock(); // measure only execution time, without data transfer
	program.grid(numGroupsX, numGroupsY, numGroupsZ).spec(groupSizeX, groupSizeY, groupSizeZ);
	{
		//SCOPE_TIMER("maxpool_shader");
		program({ batch_size, h, w, c, size, stride, padding, out_h, out_w }, *inputImage, *outputImage);
	}
	if (outTime)
		*outTime = (float)(clock() - start) / CLOCKS_PER_SEC;
}

void vulkan_operator_maxpool_backprop(vuh::Array<float>* inputImage, int batch_size, int h, int w, int c, int size, int stride, int padding, int out_h, int out_w,
	vuh::Array<float>* outputImage, vuh::Array<float>* derivatives, vuh::Array<float>* prev_derivatives, float* outTime)
{
	using Specs = vuh::typelist<uint32_t, uint32_t, uint32_t>;
	struct Params {
		int batch_size, h, w, c, size, stride, padding, out_h, out_w;
	};
	const int groupSizeX = 8;
	const int groupSizeY = 8;
	const int groupSizeZ = 16;
	const int numGroupsX = (out_h + groupSizeX - 1) / groupSizeX;
	const int numGroupsY = (out_w + groupSizeY - 1) / groupSizeY;
	const int numGroupsZ = (c + groupSizeZ - 1) / groupSizeZ;

	static auto program = vuh::Program<Specs, Params>(InstanceManger::getInstance().getDefaultDevice(), SHADERS_LOCATION "max_pool_backprop.spv");

	clock_t start = clock(); // measure only execution time, without data transfer
	program.grid(numGroupsX, numGroupsY, numGroupsZ).spec(groupSizeX, groupSizeY, groupSizeZ); 
	{
		//SCOPE_TIMER("maxpool_backprop_shader");
		program({ batch_size, h, w, c, size, stride, padding, out_h, out_w }, *inputImage, *derivatives, *prev_derivatives);
	}
	if (outTime)
		*outTime = (float)(clock() - start) / CLOCKS_PER_SEC;
}