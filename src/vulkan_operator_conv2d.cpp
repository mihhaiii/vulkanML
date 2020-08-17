#include "vulkan_operator_conv2d.h"
#include <vuh/array.hpp>
#include <vuh/vuh.h>
#include <vector>
#include <ctime>
#include "InstanceManager.h"

void vulkan_operator_conv2d(float* inputImage, float* weights, float* biases, int h, int w, int c, int filters, int size, int stride, int padding, int out_h, int out_w, bool useBias, float* outputImage, float* outTime)
{
	// create device buffers for all the input parameters
	auto d_inputImage = vuh::Array<float>(InstanceManger::getInstance().getDefaultDevice(), h * w * c);
	d_inputImage.fromHost(inputImage, inputImage + h * w * c);

	auto d_weights = vuh::Array<float>(InstanceManger::getInstance().getDefaultDevice(), size * size * c * filters);
	d_weights.fromHost(weights, weights + size * size * c * filters);

	auto d_biases = vuh::Array<float>(InstanceManger::getInstance().getDefaultDevice(), filters);
	d_biases.fromHost(biases, biases + filters);

	auto d_outputImage = vuh::Array<float>(InstanceManger::getInstance().getDefaultDevice(), out_h * out_w * filters);

	vulkan_operator_conv2d(&d_inputImage, &d_weights, &d_biases, h, w, c, filters, size, stride, padding, out_h, out_w, useBias, &d_outputImage, outTime);

	d_outputImage.toHost(outputImage);
}

void vulkan_operator_conv2d(vuh::Array<float>* inputImage, vuh::Array<float>* weights, vuh::Array<float>* biases, int h, int w, int c, int filters, int size, int stride, int padding, int out_h, int out_w, bool useBias, vuh::Array<float>* outputImage, float* outTime)
{
	using Specs = vuh::typelist<uint32_t>;
	struct Params {
		int h, w, c, filters, size, stride, padding, out_h, out_w, useBias;
	};
	const int groupSize = 32;
	const int numGroups = (filters + groupSize - 1) / groupSize;

	static auto program = vuh::Program<Specs, Params>(InstanceManger::getInstance().getDefaultDevice(), "conv2d.spv");

	clock_t start = clock(); // measure only execution time, without data transfer
	program.grid(numGroups).spec(groupSize);
	program({ h, w, c, filters, size, stride, padding, out_h, out_w, (int)useBias }, *inputImage, *weights, *biases, *outputImage);
	if (outTime)
		*outTime = (float)(clock() - start) / CLOCKS_PER_SEC;
}
