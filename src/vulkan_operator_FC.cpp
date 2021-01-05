#include "vulkan_operator_FC.h"
#include <vuh/array.hpp>
#include <vuh/vuh.h>
#include <vector>
#include <ctime>
#include "InstanceManager.h"

void vulkan_operator_FC(float * inputs, float * weigths, float * biases, float * outputs, int numNeuronsLastLayer, int numNeurons, int batch_size,
	bool applyRelu, float * outTime)
{
	auto device = InstanceManger::getInstance().getDefaultDevice();
	// create device buffers for all the input parameters
	auto d_inputs = vuh::Array<float>(device, numNeuronsLastLayer * batch_size);
	d_inputs.fromHost(inputs, inputs + numNeuronsLastLayer * batch_size);

	auto d_weights = vuh::Array<float>(device, numNeuronsLastLayer * numNeurons);
	d_weights.fromHost(weigths, weigths + numNeuronsLastLayer * numNeurons);

	auto d_biases = vuh::Array<float>(device, numNeurons);
	d_biases.fromHost(biases, biases + numNeurons);

	auto d_outputs = vuh::Array<float>(device, numNeurons * batch_size);
	d_outputs.fromHost(outputs, outputs + numNeurons * batch_size);

	using Specs = vuh::typelist<uint32_t>;
	struct Params { int numNeuronsLastLayer; int numNeurons; int batch_size; bool applyRelu; };
	const int groupSize = 128;
	const int numGroups = (numNeurons + groupSize - 1) / groupSize;

	auto program = vuh::Program<Specs, Params>(device, SHADERS_LOCATION "FC.spv");

	clock_t start = clock(); // measure only execution time, without data transfer
	program.grid(numGroups).spec(groupSize)({ numNeuronsLastLayer, numNeurons, batch_size, applyRelu }, d_inputs, d_weights, d_biases, d_outputs);
	if (outTime)
		*outTime = (float)(clock() - start) / CLOCKS_PER_SEC;

	d_outputs.toHost(outputs);
}

void vulkan_operator_FC(vuh::Array<float>* inputs, vuh::Array<float>* weigths, vuh::Array<float>* biases, 
	vuh::Array<float>* outputs, int numNeuronsLastLayer, int numNeurons, int batch_size, bool applyRelu, float * outTime)
{
	using Specs = vuh::typelist<uint32_t>;
	struct Params { int numNeuronsLastLayer; int numNeurons; int batch_size; bool applyRelu; };
	const int groupSize = 128;
	const int numGroups = (numNeurons + groupSize - 1) / groupSize;

	static auto program = vuh::Program<Specs, Params>(InstanceManger::getInstance().getDefaultDevice(), SHADERS_LOCATION "FC.spv");

	clock_t start = clock(); // measure only execution time, without data transfer
	program.grid(numGroups).spec(groupSize)({ numNeuronsLastLayer, numNeurons, batch_size, applyRelu }, *inputs, *weigths, *biases, *outputs);
	if (outTime)
		*outTime = (float)(clock() - start) / CLOCKS_PER_SEC;
}
