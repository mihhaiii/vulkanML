#include "vulkan_operator_relu.h"
#include <vuh/array.hpp>
#include <vuh/vuh.h>
#include <vector>
#include <ctime>
#include "InstanceManager.h"


void vulkan_operator_relu(float * inputs, float * outputs, int numInputs, float * outTime)
{
	auto device = InstanceManger::getInstance().getDefaultDevice();
	using Specs = vuh::typelist<uint32_t>;
	struct Params { int numInputs; };
	const int groupSize = 1024;
	const int numGroups = (numInputs + groupSize - 1) / groupSize;

	auto d_inputs = vuh::Array<float>(device, numInputs);
	d_inputs.fromHost(inputs, inputs + numInputs);
	auto d_outputs = vuh::Array<float>(device, numInputs);

	auto program = vuh::Program<Specs, Params>(device, "relu.spv");
	program.grid(numGroups).spec(groupSize)({ numInputs }, d_inputs, d_outputs);
	d_outputs.toHost(outputs);
}

void vulkan_operator_relu(vuh::Array<float>* inputs, vuh::Array<float>* outputs, int numInputs, float* outTime)
{
	using Specs = vuh::typelist<uint32_t>;
	struct Params { int numInputs; };
	const int groupSize = 1024;
	const int numGroups = (numInputs + groupSize - 1) / groupSize;

	auto program = vuh::Program<Specs, Params>(InstanceManger::getInstance().getDefaultDevice(), "relu.spv");
	program.grid(numGroups).spec(groupSize)({ numInputs }, *inputs, *outputs);
}