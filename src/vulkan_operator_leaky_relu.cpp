#include <vuh/array.hpp>
#include <vuh/vuh.h>
#include <vector>
#include <ctime>
#include "InstanceManager.h"
#include "vulkan_operator_leaky_relu.h"


void vulkan_operator_leaky_relu(float * inputs, float * outputs, int numInputs, float * outTime)
{
	auto device = InstanceManger::getInstance().getDefaultDevice();
	using Specs = vuh::typelist<uint32_t>;
	struct Params { int numInputs; };
	const int groupSize = 1024;
	const int numGroups = (numInputs + groupSize - 1) / groupSize;

	auto d_inputs = vuh::Array<float>(device, numInputs);
	d_inputs.fromHost(inputs, inputs + numInputs);
	auto d_outputs = vuh::Array<float>(device, numInputs);

	auto program = vuh::Program<Specs, Params>(device, SHADERS_LOCATION "leaky_relu.spv");
	program.grid(numGroups).spec(groupSize)({ numInputs }, d_inputs, d_outputs);
	d_outputs.toHost(outputs);
}

void vulkan_operator_leaky_relu(vuh::Array<float>* inputs, vuh::Array<float>* outputs, int numInputs, float* outTime)
{
	using Specs = vuh::typelist<uint32_t>;
	struct Params { int numInputs; };
	const int groupSize = 1024;
	const int numGroups = (numInputs + groupSize - 1) / groupSize;

	static auto program = vuh::Program<Specs, Params>(InstanceManger::getInstance().getDefaultDevice(), SHADERS_LOCATION "leaky_relu.spv");
	program.grid(numGroups).spec(groupSize)({ numInputs }, *inputs, *outputs);
}