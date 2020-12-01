#include "vulkan_operator_relu.h"
#include <vuh/array.hpp>
#include <vuh/vuh.h>
#include <vector>
#include <ctime>
#include "InstanceManager.h"


void vulkan_operator_batch_normalization(vuh::Array<float>* inputs, int numInputs, int filters, vuh::Array<float>* params)
{
	// params is 4 x filters, order same as tf: gamma, beta, mean, variance
	using Specs = vuh::typelist<uint32_t>;
	struct Params {
		int numInputs, filters;
	};
	const int groupSize = 1024;
	const int numGroups = (numInputs + groupSize - 1) / groupSize;

	static auto program = vuh::Program<Specs, Params>(InstanceManger::getInstance().getDefaultDevice(), SHADERS_LOCATION "batch_normalization.spv");
	program.grid(numGroups).spec(groupSize)({ numInputs, filters }, *inputs, *params);
}