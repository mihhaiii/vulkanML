#include "vulkan_operator_softmax.h"
#include <vuh/array.hpp>
#include <vuh/vuh.h>
#include <vector>
#include <ctime>
#include "vulkan_operator_reduce_max.h"
#include "vulkan_operator_map_add.h"
#include "vulkan_operator_map_exp.h"
#include "vulkan_operator_reduce_sum.h"
#include "vulkan_operator_map_div.h"
#include "InstanceManager.h"

void vulkan_operator_softmax(float * inputs, float * outputs, int numInputs, float * outTime)

{
	float maxElem;
	vulkan_operator_reduce_max(inputs, numInputs, &maxElem);

	vulkan_operator_map_add(inputs, numInputs, outputs, -maxElem);

	vulkan_operator_map_exp(outputs, numInputs, outputs); 

	float sum;
	vulkan_operator_reduce_sum(outputs, numInputs, &sum);

	vulkan_operator_map_div(outputs, numInputs, outputs, sum);
}


void vulkan_operator_softmax(vuh::Array<float>* inputs, vuh::Array<float>* outputs, int batch_size, int sample_size)
{
	using Specs = vuh::typelist<uint32_t>;
	struct Params {
		int batch_size, sample_size;
	};
	const int groupSize = 1024;
	const int numGroups = (batch_size + groupSize - 1) / groupSize;

	static auto program = vuh::Program<Specs, Params>(InstanceManger::getInstance().getDefaultDevice(), SHADERS_LOCATION "softmax.spv");
	program.grid(numGroups).spec(groupSize)({ batch_size, sample_size }, *inputs, *outputs);
}

void vulkan_operator_softmax_backprop(vuh::Array<float>* derivatives, vuh::Array<float>* outputs, vuh::Array<float>* ground_truth, int batch_size, int sample_size)
{
	using Specs = vuh::typelist<uint32_t>;
	struct Params {
		int batch_size, sample_size;
	};
	const int groupSize = 1024;
	const int numGroups = (batch_size + groupSize - 1) / groupSize;

	static auto program = vuh::Program<Specs, Params>(InstanceManger::getInstance().getDefaultDevice(), SHADERS_LOCATION "softmax_backprop.spv");
	program.grid(numGroups).spec(groupSize)({ batch_size, sample_size }, *derivatives, *outputs, *ground_truth);
}


void vulkan_operator_argmax(vuh::Array<float>* inputs, vuh::Array<float>* outputs, int batch_size, int sample_size)
{
	using Specs = vuh::typelist<uint32_t>;
	struct Params {
		int batch_size, sample_size;
	};
	const int groupSize = 1024;
	const int numGroups = (batch_size + groupSize - 1) / groupSize;

	static auto program = vuh::Program<Specs, Params>(InstanceManger::getInstance().getDefaultDevice(), SHADERS_LOCATION "argmax.spv");
	program.grid(numGroups).spec(groupSize)({ batch_size, sample_size }, *inputs, *outputs);
}
