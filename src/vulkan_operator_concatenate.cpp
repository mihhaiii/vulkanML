#include <vuh/array.hpp>
#include <vuh/vuh.h>
#include <vector>
#include <ctime>
#include "InstanceManager.h"
#include "vulkan_operator_concatenate.h"


void vulkan_operator_concatenate(vuh::Array<float>* input1, vuh::Array<float>* input2, vuh::Array<float>* outputs, int batch_size, int h, int w, int c1, int c2)
{
	using Specs = vuh::typelist<uint32_t>;
	struct Params { int batch_size, h, w, c1, c2; };
	const int groupSize = 32;
	const int numGroups = ((c1 + c2) + groupSize - 1) / groupSize;

	static auto program = vuh::Program<Specs, Params>(InstanceManger::getInstance().getDefaultDevice(), SHADERS_LOCATION "concatenate.spv");
	program.grid(numGroups).spec(groupSize)({ batch_size, h, w, c1, c2 }, *input1, *input2, *outputs);
}