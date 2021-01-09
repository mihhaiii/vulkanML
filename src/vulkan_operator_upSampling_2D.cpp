#include <vuh/array.hpp>
#include <vuh/vuh.h>
#include <vector>
#include <ctime>
#include "InstanceManager.h"
#include "vulkan_operator_upSampling_2D.h"

void vulkan_operator_upSampling_2D(vuh::Array<float>* inputs, vuh::Array<float>* outputs, int batch_size, int h, int w, int c, int size)
{
	using Specs = vuh::typelist<uint32_t>;
	struct Params { int batch_size, h, w, out_h, out_w, c, size; };
	const int groupSize = 32;
	const int numGroups = (c + groupSize - 1) / groupSize;

	static auto program = vuh::Program<Specs, Params>(InstanceManger::getInstance().getDefaultDevice(), SHADERS_LOCATION "upSampling2D.spv");
	program.grid(numGroups).spec(groupSize)({ batch_size, h, w, h * size, w * size, c, size }, *inputs, *outputs);
}