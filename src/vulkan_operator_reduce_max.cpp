#include "vulkan_operator_reduce_max.h"
#include <vuh/array.hpp>
#include <vuh/vuh.h>
#include <vector>
#include <ctime>

void vulkan_operator_reduce_max(float * inputs, int numInputs, float * result, float * outTime)
{
	auto instance = vuh::Instance();
	auto devices = instance.devices();
	auto it_discrete = std::find_if(begin(devices), end(devices), [&](const auto& d) { return
		d.properties().deviceType == vk::PhysicalDeviceType::eDiscreteGpu;
	});
	auto device = it_discrete != end(devices) ? *it_discrete : devices.at(0);


	using Specs = vuh::typelist<uint32_t>;
	struct Params { int numInputs; };
	const int groupSize = 1024;
	const int numGroups = (numInputs + groupSize - 1) / groupSize;

	auto d_inputs = vuh::Array<float>(device, numGroups * groupSize);
	d_inputs.fromHost(inputs, inputs + numInputs);
	auto d_maxPerGroup = vuh::Array<float>(device, numGroups);

	auto program = vuh::Program<Specs, Params>(device, SHADERS_LOCATION "reduce_max.spv");
	program.grid(numGroups).spec(groupSize)({ numInputs }, d_inputs, d_maxPerGroup);
	std::vector<float> maxPerGroup(numGroups);
	d_maxPerGroup.toHost(begin(maxPerGroup));
	*result = *std::max_element(maxPerGroup.begin(), maxPerGroup.end());
}