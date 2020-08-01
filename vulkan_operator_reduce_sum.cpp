#include "vulkan_operator_reduce_sum.h"
#include <vuh/array.hpp>
#include <vuh/vuh.h>
#include <vector>
#include <ctime>
#include <numeric>

void vulkan_operator_reduce_sum(float * inputs, int numInputs, float * result, float * outTime)
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
	auto d_sumPerGroup = vuh::Array<float>(device, numGroups);

	auto program = vuh::Program<Specs, Params>(device, "reduce_sum.spv");
	program.grid(numGroups).spec(groupSize)({ numInputs }, d_inputs, d_sumPerGroup);
	std::vector<float> sumPerGroup(numGroups);
	d_sumPerGroup.toHost(begin(sumPerGroup));
	*result = std::accumulate(sumPerGroup.begin(), sumPerGroup.end(), 0.);
}