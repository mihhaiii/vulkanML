#include "vulkan_operator_relu.h"
#include <vuh/array.hpp>
#include <vuh/vuh.h>
#include <vector>
#include <ctime>
#include "InstanceManager.h"


void vulkan_operator_relu(float * inputs, float * outputs, int numInputs, float * outTime)
{
	/*auto instance = vuh::Instance();
	auto devices = instance.devices();
	auto it_discrete = std::find_if(begin(devices), end(devices), [&](const auto& d) { return
		d.properties().deviceType == vk::PhysicalDeviceType::eDiscreteGpu;
	});
	auto device = it_discrete != end(devices) ? *it_discrete : devices.at(0);*/
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