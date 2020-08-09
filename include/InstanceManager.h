#pragma once
#include "vuh/device.h"
#include "vuh/instance.h"

class InstanceManger
{
public:
	static InstanceManger& getInstance() {
		static InstanceManger obj;
		return obj;
	}

	vuh::Device& getDefaultDevice() {
		auto devices = instance.devices();
		auto it_discrete = std::find_if(begin(devices), end(devices), [&](const auto& d) { return
			d.properties().deviceType == vk::PhysicalDeviceType::eDiscreteGpu;
		});
		static vuh::Device dev = it_discrete != end(devices) ? *it_discrete : devices.at(0);
		return dev;
	}

	vuh::Device getDevice() {
		auto devices = instance.devices();
		auto it_discrete = std::find_if(begin(devices), end(devices), [&](const auto& d) { return
			d.properties().deviceType == vk::PhysicalDeviceType::eDiscreteGpu;
		});
		return it_discrete != end(devices) ? *it_discrete : devices.at(0);
	}

	InstanceManger(const  InstanceManger& other) = delete;
	InstanceManger& operator=(const  InstanceManger& other) = delete;
private:

	InstanceManger() {}

	vuh::Instance instance;
};