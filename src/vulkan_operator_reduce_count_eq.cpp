#include <vuh/array.hpp>
#include <vuh/vuh.h>
#include <vector>
#include <ctime>
#include "vulkan_operator_reduce_count_eq.h"
#include <numeric>
#include "InstanceManager.h"

int vulkan_operator_reduce_count_eq(vuh::Array<float>* t1, vuh::Array<float>* t2, int numInputs)
{
	using Specs = vuh::typelist<uint32_t>;
	struct Params { int numInputs; };
	const int groupSize = 1024;
	const int numGroups = (numInputs + groupSize - 1) / groupSize;
	auto d_countPerGroup = vuh::Array<float>(InstanceManger::getInstance().getDefaultDevice(), numGroups);

	static auto program = vuh::Program<Specs, Params>(InstanceManger::getInstance().getDefaultDevice(), SHADERS_LOCATION "reduce_count_eq.spv");
	program.grid(numGroups).spec(groupSize)({ numInputs }, *t1, *t2, d_countPerGroup);
	std::vector<float> countPerGroup(numGroups);
	d_countPerGroup.toHost(begin(countPerGroup));
	return (int)std::accumulate(countPerGroup.begin(), countPerGroup.end(), 0.f);
}
