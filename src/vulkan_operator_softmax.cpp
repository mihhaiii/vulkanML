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