#include "operator_softmax_cpu.h"
#include <algorithm>
#include <numeric>

void operator_softmax_cpu(float * inputs, float * outputs, int numInputs) 
{
	for (int i = 0; i < numInputs; i++) {
		outputs[i] = inputs[i];
	}

	float * last = outputs + numInputs;
	float maxElem = *std::max_element(outputs, last);
	if (maxElem > 0)
		std::for_each(outputs, last, [&](float& v) { v -= maxElem; });
	std::for_each(outputs, last, [&](float& v) { v = exp(v); });
	float sum = 0;
	sum = std::accumulate(outputs, last, sum);
	std::for_each(outputs, last, [&](float& v) { v /= sum; });
}