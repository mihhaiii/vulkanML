#include "operator_FC_cpu.h"
#include <algorithm>

void operator_FC_cpu(float * inputs, float * weigths, float * biases, float * outputs, int numInputs, int numOutputs, bool applyRelu)
{
	// weights is numInputs x numOutputs
	for (int i = 0; i < numOutputs; i++) {
		outputs[i] = biases[i];
		for (int j = 0; j < numInputs; j++) {
			outputs[i] += weigths[j * numOutputs + i] * inputs[j];
		}
		if (applyRelu)
			outputs[i] = std::max(outputs[i], 0.f);
	}
}