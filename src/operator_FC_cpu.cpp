#include "operator_FC_cpu.h"
#include <algorithm>

void operator_FC_cpu(float * inputs, float * weigths, float * biases, float * outputs, int numNeuronsLastLayer, int numNeurons, int batch_size, bool applyRelu)
{
	// weights is (numNeuronsLastLayer x numNeurons)
	// biases is (numNeurons)
	// inputs is (batch_size x numNeuronsLastLayer)
	// output is (batch_size x numNeurons)

	for (int sample = 0; sample < batch_size; sample++) {

		for (int i = 0; i < numNeurons; i++) {
			outputs[sample * numNeurons + i] = biases[i];
			for (int j = 0; j < numNeuronsLastLayer; j++) {
				outputs[sample * numNeurons + i] += weigths[j * numNeurons + i] * inputs[sample * numNeuronsLastLayer + j];
			}
			if (applyRelu)
				outputs[sample * numNeurons + i] = std::max(outputs[sample * numNeurons + i], 0.f);
		}

	}
	
}