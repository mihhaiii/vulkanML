#include "operator_softmax_cpu.h"
#include <algorithm>
#include <numeric>

void operator_softmax_cpu(float * inputs, float * outputs, int batch_size, int sample_size)
{
	// inputs and outputs are both (batch_size x sample_size)
	int size = batch_size * sample_size;
	for (int i = 0; i < size; i++) {
		outputs[i] = inputs[i];
	}

	for (int b = 0; b < batch_size; b++)
	{
		int firstIdx = b * sample_size;
		int endIdx = firstIdx + sample_size;

		float maxElem = -FLT_MAX;
		for (int i = firstIdx; i < endIdx; i++) {
			maxElem = (std::max)(maxElem, outputs[i]);
		}

		if (maxElem > 0) {
			for (int i = firstIdx; i < endIdx; i++) {
				outputs[i] -= maxElem;
			}
		}
		float sum = 0;
		for (int i = firstIdx; i < endIdx; i++) {
			outputs[i] = exp(outputs[i]);
			sum += outputs[i];
		}

		for (int i = firstIdx; i < endIdx; i++) {
			outputs[i] /= sum;
		}
	}

}