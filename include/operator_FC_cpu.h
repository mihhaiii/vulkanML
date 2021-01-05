#pragma once

void operator_FC_cpu(float * inputs, float * weigths, float * biases, float * outputs, int numNeuronsLastLayer, int numNeurons, int batch_size = 1, bool applyRelu = false);