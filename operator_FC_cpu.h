#pragma once

void operator_FC_cpu(float * inputs, float * weigths, float * biases, float * outputs, int numInputs, int numOutputs, bool applyRelu = false);