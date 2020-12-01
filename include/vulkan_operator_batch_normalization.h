#pragma once
#include "vuh/array.hpp"

void vulkan_operator_batch_normalization(vuh::Array<float>* inputs, int numInputs, int filters, vuh::Array<float>* params);