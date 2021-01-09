#pragma once
#include "vuh/array.hpp"

void vulkan_operator_concatenate(vuh::Array<float>* input1, vuh::Array<float>* input2, vuh::Array<float>* outputs, int batch_size, int h,  int w, int c1, int c2);