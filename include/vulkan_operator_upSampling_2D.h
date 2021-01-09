#pragma once
#include "vuh/array.hpp"

void vulkan_operator_upSampling_2D(vuh::Array<float>* inputs, vuh::Array<float>* outputs, int batch_size, int h,  int w, int c, int size);