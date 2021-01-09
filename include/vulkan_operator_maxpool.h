#pragma once
#include "vuh/array.hpp"

void vulkan_operator_maxpool(float* inputImage, int batch_size, int h, int w, int c, int size, int stride, int padding, int out_h, int out_w, float* outputImage, float* outTime = nullptr);
void vulkan_operator_maxpool(vuh::Array<float>* inputImage, int batch_size, int h, int w, int c, int size, int stride, int padding, int out_h, int out_w, vuh::Array<float>* outputImage, float* outTime = nullptr);