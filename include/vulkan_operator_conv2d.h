#pragma once
#include "vuh/array.hpp"

void vulkan_operator_conv2d(float* inputImage, float* weights, float* biases, int batch_size, int h, int w, int c, int filters, int size, int stride, int padding, int out_h, int out_w, bool useBias, float* outputImage, float * outTime = nullptr);
void vulkan_operator_conv2d(vuh::Array<float>* inputImage, vuh::Array<float>* weights, vuh::Array<float>* biases, int batch_size, int h, int w, int c, int filters, int size, int stride, int padding, int out_h, int out_w, bool useBias, vuh::Array<float>* outputImage, float * outTime = nullptr);
