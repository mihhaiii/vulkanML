#pragma once

void operator_conv2d_cpu(float* inputImage, float* weights, float* biases, int h, int w, int c, int filters, int size, int stride, int padding, int out_h, int out_w, bool useBias, float* outputImage);