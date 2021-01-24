#pragma once

void operator_conv2d_cpu(float* inputImage, float* weights, float* biases, int batch_size, int h, int w, int c, int filters, int size, int stride, int padding, int out_h, int out_w, bool useBias, float* outputImage);
void operator_conv2d_cpu_backprop(float* inputImage, float* weights, float* biases, int batch_size, int h, int w, int c, int filters, int size, int stride, int padding, int out_h, int out_w, bool useBias, float learning_rate, float* outputImage, float* derivatives, float* prev_derivatives);