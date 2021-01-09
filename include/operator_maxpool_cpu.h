#pragma once

void operator_maxpool_cpu(float* inputImage, int batch_size, int h, int w, int c, int size, int stride, int padding, int out_h, int out_w, float* outputImage);