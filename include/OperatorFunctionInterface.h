#pragma once
#include "MemoryManager.h"
#include "Layers.h"


Conv2dLayer& Conv2D(int filters, int size, int stride = 1, const std::string& padding = "valid", bool useBias = true);
BatchNormalizationLayer& BatchNorm();
MaxPooling2dLayer& MaxPool2D(int size, int stride, std::string padding = "valid");
FCLayer& Dense(int numNeurons, bool applyRelu = false);
Tensor* Input(const Shape& shape, const char* binFile = nullptr);
ReLULayer& ReLU();
LeakyReLULayer& LeakyReLU();
UpSampling2DLayer& UpSampling2D(int size);
ConcatenateLayer& Concatenate();
SoftmaxLayer& Softmax();
ArgmaxLayer& Argmax();