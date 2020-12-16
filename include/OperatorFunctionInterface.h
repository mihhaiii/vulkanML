#pragma once
#include "LayerManager.h"
#include "InferenceFramework.h"


Conv2dLayer& Conv2D(int filters, int size, int stride = 1, const std::string& padding = "valid", bool useBias = true, EnumDevice device = DEVICE_CPU)
{
	Conv2dLayer* l = new Conv2dLayer(filters, size, stride, padding, useBias, device);
	LayerManager::getInstance().add(l);
	return *l;
}

MaxPooling2dLayer& MaxPool2D(int size, int stride, std::string padding = "valid", EnumDevice device = DEVICE_CPU)
{
	MaxPooling2dLayer* l = new MaxPooling2dLayer(size, stride, padding, device);
	LayerManager::getInstance().add(l);
	return *l;
}

FCLayer& Dense(int numNeurons, EnumDevice device = DEVICE_CPU, bool applyRelu = false)
{
	FCLayer* l = new FCLayer(numNeurons, device, applyRelu);
	LayerManager::getInstance().add(l);
	return *l;
}

Tensor* Input(const Shape& shape, EnumDevice device = DEVICE_CPU, const char* binFile = nullptr)
{
	InputLayer* l = new InputLayer(shape, device, binFile);
	LayerManager::getInstance().add(l);
	return l->getOutputTensor();
}

ReLULayer& ReLU(EnumDevice device = DEVICE_CPU)
{
	ReLULayer* l = new ReLULayer(device);
	LayerManager::getInstance().add(l);
	return *l;
}
