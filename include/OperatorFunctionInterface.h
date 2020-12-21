#pragma once
#include "LayerManager.h"
#include "InferenceFramework.h"


Conv2dLayer& Conv2D(int filters, int size, int stride = 1, const std::string& padding = "valid", bool useBias = true)
{
	Conv2dLayer* l = new Conv2dLayer(filters, size, stride, padding, useBias);
	LayerManager::getInstance().add(l);
	return *l;
}

BatchNormalizationLayer& BatchNorm()
{
	BatchNormalizationLayer* l = new BatchNormalizationLayer();
	LayerManager::getInstance().add(l);
	return *l;
}

MaxPooling2dLayer& MaxPool2D(int size, int stride, std::string padding = "valid")
{
	MaxPooling2dLayer* l = new MaxPooling2dLayer(size, stride, padding);
	LayerManager::getInstance().add(l);
	return *l;
}

FCLayer& Dense(int numNeurons, bool applyRelu = false)
{
	FCLayer* l = new FCLayer(numNeurons, applyRelu);
	LayerManager::getInstance().add(l);
	return *l;
}

Tensor* Input(const Shape& shape, const char* binFile = nullptr)
{
	InputLayer* l = new InputLayer(shape, binFile);
	LayerManager::getInstance().add(l);
	return l->getOutputTensor();
}

ReLULayer& ReLU()
{
	ReLULayer* l = new ReLULayer();
	LayerManager::getInstance().add(l);
	return *l;
}

LeakyReLULayer& LeakyReLU()
{
	LeakyReLULayer* l = new LeakyReLULayer();
	LayerManager::getInstance().add(l);
	return *l;
}
