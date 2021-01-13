#pragma once
#include "MemoryManager.h"
#include "OperatorFunctionInterface.h"


Conv2dLayer& Conv2D(int filters, int size, int stride, const std::string& padding, bool useBias)
{
	Conv2dLayer* l = new Conv2dLayer(filters, size, stride, padding, useBias);
	MemoryManager::getInstance().add(l);
	return *l;
}

BatchNormalizationLayer& BatchNorm()
{
	BatchNormalizationLayer* l = new BatchNormalizationLayer();
	MemoryManager::getInstance().add(l);
	return *l;
}

MaxPooling2dLayer& MaxPool2D(int size, int stride, std::string padding)
{
	MaxPooling2dLayer* l = new MaxPooling2dLayer(size, stride, padding);
	MemoryManager::getInstance().add(l);
	return *l;
}

FCLayer& Dense(int numNeurons, bool applyRelu)
{
	FCLayer* l = new FCLayer(numNeurons, applyRelu);
	MemoryManager::getInstance().add(l);
	return *l;
}

Tensor* Input(const Shape& shape, const char* binFile)
{
	InputLayer* l = new InputLayer(shape, binFile);
	MemoryManager::getInstance().add(l);
	return l->getOutputTensor();
}

ReLULayer& ReLU()
{
	ReLULayer* l = new ReLULayer();
	MemoryManager::getInstance().add(l);
	return *l;
}

LeakyReLULayer& LeakyReLU()
{
	LeakyReLULayer* l = new LeakyReLULayer();
	MemoryManager::getInstance().add(l);
	return *l;
}

UpSampling2DLayer& UpSampling2D(int size)
{
	UpSampling2DLayer* l = new UpSampling2DLayer(size);
	MemoryManager::getInstance().add(l);
	return *l;
}

ConcatenateLayer& Concatenate()
{
	ConcatenateLayer* l = new ConcatenateLayer();
	MemoryManager::getInstance().add(l);
	return *l;
}

SoftmaxLayer& Softmax()
{
	SoftmaxLayer* l = new SoftmaxLayer();
	MemoryManager::getInstance().add(l);
	return *l;
}

ArgmaxLayer& Argmax()
{
	ArgmaxLayer* l = new ArgmaxLayer();
	MemoryManager::getInstance().add(l);
	return *l;
}