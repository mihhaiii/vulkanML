#include "TensorUtils.h"
#include "MemoryManager.h"
#include "SequentialModel.h"
#include <time.h>
#include <algorithm>
#include <iostream>
#include <stdlib.h>
#include <cassert>
#include "Tensor.h"



SequentialModel::SequentialModel(EnumDevice device)
	: m_device(device),
	numConv(0), numFC(0), numRELU(0), numMaxPool(0), numLeakyRELU(0), numBatchNorm(0), numUpSampling2D(0), numConcatenate(0)
{
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

InputLayer* SequentialModel::addInputLayer(const Shape& shape, const char* binFile) {
	InputLayer* inputLayer = new InputLayer(shape, binFile);
	inputLayer->setDevice(m_device);
	m_layers.push_back(inputLayer);
	m_layerNames.push_back("input");
	return (InputLayer*)m_layers.back();
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

Conv2dLayer* SequentialModel::addConv2dLayer(int filters, int size, int stride, const std::string& padding, bool useBias, const char* weigthsFile, const char* biasesFile ) {
	Tensor* prevLayerOutput = m_layers.back()->getOutputTensor();
	Conv2dLayer* layer = new Conv2dLayer(filters, size, stride, padding, useBias);
	layer->setDevice(m_device);
	layer->of({ prevLayerOutput }); // link layer input
	m_layers.push_back(layer);
	if (weigthsFile) layer->readWeights(weigthsFile);
	if (biasesFile) layer->readBiases(biasesFile);
	m_layerNames.push_back("conv_" + std::to_string(numConv++));
	return layer;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

MaxPooling2dLayer* SequentialModel::addMaxPooling2dLayer(int size, int stride, const std::string& padding) {
	Tensor* prevLayerOutput = m_layers.back()->getOutputTensor();
	MaxPooling2dLayer* layer = new MaxPooling2dLayer(size, stride, padding);
	layer->setDevice(m_device);
	layer->of({ prevLayerOutput }); // link layer input
	m_layers.push_back(layer);
	m_layerNames.push_back("max_pool_" + std::to_string(numMaxPool++));
	return (MaxPooling2dLayer*)m_layers.back();
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

FCLayer* SequentialModel::addFCLayer(int numNeurons, bool applyRelu, const char* weigthsFile, const char* biasesFile) {
	Tensor* prevLayerOutput = m_layers.back()->getOutputTensor();
	FCLayer* layer = new FCLayer(numNeurons, applyRelu);
	layer->setDevice(m_device);
	layer->of({ prevLayerOutput }); // link layer input
	m_layers.push_back(layer);
	if (weigthsFile) layer->readWeights(weigthsFile);
	if (biasesFile) layer->readBiases(biasesFile);
	m_layerNames.push_back("fc_" + std::to_string(numFC++));
	return layer;
}

ReLULayer* SequentialModel::addReLULayer() {
	Tensor* prevLayerOutput = m_layers.back()->getOutputTensor();
	ReLULayer* layer = new ReLULayer();
	layer->setDevice(m_device);
	layer->of({ prevLayerOutput }); // link layer input
	m_layers.push_back(layer);
	m_layerNames.push_back("relu_" + std::to_string(numRELU++));
	return layer;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

LeakyReLULayer* SequentialModel::addLeakyReLULayer() {
	Tensor* prevLayerOutput = m_layers.back()->getOutputTensor();
	LeakyReLULayer* layer = new LeakyReLULayer();
	layer->setDevice(m_device);
	layer->of({ prevLayerOutput }); // link layer input
	m_layers.push_back(layer);
	m_layerNames.push_back("leaky_relu_" + std::to_string(numLeakyRELU++));
	return layer;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

UpSampling2DLayer* SequentialModel::addUpSampling2DLayer(int size) {
	Tensor* prevLayerOutput = m_layers.back()->getOutputTensor();
	UpSampling2DLayer* layer = new UpSampling2DLayer(size);
	layer->setDevice(m_device);
	layer->of({ prevLayerOutput }); // link layer input
	m_layers.push_back(layer);
	m_layerNames.push_back("UpSampling2DLayer_" + std::to_string(numLeakyRELU++));
	return layer;
}
ConcatenateLayer* SequentialModel::addConcatenateLayer(int size) {
	Tensor* prevLayerOutput = m_layers.back()->getOutputTensor();
	ConcatenateLayer* layer = new ConcatenateLayer();
	layer->setDevice(m_device);
	layer->of({ prevLayerOutput }); // link layer input
	m_layers.push_back(layer);
	m_layerNames.push_back("ConcatenateLayer_" + std::to_string(numConcatenate++));
	return layer;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

BatchNormalizationLayer* SequentialModel::addBatchNormalizationLayer(const char* paramsFilename) {
	Tensor* prevLayerOutput = m_layers.back()->getOutputTensor();
	BatchNormalizationLayer* layer = new BatchNormalizationLayer();
	layer->setDevice(m_device);
	layer->of({ prevLayerOutput }); // link layer input
	if (paramsFilename) {
		layer->readParams(paramsFilename);
	}
	m_layers.push_back(layer);
	m_layerNames.push_back("batch_norm_" + std::to_string(numBatchNorm++));
	return layer;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

float* SequentialModel::getOutputData() {
	Tensor* outputTensor = m_layers.back()->getOutputTensor();
	return outputTensor->getData();
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int SequentialModel::getOutputDataSize() {
	return m_layers.back()->getOutputTensor()->getSize();
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void SequentialModel::run() {
	// skip the first layer which is the input layer
	for (int i = 1; i < m_layers.size(); i++) {
#ifndef NDEBUG
		clock_t start = clock();
#endif
		m_layers[i]->forward();
#ifndef NDEBUG
		std::cout << m_layerNames[i] << ": " << (float)(clock() - start) / CLOCKS_PER_SEC << std::endl;
#endif
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

SequentialModel::~SequentialModel() {
	for (auto& l : m_layers) {
		delete l;
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////