#pragma once
#include "Layers.h"


class SequentialModel
{
public:
	SequentialModel(EnumDevice device);

	InputLayer* addInputLayer(const Shape& shape, const char* binFile = nullptr);
	Conv2dLayer* addConv2dLayer(int filters, int size, int stride, const std::string& padding = "valid", bool useBias = true, const char* weigthsFile = nullptr, const char* biasesFile = nullptr);
	MaxPooling2dLayer* addMaxPooling2dLayer(int size, int stride, const std::string& padding = "valid");
	FCLayer* addFCLayer(int numNeurons, bool applyRelu, const char* weigthsFile = nullptr, const char* biasesFile = nullptr);
	ReLULayer* addReLULayer();
	LeakyReLULayer* addLeakyReLULayer();
	UpSampling2DLayer* addUpSampling2DLayer(int size);
	ConcatenateLayer* addConcatenateLayer(int size);
	BatchNormalizationLayer* addBatchNormalizationLayer(const char* paramsFilename = nullptr);

	float* getOutputData();
	int getOutputDataSize();
	
	void run();

	~SequentialModel();

private:
	std::vector<Layer*> m_layers;
	EnumDevice m_device;

	int numConv, numFC, numRELU, numMaxPool, numLeakyRELU, numBatchNorm, numUpSampling2D, numConcatenate;
	std::vector<std::string> m_layerNames;
};