#pragma once
#include <set>
#include <vector>
#include "Layers.h"

class Model
{
public:
	Model(Tensor* input, Tensor* output, EnumDevice device = DEVICE_CPU);
	Model(const std::vector<Tensor*>& inputs, Tensor* output, EnumDevice device = DEVICE_CPU);

	Tensor* run(const float* inputData, const int size);
	Tensor* run(const std::vector<float>& input);
	Tensor* run(Tensor* inputTensor);
	Tensor* run();
	Tensor* run(const std::vector<std::vector<float>>& inputs);

	std::vector<Layer*>& layers() { return sortedLayers; }

	void readWeights(const char* filename, int ignoreFirst = 0);
	void readDarknetWeights(const char* filename, int ignoreFirst = 0);

	int getParamCount();
	void showInfo();

	void fit(Tensor* x, Tensor* y, int epochs = 1);

private:
	void setDevice();
	void randomWeightInit();
	void topologicalSort();
	void dfs(Layer* l, std::set<Layer*>& visited);
	void setBatchSize();
	void setLearningRate();

private:
	std::vector<Tensor*> m_inputs;
	Tensor* m_output;
	std::vector<Layer*> sortedLayers;

	int batch_size;
	float learning_rate;
	EnumDevice m_device;
};
