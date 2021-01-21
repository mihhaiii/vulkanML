#pragma once
#include <set>
#include <vector>
#include "Layers.h"

class Model
{
public:
	Model(Tensor* input, Tensor* output, EnumDevice device = DEVICE_CPU);
	Model(const std::vector<Tensor*>& inputs, Tensor* output, EnumDevice device = DEVICE_CPU);

	Tensor* run(float* inputData, const int size, bool training = false);
	Tensor* run(std::vector<float>& input, bool training = false);
	Tensor* run(Tensor* inputTensor, bool training = false);
	Tensor* run();
	Tensor* run(std::vector<std::vector<float>>& inputs);

	std::vector<Layer*>& layers() { return sortedLayers; }

	void readWeights(const char* filename, int ignoreFirst = 0);
	void readDarknetWeights(const char* filename, int ignoreFirst = 0);

	int getParamCount();
	void showInfo();

	void fit(Tensor* x, Tensor* y, int epochs = 1, Tensor* test_x = nullptr, Tensor* test_y = nullptr, bool printAccuracy = true);

private:
	void setDevice();
	void randomWeightInit();
	void topologicalSort();
	void dfs(Layer* l, std::set<Layer*>& visited);
	void setBatchSize(float batch_size, bool training = false);
	void setLearningRate(float learning_rate);

private:
	std::vector<Tensor*> m_inputs;
	Tensor* m_output;
	std::vector<Layer*> sortedLayers;

	int m_batch_size;
	float m_learning_rate;
	EnumDevice m_device;
};
