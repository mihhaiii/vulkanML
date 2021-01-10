#pragma once
#include <set>
#include <vector>
#include "Layers.h"

class Model
{
public:
	Model(Tensor* input, Tensor* output, EnumDevice device = DEVICE_CPU);
	Model(const std::vector<Tensor*>& inputs, Tensor* output, EnumDevice device = DEVICE_CPU);

	Tensor* run(const std::vector<std::vector<float>>& inputs);
	Tensor* run(const std::vector<float>& input);
	Tensor* run(float* input, const int size);
	Tensor* run();

	std::vector<Layer*>& layers() { return sortedLayers; }

	void readWeights(const char* filename, int ignoreFirst = 0);
	void readDarknetWeights(const char* filename, int ignoreFirst = 0);

	int getParamCount();
	void showInfo();

	void fit(Tensor* x, Tensor* y);

private:
	void setDevice();
	void randomWeightInit();
	void topologicalSort();
	void dfs(Layer* l, std::set<Layer*>& visited);

private:
	std::vector<Tensor*> m_inputs;
	Tensor* m_output;
	std::vector<Layer*> sortedLayers;

	EnumDevice m_device;
};
