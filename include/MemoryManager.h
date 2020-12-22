#pragma once
#include <vector>
#include "InferenceFramework.h"

class Layer;
class MemoryManager
{
public:
	static MemoryManager& getInstance() {
		static MemoryManager obj;
		return obj;
	}

	void add(Layer* l) {
		layers.push_back(l);
	}
	void add(Tensor* t) {
		tensors.push_back(t);
	}

	~MemoryManager()
	{
		for (Layer* l : layers) {
			delete l;
		}
		for (Tensor* t : tensors) {
			delete t;
		}
	}

	MemoryManager(const  MemoryManager& other) = delete;
	MemoryManager& operator=(const  MemoryManager& other) = delete;
private:

	MemoryManager() {}

	std::vector<Layer*> layers;
	std::vector<Tensor*> tensors;
};