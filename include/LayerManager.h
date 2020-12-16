#pragma once
#include <vector>
#include "InferenceFramework.h"

class Layer;
class LayerManager
{
public:
	static LayerManager& getInstance() {
		static LayerManager obj;
		return obj;
	}

	void add(Layer* l) {
		layers.push_back(l);
	}

	~LayerManager()
	{
		for (Layer* l : layers) {
			delete l;
		}
	}

	LayerManager(const  LayerManager& other) = delete;
	LayerManager& operator=(const  LayerManager& other) = delete;
private:

	LayerManager() {}

	std::vector<Layer*> layers;
};