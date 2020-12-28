#include "InferenceFramework.h"
#include "TensorUtils.h"
#include "MemoryManager.h"

std::vector<Tensor*> split(Tensor* t, const std::vector<int>& dims, int axis)
{
	assert(t->getDevice() == EnumDevice::DEVICE_CPU); // only supported on cpu
	Shape sh = t->getShape();
	if (axis == -1) axis = sh.size() - 1;
	std::vector<Tensor*> tensors(dims.size());
	int sum = 0;
	for (int i = 0; i < dims.size(); i++) {
		assert(dims[i] != 0);
		sum += dims[i];
	}
	assert(sum == sh[axis]);
	
	sum = 0;
	for (int i = 0; i < dims.size(); i++) {
		sh[axis] = dims[i];
		tensors[i] = new Tensor(sh, DEVICE_CPU);
		MemoryManager::getInstance().add(tensors[i]);

		Shape it = tensors[i]->begin();
		while (true)
		{
			Shape fromIt = it;
			fromIt[axis] += sum;
			tensors[i]->at(it) = t->at(fromIt);
			bool res = tensors[i]->inc(it);
			if (!res)
				break;
		}

		sum += dims[i];
	}
	return tensors;
}

float sigmoid(float x) {
	return 1.f / (1.f + exp(-x));
}

void tensor_sigmoid(Tensor* t) {
	assert(t->getDevice() == EnumDevice::DEVICE_CPU); // only supported on cpu
	for (int i = 0; i < t->getSize(); i++) {
		t->getData()[i] = sigmoid(t->getData()[i]);
	}
}

void tensor_exp(Tensor* t) {
	assert(t->getDevice() == EnumDevice::DEVICE_CPU); // only supported on cpu
	for (int i = 0; i < t->getSize(); i++) {
		t->getData()[i] = exp(t->getData()[i]);
	}
}