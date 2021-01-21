#include "TensorUtils.h"
#include "MemoryManager.h"
#include "Tensor.h"
#include "vulkan_operator_reduce_count_eq.h"

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

int CountEqual(Tensor* t1, Tensor* t2)
{
	assert(t1->getDevice() == t2->getDevice());
	int size = t1->getSize();
	assert(size == t2->getSize());
	if (t1->getDevice() == DEVICE_CPU) {
		int res = 0;
		float* data1 = t1->getData();
		float* data2 = t2->getData();
		for (int i = 0; i < size; i++) {
			res += (data1[i] == data2[i]);
		}
		return res;
	}
	else if (t1->getDevice() == DEVICE_VULKAN) {
		// ! changes device data of t1
		return vulkan_operator_reduce_count_eq(t1->getDeviceData(), t2->getDeviceData(), size);
	}
	return 0;
}
