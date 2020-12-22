#include "unit_tests.h"
#include <iostream>
#include "InferenceFramework.h"
#include "OperatorFunctionInterface.h"
#include "TensorUtils.h"

static float sqr(float a) {
	return a * a;
}

static void batch_norm_test1()
{
	SequentialModel m1(DEVICE_CPU);
	SequentialModel m2(DEVICE_VULKAN);

	const int n = 10;
	std::vector<float> data(n);
	float mean = 0, variance = 0;
	for (int i = 0; i < 10; i++) {
		data[i] = i;
		mean += data[i];
	}
	mean /= n;
	for (int i = 0; i < n; i++) {
		variance += sqr(data[i] - mean);
	}
	variance /= n;

	m1.addInputLayer({ n, 1 })->fill(data);
	m2.addInputLayer({ n, 1 })->fill(data);

	const float gamma = 10; // scale 
	const float beta = 100; // center
	m1.addBatchNormalizationLayer()->setParams({ gamma,beta, mean, variance });
	m2.addBatchNormalizationLayer()->setParams({ gamma,beta, mean, variance });
	
	m1.run();
	m2.run();

	float* output_cpu = m1.getOutputData();
	float* output_vulkan = m2.getOutputData();
	int size = m1.getOutputDataSize();

	for (int i = 0; i < size; i++) {
		assert(abs(output_cpu[i] - output_vulkan[i]) < 0.001);
	}
}


void upSampling2D_test1()
{
	Tensor* input = Input({ 2,3,2 });
	Tensor* x = UpSampling2D(2)(input);

	Model* m1 = new Model(input, x, EnumDevice::DEVICE_CPU);
	Model* m2 = new Model(input, x, EnumDevice::DEVICE_VULKAN);

	std::vector<float> data = {
		1, 7, 2, 8, 3, 9,
		4, 10, 5, 11, 6, 12
	};

	float* output1 = m1->run(data)->getData();
	float* output2 = m2->run(data)->getData();

	int size = x->getSize();
	std::vector<float> out1(output1, output1 + size);
	std::vector<float> out2(output2, output2 + size);
	for (int i = 0; i < size; i++) {
		assert(abs(out1[i] - out2[i]) < 0.001);
	}
}

void concatenateLayer_test1()
{
	Tensor* input1 = Input({ 2,3,2 });
	Tensor* input2 = Input({ 2,3,1 });

	std::vector<float> data1 = {
		1, 7, 2, 8, 3, 9,
		4, 10, 5, 11, 6, 12
	};
	std::vector<float> data2 = {
		13, 14, 15,
		16, 17, 18
	};

	Tensor* x = Concatenate()({ input1, input2 });

	Model* m1 = new Model({ input1, input2 }, x, EnumDevice::DEVICE_CPU);
	Model* m2 = new Model({ input1, input2 }, x, EnumDevice::DEVICE_VULKAN);

	float* output1 = m1->run({ data1, data2 })->getData();
	float* output2 = m2->run({ data1, data2 })->getData();

	int size = x->getSize();
	std::vector<float> out1(output1, output1 + size);
	std::vector<float> out2(output2, output2 + size);
	for (int i = 0; i < size; i++) {
		assert(abs(out1[i] - out2[i]) < 0.001);
	}
}

void test_batch_norm()
{
	std::cout << "Running batch norm tests..." << std::endl;
	
	batch_norm_test1();	
}

void test_UpSampling2D()
{
	std::cout << "Running upSampling2D  tests..." << std::endl;

	upSampling2D_test1();
}
void test_concatenate()
{
	std::cout << "Running concatenate layer tests..." << std::endl;

	concatenateLayer_test1();
}

void test_tensor_split_1()
{
	Tensor* t = new Tensor({ 2,3,4 }, DEVICE_CPU);
	std::vector<float> data = {
		1,2,2,3,  1,2,2,3,  1,2,2,3,
		1,2,2,3,  1,2,2,3,  1,2,2,3
	};
	t->setData(data);
	auto ts = split(t, { 1,2,1 }, -1);
	auto t1 = ts[0];
	auto t2 = ts[1];
	auto t3 = ts[2];
	assert(t1->getSize() == 6);
	assert(t2->getSize() == 12);
	assert(t3->getSize() == 6);
	for (int ti = 0; ti < ts.size(); ti++) {
		for (int i = 0; i < ts[ti]->getSize(); i++) {
			int x = ts[ti]->getData()[i];
			assert(x == ti + 1);
		}
	}
}

void test_tensor_split()
{
	std::cout << "Running tensor split tests..." << std::endl;

	test_tensor_split_1();
}