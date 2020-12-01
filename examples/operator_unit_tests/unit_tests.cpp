#include "unit_tests.h"
#include <iostream>
#include "InferenceFramework.h"

static float sqr(float a) {
	return a * a;
}

static void batch_norm_test1()
{
	Model m1(DEVICE_CPU);
	Model m2(DEVICE_VULKAN);

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


void test_batch_norm()
{
	std::cout << "Running batch norm tests..." << std::endl;
	
	batch_norm_test1();	
}