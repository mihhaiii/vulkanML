#include <iostream>
#include <cassert>
#include "vulkan_operator_FC.h"
#include "operator_FC_cpu.h"
#include <ctime>
#include <vector>
#define STB_IMAGE_IMPLEMENTATION
#define NOMINMAX
#include "stb_image.h"
#include "operator_softmax_cpu.h"
#include <algorithm>
#include "vulkan_operator_reduce_max.h"
#include "vulkan_operator_reduce_sum.h"
#include <numeric>
#include "vulkan_operator_map_add.h"
#include "vulkan_operator_map_div.h"
#include "vulkan_operator_map_exp.h"
#include "vulkan_operator_softmax.h"
#include "vulkan_operator_relu.h"
#include "test_conv_operators.h"
#include "operator_conv2d_cpu.h"
#include "operator_maxpool_cpu.h"
#include "vulkan_operator_conv2d.h"
#include "vulkan_operator_maxpool.h"
#include "InferenceFramework.h"
#include <fstream>
#include "InstanceManager.h"

void test_vulkan_operator_FC()
{
	const int numInputs = 10000;
	const int numOutputs = 2000;
	float * inputs = new float[numInputs];
	float * outputs_vk = new float[numOutputs];
	float * outputs_cpu = new float[numOutputs];
	float * biases = new float[numOutputs];
	float * weights = new float[numInputs * numOutputs];

	for (int i = 0; i < numInputs; i++) {
		inputs[i] = (float)rand() / (RAND_MAX);
	}
	for (int i = 0; i < numOutputs; i++) {
		for (int j = 0; j < numInputs; j++) {
			weights[i * numInputs + j] = (float)rand() / (RAND_MAX);
		}
		biases[i] = (float)rand() / (RAND_MAX);
	}
	float timeVulkan = 0;
	vulkan_operator_FC(inputs, weights, biases, outputs_vk, numInputs, numOutputs, true/*apply relu*/, &timeVulkan);
	clock_t start = clock();
	operator_FC_cpu(inputs, weights, biases, outputs_cpu, numInputs, numOutputs);
	float timeCPU = (float)(clock() - start) / CLOCKS_PER_SEC;

	for (int i = 0; i < numOutputs; i++) {
		assert(abs(outputs_cpu[i] - outputs_vk[i]) <= 0.1);
	}

	std::cout << "CPU Time: " << timeCPU << "\n" << "Vulkan Time: " << timeVulkan << "\n";

	delete[] inputs;
	delete[] weights;
	delete[] biases;
	delete[] outputs_vk;
	delete[] outputs_cpu;
}

void test_vulkan_reduce()
{
	// test reduce max and sum operators

	const int numInputs = 500;
	std::vector<float> input(numInputs);
	for (int i = 0; i < numInputs; i++) {
		input[i] = (float)rand() / (RAND_MAX);
	}
	float maxVulkan;
	vulkan_operator_reduce_max(&input[0], numInputs, &maxVulkan);
	float maxCpu = *std::max_element(input.begin(), input.end());
	if (abs(maxVulkan - maxCpu) > 0.1) {
		std::cout << "error: reduce max produces different results with Vulkan and cpu\n";
	}

	float sumVulkan;
	vulkan_operator_reduce_sum(&input[0], numInputs, &sumVulkan);
	float sumCpu = std::accumulate(input.begin(), input.end(), 0.);
	if (abs(sumVulkan - sumCpu) > 0.1) {
		std::cout << "error: reduce max produces different results with Vulkan and cpu\n";
	}
}


void test_vulkan_map()
{
	const int numInputs = 500;
	std::vector<float> input(numInputs);
	for (int i = 0; i < numInputs; i++) {
		input[i] = (float)rand() / (RAND_MAX);
	}
	std::vector<float> resVulkan(numInputs);
	const float addScalar = 1.;
	vulkan_operator_map_add(&input[0], numInputs, &resVulkan[0], addScalar);
	// check correctness
	for (int i = 0; i < numInputs; i++) {
		if (abs(resVulkan[i] - (input[i] + addScalar)) >= 0.1) {
			std::cout << "error: map add scalar produces different results with Vulkan and cpu\n";
		}
	}
	const float divScalar = 2.;
	vulkan_operator_map_div(&input[0], numInputs, &resVulkan[0], divScalar);
	// check correctness
	for (int i = 0; i < numInputs; i++) {
		if (abs(resVulkan[i] - (input[i] / divScalar)) >= 0.1) {
			std::cout << "error: map div scalar produces different results with Vulkan and cpu\n";
		}
	}

	vulkan_operator_map_exp(&input[0], numInputs, &resVulkan[0]);
	// check correctness
	for (int i = 0; i < numInputs; i++) {
		if (abs(resVulkan[i] - exp(input[i])) >= 0.1) {
			std::cout << "error: map exp produces different results with Vulkan and cpu\n";
		}
	}
}

void test_vulkan_softmax()
{
	const int numInputs = 500;
	std::vector<float> input(numInputs);
	for (int i = 0; i < numInputs; i++) {
		input[i] = (float)rand() / (RAND_MAX);
	}
	std::vector<float> outputVk(numInputs);
	std::vector<float> outputCpu(numInputs);
	vulkan_operator_softmax(&input[0], &outputVk[0], numInputs);
	operator_softmax_cpu(&input[0], &outputCpu[0], numInputs);

	for (int i = 0; i < numInputs; i++) {
		if (abs(outputVk[i] - outputCpu[i]) >= 0.1) {
			std::cout << "error: softmax produces different results with Vulkan and cpu\n";
		}
	}
}

void test_operator_relu() 
{
	const int numInputs = 500;
	std::vector<float> input(numInputs);
	for (int i = 0; i < numInputs; i++) {
		input[i] = (float)rand() / (RAND_MAX);
	}
	std::vector<float> output(numInputs);
	vulkan_operator_relu(&input[0], &output[0], numInputs);
	for (int i = 0; i < numInputs; i++) {
		if (abs(output[i] - std::max(0.f, input[i])) > 0.1) {
			std::cout << "error: relu produces different results with Vulkan and cpu\n";
		}
	}
}

void test_lenet()
{
	const int numInputs = 784;
	const int numOutputs = 10;
	//read weights
	std::vector<float> weights(numInputs * numOutputs);
	FILE* weightsFile = fopen("Data/dense1_weights.bin", "rb");
	fread(&weights[0], sizeof(float), weights.size(), weightsFile);
	fclose(weightsFile);

	// read biases
	std::vector<float> biases(numOutputs);
	FILE* biasesFile = fopen("Data/dense1_bias.bin", "rb");
	fread(&biases[0], sizeof(float), biases.size(), biasesFile);
	fclose(biasesFile);

	// read input
	std::vector<float> input(numInputs);
	FILE* inputFile = fopen("Data/Example_0.bin", "rb");
	fread(&input[0], sizeof(float), input.size(), inputFile);
	fclose(inputFile);

	std::vector<float> outputFC(numOutputs);
	std::vector<float> outputSoftmax(numOutputs);
	std::vector<float> outputSoftmaxVk(numOutputs);

	// run model
	vulkan_operator_FC(&input[0], &weights[0], &biases[0], &outputFC[0], numInputs, numOutputs);
	operator_softmax_cpu(&outputFC[0], &outputSoftmax[0], numOutputs);
	vulkan_operator_softmax(&outputFC[0], &outputSoftmaxVk[0], numOutputs);
	auto itMax = std::max_element(outputSoftmax.begin(), outputSoftmax.end());
	int digit = itMax - outputSoftmax.begin();
	float prob = *itMax;
	std::cout << "Digit is " << digit << " with probability " << prob << '\n';
}

void readBinFile(const char * fileName, std::vector<float>& outVec) {
	FILE* file = fopen(fileName, "rb");
	int read = fread(&outVec[0], sizeof(float), outVec.size(), file);
	fclose(file);
};

std::vector<float> loadImage(const char * filename, int& width, int& height, int& channels) {
	// read an input image
	unsigned char *image = stbi_load(filename,
		&width,
		&height,
		&channels,
		STBI_grey);
	std::vector<float> img(width * height * channels);
	for (int i = 0; i < img.size(); i++) {
		img[i] = (float)image[i] / 255.0;
	}
	return img;
}
void test_mnist_simple()
{
	// read model's weights and biases
	const int numInputs = 784;
	const int fc1_numNeurons = 128;
	const int numOutputs = 10;
	const char* fc1_WeigthsFile = "model_weights/fc1_weights_file.bin";
	const char* fc1_BiasesFile = "model_weights/fc1_biases_file.bin";
	const char* fc2_WeigthsFile = "model_weights/fc2_weights_file.bin";
	const char* fc2_BiasesFile = "model_weights/fc2_biases_file.bin";

	std::vector<float> fc1_Weights(numInputs * fc1_numNeurons);
	std::vector<float> fc1_Biases(fc1_numNeurons);
	std::vector<float> fc2_Weights(fc1_numNeurons * numOutputs);
	std::vector<float> fc2_Biases(numOutputs);

	/*auto readBinFile = [](const char * fileName, std::vector<float>& outVec) {
		FILE* file = fopen(fileName, "rb");
		int read = fread(&outVec[0], sizeof(float), outVec.size(), file);
		fclose(file);
	};*/

	readBinFile(fc1_WeigthsFile, fc1_Weights);
	readBinFile(fc1_BiasesFile, fc1_Biases);
	readBinFile(fc2_WeigthsFile, fc2_Weights);
	readBinFile(fc2_BiasesFile, fc2_Biases);

	// read an input image
	const char * inputImageFile = "trainingSample/3/img_13.jpg";
	int width, height, channels;
	unsigned char *image = stbi_load(inputImageFile,
		&width,
		&height,
		&channels,
		STBI_grey);
	assert(width * height * channels == numInputs);
	std::vector<float> img(numInputs);
	for (int i = 0; i < numInputs; i++) {
		img[i] = (float)image[i] / 255.0;
	}
	auto transpose = [](std::vector<float>& v, int n, int m){
		std::vector<std::vector<float>> newv(m, std::vector<float>(n));
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < m; j++) {
				int k = i * m + j;
				newv[j][i] = v[k];
			}
		}
		int k = 0;
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				v[k++] = newv[i][j];
			}
		}
	};
	//transpose(fc1_Weights, numInputs, fc1_numNeurons);
	//transpose(fc2_Weights, fc1_numNeurons, numOutputs);

	std::vector<float> fc1_outputs(fc1_numNeurons);
	std::vector<float> fc2_outputs(numOutputs);
	vulkan_operator_FC(&img[0], &fc1_Weights[0], &fc1_Biases[0], 
		&fc1_outputs[0], numInputs, fc1_numNeurons);
	vulkan_operator_relu(&fc1_outputs[0], &fc1_outputs[0], fc1_numNeurons);
	vulkan_operator_FC(&fc1_outputs[0], &fc2_Weights[0], &fc2_Biases[0],
		&fc2_outputs[0], fc1_numNeurons, numOutputs);
	vulkan_operator_softmax(&fc2_outputs[0], &fc2_outputs[0], numOutputs);
	auto itMax = std::max_element(fc2_outputs.begin(), fc2_outputs.end());
	int digit = itMax - fc2_outputs.begin();
	float prob = *itMax;
	std::cout << "Digit is " << digit << " with probability " << prob << '\n';
}

//void test_conv_net()
//{
//	// layers as follows:
//	// layer 0:  conv, weights 3x3x1x32 biases 32
//	// layer :   max pool 2x2
//	// layer 1:  conv, weights 3x3x32x64 biases 64
//	// layer :   max pool 2x2
//	// layer 2:  conv weights 3x3x64x64 biases  64
//	// layer 3:  fc   weights 576x64    biases  64
//	// layer 4:  fc   weights 64x10		biases  10
//	std::vector<float> conv0_w(3 * 3 * 32);
//	std::vector<float> conv0_b(32);
//	std::vector<float> conv1_w(3 * 3 * 32 * 64);
//	std::vector<float> conv1_b(64);
//	std::vector<float> conv2_w(3 * 3 * 64 * 64);
//	std::vector<float> conv2_b(64);
//	std::vector<float> fc0_w(576 * 64);
//	std::vector<float> fc0_b(64);
//	std::vector<float> fc1_w(64 * 10);
//	std::vector<float> fc1_b(10);
//	readBinFile("conv_mnist/layer_0_weights", conv0_w);
//	readBinFile("conv_mnist/layer_0_biases", conv0_b);
//	readBinFile("conv_mnist/layer_1_weights", conv1_w);
//	readBinFile("conv_mnist/layer_1_biases", conv1_b);
//	readBinFile("conv_mnist/layer_2_weights", conv2_w);
//	readBinFile("conv_mnist/layer_2_biases", conv2_b);
//	readBinFile("conv_mnist/layer_3_weights", fc0_w);
//	readBinFile("conv_mnist/layer_3_biases", fc0_b);
//	readBinFile("conv_mnist/layer_4_weights", fc1_w);
//	readBinFile("conv_mnist/layer_4_biases", fc1_b);
//	
//	int width, height, channels;
//	std::vector<float> img = loadImage("trainingSample/2/img_73.jpg", width, height, channels);
//
//	std::vector<float> tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;
//	tmp1.resize(26*26*32);
//	tmp2.resize(13 * 13 * 32);
//	tmp3.resize(11 * 11 * 64);
//	tmp4.resize(5 * 5 * 64);
//	tmp5.resize(3 * 3 * 64);
//	tmp6.resize(64);
//	tmp7.resize(10);
//
//	vulkan_operator_conv2d(&img[0], height, width, channels, &conv0_w[0], 3, 3, &conv0_b[0], &tmp1[0], 32);
//	//vulkan_operator_relu(&tmp1[0], &tmp1[0], tmp1.size());
//	vulkan_operator_maxpool(&tmp1[0], 26, 26, 32, 2, 2, &tmp2[0]);
//
//	vulkan_operator_conv2d(&tmp2[0], 13, 13, 32, &conv1_w[0], 3, 3, &conv1_b[0], &tmp3[0], 64);
//	//vulkan_operator_relu(&tmp3[0], &tmp3[0], tmp3.size());
//	vulkan_operator_maxpool(&tmp3[0], 11, 11, 64, 2, 2, &tmp4[0]);
//
//	vulkan_operator_conv2d(&tmp4[0], 5, 5, 64, &conv2_w[0], 3, 3, &conv2_b[0], &tmp5[0], 64);
//	//vulkan_operator_relu(&tmp5[0], &tmp5[0], tmp5.size());
//
//	vulkan_operator_FC(&tmp5[0], &fc0_w[0], &fc0_b[0], &tmp6[0], tmp5.size(), tmp6.size(), true/*apply relu*/);
//	//vulkan_operator_relu(&tmp6[0], &tmp6[0], tmp6.size());
//
//
//	vulkan_operator_FC(&tmp6[0], &fc1_w[0], &fc1_b[0], &tmp7[0], tmp6.size(), tmp7.size());
//
//	std::ofstream outF("out1.txt");
//	outF << tmp7.size() << '\n';
//	for (int i = 0; i < tmp7.size(); i++) {
//		outF << tmp7[i] << ' ';
//	}
//
//
//	operator_softmax_cpu(&tmp7[0], &tmp7[0], tmp7.size());
//	auto itMax = std::max_element(tmp7.begin(), tmp7.end());
//	int digit = itMax - tmp7.begin();
//	float prob = *itMax;
//	std::cout << "Digit is " << digit << " with probability " << prob << '\n';
//}

void test_conv_net1()
{
	Model m(EnumDevice::DEVICE_VULKAN);

	int h, w, c;
	std::vector<float> img = loadImage("trainingSample/1/img_182.jpg",h,w,c);
	InputLayer* inputLayer = m.addInputLayer({ 28,28,1 });
	inputLayer->fill(img);

	m.addConv2dLayer(32, 3, 1, "valid", true, "conv_mnist/layer_0_weights", "conv_mnist/layer_0_biases");
	m.addReLULayer();
	m.addMaxPooling2dLayer(2, 2);

	m.addConv2dLayer(64, 3, 1, "valid", true, "conv_mnist/layer_1_weights", "conv_mnist/layer_1_biases");
	m.addReLULayer();
	m.addMaxPooling2dLayer(2, 2);

	m.addConv2dLayer(64, 3, 1, "valid", true, "conv_mnist/layer_2_weights", "conv_mnist/layer_2_biases");
	m.addReLULayer();

	m.addFCLayer(64, true, "conv_mnist/layer_3_weights", "conv_mnist/layer_3_biases");
	m.addFCLayer(10, false, "conv_mnist/layer_4_weights", "conv_mnist/layer_4_biases");

	m.run();
	clock_t start = clock();
	m.run();
	float time = (float)(clock() - start) / CLOCKS_PER_SEC;

	float* output = m.getOutputData();
	operator_softmax_cpu(output, output, 10);
	auto itMax = std::max_element(output, output + 10);
	int digit = itMax - output;
	float prob = *itMax;
	std::cout << "Digit is " << digit << " with probability " << prob << '\n';
	std::cout << "Run time: " << time << '\n';
}

int main()
{
	//test_vulkan_operator_FC();

	//test_vulkan_reduce();

	//test_vulkan_map();

	//test_vulkan_softmax();

	//test_lenet();

	//test_operator_relu();
	//test_maxpool();
	//test_conv2d();

	//test_mnist_simple();

	//test_conv_net();

	test_conv_net1();
	return 0;
}