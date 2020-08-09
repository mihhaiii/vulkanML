#pragma once
#include "vuh/array.hpp"
#include <cassert>
#include <vuh/vuh.h>
#include "InstanceManager.h"
#include "operator_FC_cpu.h"
#include "vulkan_operator_FC.h"
#include "operator_maxpool_cpu.h"
#include "vulkan_operator_maxpool.h"
#include "operator_conv2d_cpu.h"
#include "vulkan_operator_conv2d.h"
#include "vulkan_operator_relu.h"
#include <iostream>

using Shape = std::vector<int>;

int getShapeSize(const Shape& shape) {
	int total = 1;
	assert(shape.size() != 0);
	for (auto& dim : shape) {
		total *= dim;
	}
	return total;
}

enum EnumDevice
{
	DEVICE_CPU = 1,
	DEVICE_VULKAN = 2,
	DEVICE_ALL = DEVICE_CPU | DEVICE_VULKAN
};


void readBinFile(const char * fileName, float* out, int count) {
	FILE* file = fopen(fileName, "rb");
	int read = fread(out, sizeof(float), count, file);
	//assert(read == count);
	if (read != count) {
		for (int i = 0; i < count; i++) {
			out[i] = 0;
		}
	}
	fclose(file);
};


class Tensor
{
public:
	Tensor(const Shape& shape, EnumDevice& device, const char* binFile = nullptr)
		: m_shape(shape),
		m_device(device),
		m_size(getShapeSize(shape)),
		m_data(nullptr),
		m_deviceData(nullptr)
	{
		if (m_device & DEVICE_CPU)
			m_data = new float[m_size];

		if (m_device & DEVICE_VULKAN)
		{
			m_deviceData = new vuh::Array<float>(InstanceManger::getInstance().getDefaultDevice(), m_size);
		}

		if (binFile) {
			readData(binFile);
		}
	}

	void readData(const char* binFile) {
		if (!m_data)
			m_data = new float[m_size];
		readBinFile(binFile, m_data, m_size);
		if (m_device & DEVICE_VULKAN) {
			m_deviceData->fromHost(m_data, m_data + m_size);
		}	
	}

	float* getData() { return m_data; }
	vuh::Array<float>* getDeviceData() { return m_deviceData; }
	float* getDataFromDeviceData() {
		if (!m_data)
			m_data = new float[m_size];
		m_deviceData->toHost(m_data);
		return m_data;
	}
	int getSize() { return m_size; }
	Shape getShape() { return m_shape; }

	~Tensor() {
		delete[] m_data;
		delete m_deviceData;
	}

private:
	int m_size;
	Shape m_shape;
	float* m_data;
	vuh::Array<float>* m_deviceData;
	EnumDevice m_device;
};

class Layer
{
public:
	virtual void forward() = 0;
	virtual Tensor* getOutputTensor() = 0;
	virtual ~Layer() {}
};

class Conv2dLayer : public Layer
{
public:
	Conv2dLayer(int numOutputChannels, int kernelHeight, int kernelWidth, Tensor* inputImage, EnumDevice device)
		: m_numOutputChannels(numOutputChannels), m_kernelHeight(kernelHeight), m_kernelWidth(kernelWidth), m_inputImage(inputImage),
		m_device(device)
	{
		assert(m_inputImage->getShape().size() == 3); 
		m_height = m_inputImage->getShape()[0];
		m_width = m_inputImage->getShape()[1];
		m_channels = m_inputImage->getShape()[2];

		m_outputImage = new Tensor({ m_height - kernelHeight + 1, m_width - kernelWidth + 1, numOutputChannels }, m_device);
		m_kernelsWeights = new Tensor({ m_kernelHeight, m_kernelWidth, m_channels, m_numOutputChannels }, m_device);
		m_kernelsBiases = new Tensor({ m_numOutputChannels }, m_device);
	}

	void readWeights(const char* binFile) {
		m_kernelsWeights->readData(binFile);
	}

	void readBiases(const char* binFile) {
		m_kernelsBiases->readData(binFile);
	}

	virtual void forward() {
		if (m_device == DEVICE_CPU) {
			operator_conv2d_cpu(m_inputImage->getData(), m_height, m_width, m_channels, m_kernelsWeights->getData(),
				m_kernelHeight, m_kernelWidth, m_kernelsBiases->getData(), m_outputImage->getData(), m_numOutputChannels);
		}
		else if (m_device == DEVICE_VULKAN) {
			vulkan_operator_conv2d(m_inputImage->getDeviceData(), m_height, m_width, m_channels, m_kernelsWeights->getDeviceData(),
				m_kernelHeight, m_kernelWidth, m_kernelsBiases->getDeviceData(), m_outputImage->getDeviceData(), m_numOutputChannels);
		}
	}

	Tensor* getOutputTensor() { return m_outputImage; }

	~Conv2dLayer() {
		delete m_outputImage;
		delete m_kernelsWeights;
		delete m_kernelsBiases;
	}

private:
	Tensor* m_inputImage;
	Tensor* m_outputImage;
	Tensor* m_kernelsWeights;
	Tensor* m_kernelsBiases;


	int m_height;
	int m_width;
	int m_channels;
	int m_kernelHeight;
	int m_kernelWidth;
	int m_numOutputChannels;
	EnumDevice m_device;
};

class MaxPooling2dLayer : public Layer
{
public:
	MaxPooling2dLayer(int kernelHeight, int kernelWidth, Tensor* inputImage, EnumDevice device)
		: m_kernelHeight(kernelHeight), m_kernelWidth(kernelWidth), m_inputImage(inputImage), m_device(device)
	{
		assert(m_inputImage->getShape().size() == 3);
		m_height = m_inputImage->getShape()[0];
		m_width = m_inputImage->getShape()[1];
		m_channels = m_inputImage->getShape()[2];

		m_outputImage = new Tensor({ m_height / m_kernelHeight, m_width / m_kernelWidth, m_channels }, m_device);
	}

	virtual void forward() {
		if (m_device == DEVICE_CPU) {
			operator_maxpool_cpu(m_inputImage->getData(), m_height, m_width, m_channels, m_kernelHeight, m_kernelWidth,
				m_outputImage->getData());
		}
		else if (m_device == DEVICE_VULKAN) {
			vulkan_operator_maxpool(m_inputImage->getDeviceData(), m_height, m_width, m_channels, m_kernelHeight, m_kernelWidth,
				m_outputImage->getDeviceData());
		}
	}

	Tensor* getOutputTensor() { return m_outputImage; }

	~MaxPooling2dLayer() {
		delete m_outputImage;
	}

private:
	Tensor* m_inputImage;
	Tensor* m_outputImage;

	int m_height;
	int m_width;
	int m_channels;
	int m_kernelHeight;
	int m_kernelWidth;
	EnumDevice m_device;
};

class FCLayer : public Layer
{
public:
	FCLayer(int numNeurons, Tensor* input, EnumDevice device, bool applyRelu)
		: m_numNeurons(numNeurons), m_input(input), m_device(device), m_applyRelu(applyRelu)
	{
		m_weights = new Tensor({ input->getSize(), numNeurons }, m_device);
		m_biases = new Tensor({ numNeurons }, m_device);
		m_output = new Tensor({ numNeurons }, m_device);
	}

	void readWeights(const char* binFile) {
		m_weights->readData(binFile);
	}

	void readBiases(const char* binFile) {
		m_biases->readData(binFile);
	}

	virtual void forward() {
		if (m_device == DEVICE_CPU) {
			operator_FC_cpu(m_input->getData(), m_weights->getData(), m_biases->getData(), m_output->getData(),
				m_input->getSize(), m_numNeurons, m_applyRelu);
		}
		else if (m_device == DEVICE_VULKAN) {
			vulkan_operator_FC(m_input->getDeviceData(), m_weights->getDeviceData(), m_biases->getDeviceData(),
				m_output->getDeviceData(), m_input->getSize(), m_numNeurons, m_applyRelu);
		}
	}

	Tensor* getOutputTensor() { return m_output; }

	~FCLayer() {
		delete m_weights;
		delete m_biases;
		delete m_output;
	}

private:

	int m_numNeurons;
	Tensor* m_input;
	Tensor* m_weights;
	Tensor* m_biases;
	Tensor* m_output;
	EnumDevice m_device;
	bool m_applyRelu;
};

class InputLayer : public Layer
{
public:
	InputLayer(const Shape& shape, EnumDevice device, const char* binFile = nullptr)
		: m_shape(shape), m_device(device)
	{
		m_data = new Tensor(shape, device);
		if (binFile)
			m_data->readData(binFile);
	}

	void fill(std::vector<float>& src) {
		assert(src.size() == m_data->getSize());
		if (m_device == DEVICE_CPU) {
			float* dest = m_data->getData();
			for (int i = 0; i < src.size(); i++) {
				dest[i] = src[i];
			}
		}
		else if(m_device == DEVICE_VULKAN) {
			m_data->getDeviceData()->fromHost(&src[0], &src[0] + src.size());
		}
	}

	virtual void forward() {}
	Tensor* getOutputTensor() { return m_data; }

	~InputLayer() {
		delete m_data;
	}
private:
	Shape m_shape;
	Tensor* m_data;
	EnumDevice m_device;
};

class ReLULayer : public Layer
{
public:
	ReLULayer(Tensor* input, EnumDevice device)
		: m_input(input), m_device(device) {}

	virtual void forward() {
		if (m_device == EnumDevice::DEVICE_CPU) {
			for (int i = 0; i < m_input->getSize(); i++) {
				m_input->getData()[i] = std::max(m_input->getData()[i], 0.f);
			}
		}
		else if (m_device == EnumDevice::DEVICE_VULKAN) {
			vulkan_operator_relu(m_input->getDeviceData(), m_input->getDeviceData(), m_input->getSize());
		}
	}
	Tensor* getOutputTensor() { return m_input; }

private:
	Tensor* m_input;
	EnumDevice m_device;
};

class Model
{
public:
	Model(EnumDevice device)
	 : m_device(device),
		numConv(0), numFC(0), numRELU(0), numMaxPool(0)
	{
	}

	InputLayer* addInputLayer(const Shape& shape, const char* binFile = nullptr) {
		m_layers.push_back(new InputLayer(shape, m_device, binFile));
		m_layerNames.push_back("input");
		return (InputLayer*)m_layers.back();
	}

	Conv2dLayer* addConv2dLayer(int numOutputChannels, int kH, int kW, const char* weigthsFile = nullptr, const char* biasesFile = nullptr) {
		Tensor* prevLayerOutput = m_layers.back()->getOutputTensor();
		Conv2dLayer* layer = new Conv2dLayer(numOutputChannels, kH, kW, prevLayerOutput, m_device);
		m_layers.push_back(layer);
		if (weigthsFile) layer->readWeights(weigthsFile);
		if (biasesFile) layer->readBiases(biasesFile);
		m_layerNames.push_back("conv_" + std::to_string(numConv++));
		return layer;
	}

	MaxPooling2dLayer* addMaxPooling2dLayer(int kH, int kW) {
		Tensor* prevLayerOutput = m_layers.back()->getOutputTensor();
		m_layers.push_back(new MaxPooling2dLayer(kH, kW, prevLayerOutput, m_device));
		m_layerNames.push_back("max_pool_" + std::to_string(numMaxPool++));
		return (MaxPooling2dLayer*)m_layers.back();
	}

	FCLayer* addFCLayer(int numNeurons, bool applyRelu, const char* weigthsFile = nullptr, const char* biasesFile = nullptr) {
		Tensor* prevLayerOutput = m_layers.back()->getOutputTensor();
		FCLayer* layer = new FCLayer(numNeurons, prevLayerOutput, m_device, applyRelu);
		m_layers.push_back(layer);
		if (weigthsFile) layer->readWeights(weigthsFile);
		if (biasesFile) layer->readBiases(biasesFile);
		m_layerNames.push_back("fc_" + std::to_string(numFC++));
		return layer;
	}

	ReLULayer* addReLULayer() {
		Tensor* prevLayerOutput = m_layers.back()->getOutputTensor();
		ReLULayer* layer = new ReLULayer(prevLayerOutput, m_device);
		m_layers.push_back(layer);
		m_layerNames.push_back("relu_" + std::to_string(numRELU++));
		return layer;
	}

	float* getOutputData() {
		Tensor* outputTensor = m_layers.back()->getOutputTensor();
		return m_device == DEVICE_CPU ? outputTensor->getData() : outputTensor->getDataFromDeviceData();
	}

	void run() {
		// skip the first layer which is the input layer
		for (int i = 1; i < m_layers.size(); i++) {
#ifndef NDEBUG
			clock_t start = clock();
#endif
			m_layers[i]->forward();
#ifndef NDEBUG
			std::cout << m_layerNames[i] << ": " << (float)(clock() - start) / CLOCKS_PER_SEC << std::endl;
#endif
		}
	}

	~Model() {
		for (auto& l : m_layers) {
			delete l;
		}
	}

private:
	std::vector<Layer*> m_layers;
	EnumDevice m_device;

	int numConv, numFC, numRELU, numMaxPool;
	std::vector<std::string> m_layerNames;
};