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
#include "vulkan_operator_leaky_relu.h"
#include "vulkan_operator_batch_normalization.h"
#include <time.h>

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
		m_shapeSuffixProduct.resize(m_shape.size() + 1);
		m_shapeSuffixProduct.back() = 1;
		for (int i = m_shapeSuffixProduct.size() - 2; i >= 0; i--) {
			m_shapeSuffixProduct[i] = m_shapeSuffixProduct[i + 1] * m_shape[i];
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

	float at(const Shape& loc)
	{
		assert(m_data);
		assert(loc.size() == m_shape.size());
		int idx = 0;
		for (int i = 0; i < loc.size(); i++) {
			idx += loc[i] * m_shapeSuffixProduct[i + 1];
		}
		assert(idx < m_size);
		return m_data[idx];
	}

	~Tensor() {
		delete[] m_data;
		delete m_deviceData;
	}

private:
	int m_size;
	Shape m_shape;
	std::vector<int> m_shapeSuffixProduct;
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
	Conv2dLayer(int _filters, int _size, int _stride, const std::string& _padding, bool _useBias, Tensor* _inputImage, EnumDevice _device)
		: filters(_filters), size(_size), stride(_stride), useBias(_useBias), inputImage(_inputImage),	device(_device)
	{
		assert(inputImage->getShape().size() == 3); 
		h = inputImage->getShape()[0];
		w = inputImage->getShape()[1];
		c = inputImage->getShape()[2];

		assert(_padding == "same" || _padding == "valid");
		padding = _padding == "valid" ? 0 : size - 1;

		out_h = (h + padding - size) / stride + 1;
		out_w = (w + padding - size) / stride + 1;

		outputImage = new Tensor({ out_h, out_w, filters }, device);
		weigths = new Tensor({ size, size, c, filters }, device);
		if (useBias)
			biases = new Tensor({ filters }, device);
	}

	void readWeights(const char* binFile) {
		weigths->readData(binFile);
	}

	void readBiases(const char* binFile) {
		if (useBias)
			biases->readData(binFile);
	}

	virtual void forward() {
		if (device == DEVICE_CPU) {
			operator_conv2d_cpu(inputImage->getData(), weigths->getData(), biases->getData(), h, w, c, filters, size, stride, padding, out_h, out_w, useBias, outputImage->getData());
		}
		else if (device == DEVICE_VULKAN) {
			vulkan_operator_conv2d(inputImage->getDeviceData(), weigths->getDeviceData(), biases->getDeviceData(), h, w, c, filters, size, stride, padding, out_h, out_w, useBias, outputImage->getDeviceData());
		}
	}

	Tensor* getOutputTensor() { return outputImage; }

	~Conv2dLayer() {
		delete outputImage;
		delete weigths;
		if (useBias)
			delete biases;
	}

private:
	Tensor* inputImage;
	Tensor* outputImage;
	Tensor* weigths;
	Tensor* biases;

	int h;
	int w;
	int c;
	int filters;
	int size;
	int stride;
	int padding;
	bool useBias;
	int out_h;
	int out_w;
	EnumDevice device;
};

class MaxPooling2dLayer : public Layer
{
public:
	MaxPooling2dLayer(int _size, int _stride, std::string _padding, Tensor* inputImage, EnumDevice device)
		: size(_size), stride(_stride), m_inputImage(inputImage), m_device(device)
	{
		assert(m_inputImage->getShape().size() == 3);
		h = m_inputImage->getShape()[0];
		w = m_inputImage->getShape()[1];
		c = m_inputImage->getShape()[2];

		assert(_padding == "same" || _padding == "valid");
		padding = _padding == "valid" ? 0 : size - 1;

		out_h = (h + padding - size) / stride + 1;
		out_w = (w + padding - size) / stride + 1;

		m_outputImage = new Tensor({ out_h, out_w, c }, m_device);
	}

	virtual void forward() {
		if (m_device == DEVICE_CPU) {
			operator_maxpool_cpu(m_inputImage->getData(), h, w, c, size, stride, padding, out_h, out_w, m_outputImage->getData());
		}
		else if (m_device == DEVICE_VULKAN) {
			vulkan_operator_maxpool(m_inputImage->getDeviceData(), h, w, c, size, stride, padding, out_h, out_w, m_outputImage->getDeviceData());
		}
	}

	Tensor* getOutputTensor() { return m_outputImage; }

	~MaxPooling2dLayer() {
		delete m_outputImage;
	}

private:
	Tensor* m_inputImage;
	Tensor* m_outputImage;
	// input
	int h;
	int w;
	int c;
	// hyper parameters
	int size;
	int stride;
	int padding;
	// output
	int out_h;
	int out_w;
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


class LeakyReLULayer : public Layer
{
public:
	LeakyReLULayer(Tensor* input, EnumDevice device)
		: m_input(input), m_device(device) {}

	virtual void forward() {
		if (m_device == EnumDevice::DEVICE_CPU) {
			for (int i = 0; i < m_input->getSize(); i++) {
				m_input->getData()[i] = (m_input->getData()[i] < 0 ? 0.1 : 1) * m_input->getData()[i];
			}
		}
		else if (m_device == EnumDevice::DEVICE_VULKAN) {
			vulkan_operator_leaky_relu(m_input->getDeviceData(), m_input->getDeviceData(), m_input->getSize());
		}
	}
	Tensor* getOutputTensor() { return m_input; }

private:
	Tensor* m_input;
	EnumDevice m_device;
};


class BatchNormalizationLayer : public Layer
{
public:
	BatchNormalizationLayer(Tensor* input, EnumDevice device)
		: m_input(input), m_device(device), eps(0.001) 
	{
		int filters = m_input->getShape().back(); // assuming data_format == channels_last
		m_params = new Tensor({ 4, filters }, m_device);
	}

	virtual void forward() {
		if (m_device == EnumDevice::DEVICE_CPU) {
			float* data = m_input->getData();
			float* params = m_params->getData();
			int filters = m_input->getShape().back();
			for (int i = 0; i < m_input->getSize(); i++) {
				int channel = i % filters;
				float gamma = m_params->at({ 0, channel });
				float beta = m_params->at({ 1, channel });
				float mean = m_params->at({ 2, channel });
				float variance = m_params->at({ 3, channel });
				data[i] = (data[i] - mean) / sqrt(variance + eps);
				data[i] = data[i] * gamma + beta;
			}
		}
		else if (m_device == EnumDevice::DEVICE_VULKAN) {
			vulkan_operator_batch_normalization(m_input->getDeviceData(), m_input->getSize(), m_input->getShape().back(), m_params->getDeviceData());
		}
	}
	Tensor* getOutputTensor() { return m_input; }

	void readParams(const char* filename)
	{
		m_params->readData(filename);
	}

	void setParams(const std::vector<float>& inParams)
	{
		assert(m_params->getSize() == inParams.size());

		if (m_device == EnumDevice::DEVICE_CPU) {
			for (int i = 0; i < m_params->getSize(); i++) {
				m_params->getData()[i] = inParams[i];
			}
		} 
		else if (m_device == EnumDevice::DEVICE_VULKAN) {
			m_params->getDeviceData()->fromHost(&inParams[0], &inParams[0] + inParams.size());
		}	
	}

	~BatchNormalizationLayer()
	{
		delete m_params;
	}

private:
	Tensor* m_input;
	EnumDevice m_device;

	Tensor* m_params; // order: gamma, beta, mean, variance (same as tf)
	const float eps;
};

class Model
{
public:
	Model(EnumDevice device)
	 : m_device(device),
		numConv(0), numFC(0), numRELU(0), numMaxPool(0), numLeakyRELU(0)
	{
	}

	InputLayer* addInputLayer(const Shape& shape, const char* binFile = nullptr) {
		m_layers.push_back(new InputLayer(shape, m_device, binFile));
		m_layerNames.push_back("input");
		return (InputLayer*)m_layers.back();
	}

	Conv2dLayer* addConv2dLayer(int filters, int size, int stride, const std::string& padding = "valid", bool useBias = true, const char* weigthsFile = nullptr, const char* biasesFile = nullptr) {
		Tensor* prevLayerOutput = m_layers.back()->getOutputTensor();
		Conv2dLayer* layer = new Conv2dLayer(filters, size, stride, padding, useBias, prevLayerOutput, m_device);
		m_layers.push_back(layer);
		if (weigthsFile) layer->readWeights(weigthsFile);
		if (biasesFile) layer->readBiases(biasesFile);
		m_layerNames.push_back("conv_" + std::to_string(numConv++));
		return layer;
	}

	MaxPooling2dLayer* addMaxPooling2dLayer(int size, int stride, const std::string& padding = "valid") {
		Tensor* prevLayerOutput = m_layers.back()->getOutputTensor();
		m_layers.push_back(new MaxPooling2dLayer(size, stride, padding, prevLayerOutput, m_device));
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
	
	LeakyReLULayer* addLeakyReLULayer() {
		Tensor* prevLayerOutput = m_layers.back()->getOutputTensor();
		LeakyReLULayer* layer = new LeakyReLULayer(prevLayerOutput, m_device);
		m_layers.push_back(layer);
		m_layerNames.push_back("leaky_relu_" + std::to_string(numLeakyRELU++));
		return layer;
	}

	BatchNormalizationLayer* addBatchNormalizationLayer(const char* paramsFilename = nullptr) {
		Tensor* prevLayerOutput = m_layers.back()->getOutputTensor();
		BatchNormalizationLayer* layer = new BatchNormalizationLayer(prevLayerOutput, m_device);
		if (paramsFilename) {
			layer->readParams(paramsFilename);
		}
		m_layers.push_back(layer);
		m_layerNames.push_back("batch_norm_" + std::to_string(numBatchNorm++));
		return layer;
	}

	float* getOutputData() {
		Tensor* outputTensor = m_layers.back()->getOutputTensor();
		return m_device == DEVICE_CPU ? outputTensor->getData() : outputTensor->getDataFromDeviceData();
	}

	int getOutputDataSize() {
		return m_layers.back()->getOutputTensor()->getSize();
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

	int numConv, numFC, numRELU, numMaxPool, numLeakyRELU, numBatchNorm;
	std::vector<std::string> m_layerNames;
};