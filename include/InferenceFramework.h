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
#include <set>
#include "vulkan_operator_upSampling_2D.h"
#include "vulkan_operator_concatenate.h"
#include <algorithm>
#include <iostream>

using Shape = std::vector<int>;

int getShapeSize(const Shape& shape);

std::ostream& operator<<(std::ostream& out, const Shape& shpe);

enum EnumDevice
{
	DEVICE_UNSPECIFIED = 0,
	DEVICE_CPU = 1,
	DEVICE_VULKAN = 2,
	DEVICE_ALL = DEVICE_CPU | DEVICE_VULKAN
};


void readBinFile(const char* fileName, float* out, int count);
void readBinFile(FILE* file, float* out, int count);

class Layer;

class Tensor
{
public:
	Tensor(const Shape& shape, EnumDevice device = EnumDevice::DEVICE_UNSPECIFIED, const char* binFile = nullptr)
		: m_shape(shape),
		m_device(device),
		m_size(getShapeSize(shape)),
		m_data(nullptr),
		m_deviceData(nullptr)
	{
		
		setDevice(device);

		if (binFile) {
			readData(binFile);
		}
		createShapeSuffixProduct();
	}

	void createShapeSuffixProduct()
	{
		m_shapeSuffixProduct.resize(m_shape.size() + 1);
		m_shapeSuffixProduct.back() = 1;
		for (int i = m_shapeSuffixProduct.size() - 2; i >= 0; i--) {
			m_shapeSuffixProduct[i] = m_shapeSuffixProduct[i + 1] * m_shape[i];
		}
	}

	void setDevice(EnumDevice device) {

		EnumDevice prevDevice = m_device;
		m_device = device;
		if (m_device & DEVICE_CPU)
		{
			if (m_data == nullptr) {
				m_data = new float[m_size];
			}
			if (prevDevice == DEVICE_VULKAN) {
				m_deviceData->toHost(m_data);
			}
		}

		if (m_device & DEVICE_VULKAN)
		{
			if (m_deviceData == nullptr) {
				m_deviceData = new vuh::Array<float>(InstanceManger::getInstance().getDefaultDevice(), m_size);
			}
			if (prevDevice == DEVICE_CPU) {
				m_deviceData->fromHost(m_data, m_data + m_size);
			}
		}

	}

	EnumDevice getDevice() { return m_device; }

	Shape begin() { return std::vector<int>(m_shape.size(), 0); }

	bool inc(Shape& it)
	{
		for (int i = it.size() - 1; i >= 0; i--) {
			if (it[i] + 1 < m_shape[i]) {
				it[i]++; 
				return true;
			}
			it[i] = 0;
		}
		return false;
	}

	void readData(const char* binFile) {
		assert(m_device != DEVICE_UNSPECIFIED);
		if (!m_data)
			m_data = new float[m_size];
		readBinFile(binFile, m_data, m_size);
		if (m_device & DEVICE_VULKAN) {
			m_deviceData->fromHost(m_data, m_data + m_size);
		}	
	}

	void readData(FILE* file) {
		assert(m_device != DEVICE_UNSPECIFIED);
		if (!m_data)
			m_data = new float[m_size];
		readBinFile(file, m_data, m_size);
		if (m_device & DEVICE_VULKAN) {
			m_deviceData->fromHost(m_data, m_data + m_size);
		}
	}

	void setData(const std::vector<float>& data)
	{
		assert(m_device != DEVICE_UNSPECIFIED);
		assert(data.size() == m_size);
		if (m_device & DEVICE_CPU) {
			for (int i = 0; i < m_size; i++) {
				m_data[i] = data[i];
			}
		}	
		if (m_device & DEVICE_VULKAN) {
			m_deviceData->fromHost(&data[0], &data[0] + m_size);
		}
	}

	float* getData() {
		assert(m_device != DEVICE_UNSPECIFIED);
		if (m_device & DEVICE_CPU)
			return m_data;
		else if (m_device & DEVICE_VULKAN)
			return getDataFromDeviceData();
		else assert(false); // unknown device
	}
	vuh::Array<float>* getDeviceData() { return m_deviceData; }
	float* getDataFromDeviceData() {
		if (!m_data)
			m_data = new float[m_size];
		m_deviceData->toHost(m_data);
		return m_data;
	}
	int getSize() { return m_size; }
	Shape getShape() { return m_shape; }

	float& at(const Shape& loc)
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

	void reshape(Shape newShape)
	{
		bool hasMinusOne = false;
		int minusOneIdx = 0;
		int prod = 1;
		for (int i = 0; i < newShape.size(); i++) {
			int x = newShape[i];
			if (x == -1) {
				if (hasMinusOne) {
					assert(false); // can't have many unknown dims
				}
				hasMinusOne = true;
				minusOneIdx = i;
			}
			else {
				prod *= x;
			}
		}
		if (hasMinusOne) {
			assert(m_size % prod == 0);
			int unknownDim = m_size / prod;
			newShape[minusOneIdx] = unknownDim;
		} 
		else {
			assert(prod == m_size);
		}
		m_shape = newShape;
		createShapeSuffixProduct();
	}

	Layer* getParentLayer() { return m_parentLayer; }

	~Tensor() {
		delete[] m_data;
		delete m_deviceData;
	}

	friend class Layer;
private:
	int m_size;
	Shape m_shape;
	std::vector<int> m_shapeSuffixProduct;
	float* m_data;
	vuh::Array<float>* m_deviceData;
	EnumDevice m_device;

	Layer* m_parentLayer; // the layer that owns this tensor or the last last layer that modifies this tensor
};

class Layer
{
public:
	Layer(EnumDevice _device = DEVICE_UNSPECIFIED)
		: m_device(_device), m_isInputLinked(false) {}
	virtual void forward() = 0;
	virtual Tensor* getOutputTensor() = 0;
	std::vector<Layer*>& getOutgoingLayers() { return m_outgoingLayers; }
	std::vector<Layer*>& getInputLayers() { return m_inputLayers; }

	virtual void readWeigthsFromFile(FILE* file) {};

	void setDevice(EnumDevice device) { 
		m_device = device; 
		for (Tensor* t : m_ownedTensors) {
			t->setDevice(m_device);
		}
	}

	static void addConnection(Layer* from, Layer* to)
	{
		from->m_outgoingLayers.push_back(to);
		to->m_inputLayers.push_back(from);
	}

	// link input(s) to layer and returns output layer
	Tensor* of(const std::vector<Tensor*>& inputs)
	{
		for (Tensor* t : inputs) {
			addConnection(t->m_parentLayer, this);
		}

		init(inputs);
		m_isInputLinked = true;

		getOutputTensor()->m_parentLayer = this;
		return getOutputTensor();
	}

	virtual int getParamCount() = 0;

	Tensor* operator()(Tensor* input) {
		return of({ input });
	}

	Tensor* operator()(const std::vector<Tensor*>& inputs) {
		return of(inputs);
	}

	virtual void showInfo()
	{
		std::cout << typeid(*this).name() << "		output_shape: " << getOutputTensor()->getShape() << "	 params: " << getParamCount() << '\n';
	}

	virtual ~Layer() 
	{
		for (Tensor* t : m_ownedTensors) {
			delete t;
		}
	}
protected:
	virtual void init(const std::vector<Tensor*>& inputs) = 0;
	Tensor* createOwnedTensor(const std::vector<int>& shape)
	{
		Tensor* t = new Tensor(shape, m_device);
		m_ownedTensors.push_back(t);
		t->m_parentLayer = this;
		return t;
	}

protected:
	std::vector<Layer*> m_inputLayers;
	std::vector<Layer*>	m_outgoingLayers;
	std::vector<Tensor*> m_ownedTensors;

	EnumDevice m_device;
	bool m_isInputLinked;
};

class Conv2dLayer : public Layer
{
public:
	Conv2dLayer(int _filters, int _size, int _stride, const std::string& _padding, bool _useBias)
		: filters(_filters), size(_size), stride(_stride), useBias(_useBias)
	{
		assert(_padding == "same" || _padding == "valid");
		padding = _padding == "valid" ? 0 : size - 1;
	}

	virtual void init(const std::vector<Tensor*>& inputs) override
	{	
		assert(inputs.size() == 1);
		inputImage = inputs[0];

		assert(inputImage->getShape().size() == 3);
		h = inputImage->getShape()[0];
		w = inputImage->getShape()[1];
		c = inputImage->getShape()[2];

		
		out_h = (h + padding - size) / stride + 1;
		out_w = (w + padding - size) / stride + 1;

		outputImage = createOwnedTensor({ out_h, out_w, filters });
		weigths = createOwnedTensor({ size, size, c, filters });
		biases = createOwnedTensor({ filters });
	}

	virtual int getParamCount() override
	{
		assert(m_isInputLinked);
		return weigths->getSize() + (useBias ? biases->getSize() : 0);
	}

	virtual void readWeigthsFromFile(FILE* file) override
	{
		readWeights(file);
		readBiases(file);
	}

	void readWeights(const char* binFile) {
		weigths->readData(binFile);
	}

	void readWeights(FILE* file) {
		weigths->readData(file);
	}

	void readWeightsAndTranspose(FILE* file) {
		Tensor* tmp = new Tensor({filters, c, size, size}, DEVICE_CPU);
		tmp->readData(file);

		EnumDevice oldDevice = weigths->getDevice();
		weigths->setDevice(DEVICE_CPU);
		
		for (int i = 0; i < size; i++) {
			for (int j = 0; j < size; j++) {
				for (int k = 0; k < c; k++) {
					for (int l = 0; l < filters; l++) {
						weigths->at({ i,j,k,l }) = tmp->at({ l, k, i, j }); // from darknet format to tf format
					}
				}
			}
		}

		weigths->setDevice(oldDevice);
		delete tmp;
	}

	void readBiases(const char* binFile) {
		if (useBias)
			biases->readData(binFile);
	}

	void readBiases(FILE* file) {
		if (useBias)
			biases->readData(file);
	}

	bool getUseBias() { return useBias;  }

	virtual void forward() {
		assert(m_isInputLinked);
		assert(m_device != DEVICE_UNSPECIFIED);
		if (m_device == DEVICE_CPU) {
			operator_conv2d_cpu(inputImage->getData(), weigths->getData(), biases->getData(), h, w, c, filters, size, stride, padding, out_h, out_w, useBias, outputImage->getData());
		}
		else if (m_device == DEVICE_VULKAN) {
			vulkan_operator_conv2d(inputImage->getDeviceData(), weigths->getDeviceData(), biases->getDeviceData(), h, w, c, filters, size, stride, padding, out_h, out_w, useBias, outputImage->getDeviceData());
			outputImage->setDevice(DEVICE_CPU);
			outputImage->setDevice(DEVICE_VULKAN);
		}
	}

	Tensor* getOutputTensor() { return outputImage; }

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
};

class MaxPooling2dLayer : public Layer
{
public:
	MaxPooling2dLayer(int _size, int _stride, std::string _padding)
		: size(_size), stride(_stride)
	{
		assert(_padding == "same" || _padding == "valid");
		padding = _padding == "valid" ? 0 : size - 1;
	}

	virtual void init(const std::vector<Tensor*>& inputs) override
	{
		assert(inputs.size() == 1);
		m_inputImage = inputs[0];

		assert(m_inputImage->getShape().size() == 3);
		h = m_inputImage->getShape()[0];
		w = m_inputImage->getShape()[1];
		c = m_inputImage->getShape()[2];

		out_h = (h + padding - size) / stride + 1;
		out_w = (w + padding - size) / stride + 1;

		m_outputImage = createOwnedTensor({ out_h, out_w, c });
	}

	virtual int getParamCount() override
	{
		return 0;
	}

	virtual void forward() {
		assert(m_device != DEVICE_UNSPECIFIED);
		if (m_device == DEVICE_CPU) {
			operator_maxpool_cpu(m_inputImage->getData(), h, w, c, size, stride, padding, out_h, out_w, m_outputImage->getData());
		}
		else if (m_device == DEVICE_VULKAN) {
			vulkan_operator_maxpool(m_inputImage->getDeviceData(), h, w, c, size, stride, padding, out_h, out_w, m_outputImage->getDeviceData());
		}
	}

	Tensor* getOutputTensor() { return m_outputImage; }

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
};

class FCLayer : public Layer
{
public:
	FCLayer(int numNeurons, bool applyRelu = false)
		: m_numNeurons(numNeurons), m_applyRelu(applyRelu)
	{
	}

	virtual void init(const std::vector<Tensor*>& inputs) override
	{
		assert(inputs.size() == 1);
		m_input = inputs[0];
		m_weights = createOwnedTensor({ m_input->getSize(), m_numNeurons });
		m_biases = createOwnedTensor({ m_numNeurons });
		m_output = createOwnedTensor({ m_numNeurons });
	}

	virtual int getParamCount() override
	{
		assert(m_isInputLinked);
		return m_weights->getSize() + m_biases->getSize();
	}

	virtual void readWeigthsFromFile(FILE* file) override
	{
		readWeights(file);
		readBiases(file);
	}

	void readWeights(const char* binFile) {
		m_weights->readData(binFile);
	}

	void readBiases(const char* binFile) {
		m_biases->readData(binFile);
	}

	void readWeights(FILE* file) {
		m_weights->readData(file);
	}

	void readBiases(FILE* file) {
		m_biases->readData(file);
	}

	virtual void forward() {
		assert(m_device != DEVICE_UNSPECIFIED);
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

private:

	int m_numNeurons;
	Tensor* m_input;
	Tensor* m_weights;
	Tensor* m_biases;
	Tensor* m_output;
	bool m_applyRelu;
};

class InputLayer : public Layer
{
public:
	InputLayer(const Shape& shape, const char* binFile = nullptr)
		: m_shape(shape)
	{
		m_data = createOwnedTensor(shape);
		if (binFile)
			m_data->readData(binFile);
	}

	virtual int getParamCount() override
	{
		return 0;
	}

	virtual void init(const std::vector<Tensor*>& inputs) override {
		assert(false); // input layer does not have other inputs
	}

	void fill(const std::vector<float>& src) {
		m_data->setData(src);
	}

	virtual void forward() {}
	Tensor* getOutputTensor() { return m_data; }

private:
	Shape m_shape;
	Tensor* m_data;
};

class ReLULayer : public Layer
{
public:
	ReLULayer() {}

	virtual void init(const std::vector<Tensor*>& inputs) override
	{
		assert(inputs.size() == 1);
		m_input = inputs[0];
	}

	virtual int getParamCount() override
	{
		return 0;
	}

	virtual void forward() {
		assert(m_device != DEVICE_UNSPECIFIED);
		if (m_device == EnumDevice::DEVICE_CPU) {
			for (int i = 0; i < m_input->getSize(); i++) {
				m_input->getData()[i] = (std::max)(m_input->getData()[i], 0.f);
			}
		}
		else if (m_device == EnumDevice::DEVICE_VULKAN) {
			vulkan_operator_relu(m_input->getDeviceData(), m_input->getDeviceData(), m_input->getSize());
		}
	}
	Tensor* getOutputTensor() { return m_input; }

private:
	Tensor* m_input;
};


class LeakyReLULayer : public Layer
{
public:
	LeakyReLULayer() {}

	virtual void init(const std::vector<Tensor*>& inputs) override
	{
		assert(inputs.size() == 1);
		m_input = inputs[0];
	}

	virtual int getParamCount() override
	{
		return 0;
	}

	virtual void forward() {
		assert(m_device != DEVICE_UNSPECIFIED);
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
};


class UpSampling2DLayer : public Layer
{
public:
	UpSampling2DLayer(int size) : size(size) {} // nearest interpolation

	virtual void init(const std::vector<Tensor*>& inputs) override
	{
		assert(inputs.size() == 1);
		m_input = inputs[0];

		assert(m_input->getShape().size() == 3);
		h = m_input->getShape()[0];
		w = m_input->getShape()[1];
		c = m_input->getShape()[2];
		out_h = h * size;
		out_w = w * size;
		m_output = createOwnedTensor({ out_h, out_w, c });
	}

	virtual int getParamCount() override
	{
		return 0;
	}

	virtual void forward() {
		assert(m_device != DEVICE_UNSPECIFIED);
		if (m_device == EnumDevice::DEVICE_CPU) {
			for (int i = 0; i < out_h; i++) {
				for (int j = 0; j < out_w; j++) {
					for (int k = 0; k < c; k++) {
						m_output->at({ i,j,k }) = m_input->at({ i / size, j / size, k });
					}
				}
			}
		}
		else if (m_device == EnumDevice::DEVICE_VULKAN) {
			vulkan_operator_upSampling_2D(m_input->getDeviceData(), m_output->getDeviceData(), h, w, c, size);
		}
	}
	Tensor* getOutputTensor() { return m_output; }

private:
	int size, h, w, c, out_h, out_w;
	Tensor* m_input;
	Tensor* m_output;
};

class ConcatenateLayer : public Layer
{
public:
	ConcatenateLayer() {} // concatenation along the last (channels) axis

	virtual void init(const std::vector<Tensor*>& inputs) override
	{
		assert(inputs.size() == 2);
		m_input1 = inputs[0];
		m_input2 = inputs[1];

		int h1 = m_input1->getShape()[0];
		int w1 = m_input1->getShape()[1];
		c1 = m_input1->getShape()[2];

		int h2 = m_input2->getShape()[0];
		int w2 = m_input2->getShape()[1];
		c2 = m_input2->getShape()[2];

		assert(h1 == h2);
		assert(w1 == w2);
		h = h1; w = w1;
		m_output = createOwnedTensor({ h, w, c1 + c2 });
	}

	virtual int getParamCount() override
	{
		return 0;
	}

	virtual void forward() {
		assert(m_device != DEVICE_UNSPECIFIED);
		if (m_device == EnumDevice::DEVICE_CPU) {
			for (int i = 0; i < h; i++) {
				for (int j = 0; j < w; j++) {
					for (int k = 0; k < c1 + c2; k++) {
						m_output->at({ i,j,k }) = k < c1 ? m_input1->at({ i, j, k }) : m_input2->at({i, j, k - c1});
					}
				}
			}
		}
		else if (m_device == EnumDevice::DEVICE_VULKAN) {
			vulkan_operator_concatenate(m_input1->getDeviceData(), m_input2->getDeviceData(), m_output->getDeviceData(), h, w, c1, c2);
		}
	}
	Tensor* getOutputTensor() { return m_output; }

private:
	int h, w, c1, c2;
	Tensor* m_input1, *m_input2;
	Tensor* m_output;
};

class BatchNormalizationLayer : public Layer
{
public:
	BatchNormalizationLayer()
		: eps(0.001)
	{
	}

	virtual void init(const std::vector<Tensor*>& inputs) override
	{
		assert(inputs.size() == 1);
		m_input = inputs[0];

		int filters = m_input->getShape().back(); // assuming data_format == channels_last
		m_params = createOwnedTensor({ 4, filters });
	}

	virtual int getParamCount() override
	{
		assert(m_isInputLinked);
		return m_params->getSize();
	}

	virtual void forward() {
		assert(m_device != DEVICE_UNSPECIFIED);
		if (m_device == EnumDevice::DEVICE_CPU) {
			float* data = m_input->getData();
			float* params = m_params->getData();
			int filters = m_input->getShape().back();
			for (int i = 0; i < m_input->getSize(); i++) {
				int channel = i % filters;
				float beta = m_params->at({ 0, channel });
				float gamma = m_params->at({ 1, channel });
				float mean = m_params->at({ 2, channel });
				float variance = m_params->at({ 3, channel });
				assert(variance >= 0);
				data[i] = (data[i] - mean) / sqrt(variance + eps);
				data[i] = data[i] * gamma + beta;
				assert(!std::isnan(data[i]));
			}
		}
		else if (m_device == EnumDevice::DEVICE_VULKAN) {
			vulkan_operator_batch_normalization(m_input->getDeviceData(), m_input->getSize(), m_input->getShape().back(), m_params->getDeviceData());
		}
	}
	Tensor* getOutputTensor() { return m_input; }

	virtual void readWeigthsFromFile(FILE* file) override
	{
		readParams(file);
	}

	void readParams(const char* filename)
	{
		m_params->readData(filename);
		//validate
		for (int i = 0; i < m_input->getShape().back(); i++) {
			float variance = m_params->at({ 3, i });
			assert(variance >= 0);
		}
	}
	
	void readParams(FILE* file)
	{
		m_params->readData(file);
		//validate
		for (int i = 0; i < m_input->getShape().back(); i++) {
			float variance = m_params->at({ 3, i });
			assert(variance >= 0);
		}
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

private:
	Tensor* m_input;

	Tensor* m_params; // order: beta, gamma, mean, variance (same as darknet)
	const float eps;
};

class SequentialModel
{
public:
	SequentialModel(EnumDevice device)
	 : m_device(device),
		numConv(0), numFC(0), numRELU(0), numMaxPool(0), numLeakyRELU(0), numBatchNorm(0), numUpSampling2D(0), numConcatenate(0)
	{
	}

	InputLayer* addInputLayer(const Shape& shape, const char* binFile = nullptr) {
		InputLayer* inputLayer = new InputLayer(shape, binFile);
		inputLayer->setDevice(m_device);
		m_layers.push_back(inputLayer);
		m_layerNames.push_back("input");
		return (InputLayer*)m_layers.back();
	}

	Conv2dLayer* addConv2dLayer(int filters, int size, int stride, const std::string& padding = "valid", bool useBias = true, const char* weigthsFile = nullptr, const char* biasesFile = nullptr) {
		Tensor* prevLayerOutput = m_layers.back()->getOutputTensor();
		Conv2dLayer* layer = new Conv2dLayer(filters, size, stride, padding, useBias);
		layer->setDevice(m_device);
		layer->of({ prevLayerOutput }); // link layer input
		m_layers.push_back(layer);
		if (weigthsFile) layer->readWeights(weigthsFile);
		if (biasesFile) layer->readBiases(biasesFile);
		m_layerNames.push_back("conv_" + std::to_string(numConv++));
		return layer;
	}

	MaxPooling2dLayer* addMaxPooling2dLayer(int size, int stride, const std::string& padding = "valid") {
		Tensor* prevLayerOutput = m_layers.back()->getOutputTensor();
		MaxPooling2dLayer* layer = new MaxPooling2dLayer(size, stride, padding);
		layer->setDevice(m_device);
		layer->of({ prevLayerOutput }); // link layer input
		m_layers.push_back(layer);
		m_layerNames.push_back("max_pool_" + std::to_string(numMaxPool++));
		return (MaxPooling2dLayer*)m_layers.back();
	}

	FCLayer* addFCLayer(int numNeurons, bool applyRelu, const char* weigthsFile = nullptr, const char* biasesFile = nullptr) {
		Tensor* prevLayerOutput = m_layers.back()->getOutputTensor();
		FCLayer* layer = new FCLayer(numNeurons, applyRelu);
		layer->setDevice(m_device);
		layer->of({ prevLayerOutput }); // link layer input
		m_layers.push_back(layer);
		if (weigthsFile) layer->readWeights(weigthsFile);
		if (biasesFile) layer->readBiases(biasesFile);
		m_layerNames.push_back("fc_" + std::to_string(numFC++));
		return layer;
	}

	ReLULayer* addReLULayer() {
		Tensor* prevLayerOutput = m_layers.back()->getOutputTensor();
		ReLULayer* layer = new ReLULayer();
		layer->setDevice(m_device);
		layer->of({ prevLayerOutput }); // link layer input
		m_layers.push_back(layer);
		m_layerNames.push_back("relu_" + std::to_string(numRELU++));
		return layer;
	}
	
	LeakyReLULayer* addLeakyReLULayer() {
		Tensor* prevLayerOutput = m_layers.back()->getOutputTensor();
		LeakyReLULayer* layer = new LeakyReLULayer();
		layer->setDevice(m_device);
		layer->of({ prevLayerOutput }); // link layer input
		m_layers.push_back(layer);
		m_layerNames.push_back("leaky_relu_" + std::to_string(numLeakyRELU++));
		return layer;
	}
	
	UpSampling2DLayer* addUpSampling2DLayer(int size) {
		Tensor* prevLayerOutput = m_layers.back()->getOutputTensor();
		UpSampling2DLayer* layer = new UpSampling2DLayer(size);
		layer->setDevice(m_device);
		layer->of({ prevLayerOutput }); // link layer input
		m_layers.push_back(layer);
		m_layerNames.push_back("UpSampling2DLayer_" + std::to_string(numLeakyRELU++));
		return layer;
	}
	ConcatenateLayer* addConcatenateLayer(int size) {
		Tensor* prevLayerOutput = m_layers.back()->getOutputTensor();
		ConcatenateLayer* layer = new ConcatenateLayer();
		layer->setDevice(m_device);
		layer->of({ prevLayerOutput }); // link layer input
		m_layers.push_back(layer);
		m_layerNames.push_back("ConcatenateLayer_" + std::to_string(numConcatenate++));
		return layer;
	}

	BatchNormalizationLayer* addBatchNormalizationLayer(const char* paramsFilename = nullptr) {
		Tensor* prevLayerOutput = m_layers.back()->getOutputTensor();
		BatchNormalizationLayer* layer = new BatchNormalizationLayer();
		layer->setDevice(m_device);
		layer->of({ prevLayerOutput }); // link layer input
		if (paramsFilename) {
			layer->readParams(paramsFilename);
		}
		m_layers.push_back(layer);
		m_layerNames.push_back("batch_norm_" + std::to_string(numBatchNorm++));
		return layer;
	}

	float* getOutputData() {
		Tensor* outputTensor = m_layers.back()->getOutputTensor();
		return outputTensor->getData();
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

	~SequentialModel() {
		for (auto& l : m_layers) {
			delete l;
		}
	}

private:
	std::vector<Layer*> m_layers;
	EnumDevice m_device;

	int numConv, numFC, numRELU, numMaxPool, numLeakyRELU, numBatchNorm, numUpSampling2D, numConcatenate;
	std::vector<std::string> m_layerNames;
};


class Model
{
public:
	Model(Tensor* input, Tensor* output, EnumDevice device = DEVICE_CPU)
		: m_device(device),
		m_inputs({ input }), m_output(output)
	{
		topologicalSort();
		setDevice();

		showInfo();
	}	
	
	Model(const std::vector<Tensor*>& inputs, Tensor* output, EnumDevice device = DEVICE_CPU)
		: m_device(device),
		m_inputs(inputs), m_output(output)
	{
		topologicalSort();
		setDevice();

		showInfo();
	}

	Tensor* run(const std::vector<std::vector<float>>& inputs)
	{
		setDevice();
		assert(inputs.size() == m_inputs.size());
		for (int i = 0; i < inputs.size(); i++) {
			m_inputs[i]->setData(inputs[i]);
		}
		for (Layer* l : sortedLayers) {
			l->forward();

			//// copy data back to cpu if vulkan is used
			//l->setDevice(DEVICE_CPU);
			//l->setDevice(m_device);
		}
		return m_output;
	}

	Tensor* run(const std::vector<float>& input)
	{
		setDevice();
		assert(m_inputs.size() == 1);
		m_inputs[0]->setData(input);
		int i = 0;
		for (Layer* l : sortedLayers) {
			l->forward();

			//// copy data back to cpu if vulkan is used
			//l->setDevice(DEVICE_CPU);
			//l->setDevice(m_device);
			//l->getOutputTensor()->setDevice(DEVICE_CPU);
			//l->getOutputTensor()->setDevice(DEVICE_VULKAN);
			assert(!std::isnan(l->getOutputTensor()->getData()[0]));
			i++;
		}
		return m_output;
	}

	std::vector<Layer*>& layers() { return sortedLayers; }

	void readWeights(const char* filename, int ignoreFirst = 0)
	{
		FILE* file = fopen(filename, "rb");

		if (ignoreFirst) {
			float* tmp = new float[ignoreFirst];
			readBinFile(file, tmp, ignoreFirst);
			delete[] tmp;
		}

		for (Layer* l : sortedLayers)
		{
			l->readWeigthsFromFile(file);
		}

		fclose(file);
	}

	void readDarknetWeights(const char* filename, int ignoreFirst = 0)
	{
		FILE* file = fopen(filename, "rb");

		if (ignoreFirst) {
			float* tmp = new float[ignoreFirst];
			readBinFile(file, tmp, ignoreFirst);
			delete[] tmp;
		}

		for (Layer* l : sortedLayers)
		{
			Conv2dLayer* conv = dynamic_cast<Conv2dLayer*>(l);
			if (conv == nullptr)
				continue;

			if (conv->getUseBias()) {
				conv->readBiases(file);
			} 
			else {
				Layer * nxt = conv->getOutgoingLayers().front();
				BatchNormalizationLayer* bn = dynamic_cast<BatchNormalizationLayer*>(nxt);
				assert(bn);
				bn->readParams(file);
			}
			conv->readWeightsAndTranspose(file);
		}

		// check that we've reached the end of the file
		float tmp;
		int read = fread(&tmp, sizeof(float), 1, file);
		assert(read == 0);

		fclose(file);
	}

	int getParamCount()
	{
		int res = 0;
		for (Layer* l : sortedLayers) {
			res += l->getParamCount();
		}
		return res;
	}
	
	void showInfo()
	{
		for (Layer* l : sortedLayers)
		{
			l->showInfo();
		}
	}

private:
	void topologicalSort()
	{
		sortedLayers.clear();
		std::set<Layer*> visited;

		for (Tensor* t : m_inputs) {
			dfs(t->getParentLayer(), visited);
		}

		reverse(sortedLayers.begin(), sortedLayers.end());

		assert(visited.find(m_output->getParentLayer()) != visited.end());
	}

	void setDevice()
	{
		for (Layer* l : sortedLayers) {
			l->setDevice(m_device);
		}
	}

	void dfs(Layer* l, std::set<Layer*>& visited)
	{
		if (visited.find(l) != visited.end())
			return;

		visited.insert(l);

		for (Layer* n : l->getOutgoingLayers()) {
			dfs(n, visited);
		}
		sortedLayers.push_back(l);
	}

private:
	std::vector<Tensor*> m_inputs;
	Tensor* m_output;
	std::vector<Layer*> sortedLayers;

	EnumDevice m_device;
};
