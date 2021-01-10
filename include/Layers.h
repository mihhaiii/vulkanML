#pragma once
#include "Tensor.h"


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

	virtual void setBatchSize(int batch_size) {
		m_batch_size = batch_size;
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

	virtual void randomWeightInit(const float factor) {}
	virtual void backprop() {}

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
	int m_batch_size;
};

class Conv2dLayer : public Layer
{
public:
	Conv2dLayer(int _filters, int _size, int _stride, const std::string& _padding, bool _useBias);

	virtual void init(const std::vector<Tensor*>& inputs) override;
	virtual int getParamCount() override;
	virtual void readWeigthsFromFile(FILE* file) override;
	void readWeights(const char* binFile);
	void readWeights(FILE* file);
	void readWeightsAndTranspose(FILE* file);
	void readBiases(const char* binFile);
	void readBiases(FILE* file);
	bool getUseBias() { return useBias; }
	virtual void forward();
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
	MaxPooling2dLayer(int _size, int _stride, std::string _padding);

	virtual void init(const std::vector<Tensor*>& inputs) override;
	virtual int getParamCount() override;
	virtual void forward();
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
	FCLayer(int numNeurons, bool applyRelu = false);

	virtual void init(const std::vector<Tensor*>& inputs);
	virtual int getParamCount();
	virtual void readWeigthsFromFile(FILE* file);
	void readWeights(const char* binFile);
	void readBiases(const char* binFile);
	void readWeights(FILE* file);
	void readBiases(FILE* file);
	virtual void forward();
	virtual void randomWeightInit(const float factor);
	Tensor* getOutputTensor() { return m_output; }

private:

	int m_numNeurons;
	int m_numNeuronsLastLayer;
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
	{
		m_shape = shape;
		m_shape.insert(m_shape.begin(), -1); // first dim is the batch_size, unknown as for now
		m_data = createOwnedTensor(m_shape);
		if (binFile)
			m_data->readData(binFile);
	}

	virtual int getParamCount() override {
		return 0;
	}

	virtual void init(const std::vector<Tensor*>& inputs) override {
		assert(false); // input layer does not have other inputs
	}

	void fill(const std::vector<float>& src) {
		m_data->setData(&src[0], src.size());
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

	virtual void init(const std::vector<Tensor*>& inputs) override;

	virtual int getParamCount() override {
		return 0;
	}

	virtual void forward();
	Tensor* getOutputTensor() { return m_input; }

private:
	Tensor* m_input;
};


class LeakyReLULayer : public Layer
{
public:
	LeakyReLULayer() {}

	virtual void init(const std::vector<Tensor*>& inputs) override;

	virtual int getParamCount() override {
		return 0;
	}

	virtual void forward();
	Tensor* getOutputTensor() { return m_input; }

private:
	Tensor* m_input;
};


class UpSampling2DLayer : public Layer
{
public:
	UpSampling2DLayer(int size) : size(size) {} // nearest interpolation

	virtual void init(const std::vector<Tensor*>& inputs) override;

	virtual int getParamCount() override {
		return 0;
	}

	virtual void forward();
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

	virtual void init(const std::vector<Tensor*>& inputs) override;

	virtual int getParamCount() override {
		return 0;
	}

	virtual void forward();
	Tensor* getOutputTensor() { return m_output; }

private:
	int h, w, c1, c2;
	Tensor* m_input1, * m_input2;
	Tensor* m_output;
};

class BatchNormalizationLayer : public Layer
{
public:
	BatchNormalizationLayer()
		: eps(0.001) {}

	virtual void init(const std::vector<Tensor*>& inputs) override;

	virtual int getParamCount() override;

	virtual void forward();
	Tensor* getOutputTensor() { return m_input; }

	virtual void readWeigthsFromFile(FILE* file);
	void readParams(const char* filename);
	void readParams(FILE* file);
	void setParams(const std::vector<float>& inParams);

private:
	Tensor* m_input;

	Tensor* m_params; // order: beta, gamma, mean, variance (same as darknet)
	const float eps;
};
