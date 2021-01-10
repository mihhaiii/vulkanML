#include "Layers.h"
#include "operator_FC_cpu.h"
#include "vulkan_operator_FC.h"
#include "operator_maxpool_cpu.h"
#include "vulkan_operator_maxpool.h"
#include "operator_conv2d_cpu.h"
#include "vulkan_operator_conv2d.h"
#include "vulkan_operator_relu.h"
#include "vulkan_operator_leaky_relu.h"
#include "vulkan_operator_batch_normalization.h"
#include "vulkan_operator_upSampling_2D.h"
#include "vulkan_operator_concatenate.h"

Conv2dLayer::Conv2dLayer(int _filters, int _size, int _stride, const std::string& _padding, bool _useBias)
	: filters(_filters), size(_size), stride(_stride), useBias(_useBias)
{
	assert(_padding == "same" || _padding == "valid");
	padding = _padding == "valid" ? 0 : size - 1;
}

void Conv2dLayer::init(const std::vector<Tensor*>& inputs) 
{
	assert(inputs.size() == 1);
	inputImage = inputs[0];

	assert(inputImage->getShape().size() == 4);
	batch_size = inputImage->getShape()[0];
	h = inputImage->getShape()[1];
	w = inputImage->getShape()[2];
	c = inputImage->getShape()[3];


	out_h = (h + padding - size) / stride + 1;
	out_w = (w + padding - size) / stride + 1;

	outputImage = createOwnedTensor({ batch_size, out_h, out_w, filters });
	weigths = createOwnedTensor({ size, size, c, filters });
	biases = createOwnedTensor({ filters });
}

int Conv2dLayer::getParamCount() 
{
	assert(m_isInputLinked);
	return weigths->getSize() + (useBias ? biases->getSize() : 0);
}

void Conv2dLayer::readWeigthsFromFile(FILE* file) 
{
	readWeights(file);
	readBiases(file);
}

void Conv2dLayer::readWeights(const char* binFile) {
	weigths->readData(binFile);
}

void Conv2dLayer::readWeights(FILE* file) {
	weigths->readData(file);
}

void Conv2dLayer::readWeightsAndTranspose(FILE* file) {
	Tensor* tmp = new Tensor({ filters, c, size, size }, DEVICE_CPU);
	tmp->readData(file);

	EnumDevice oldDevice = weigths->getDevice();
	weigths->setDevice(DEVICE_CPU);
	int sz = tmp->getSize();
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

void Conv2dLayer::readBiases(const char* binFile) {
	if (useBias)
		biases->readData(binFile);
}

void Conv2dLayer::readBiases(FILE* file) {
	if (useBias)
		biases->readData(file);
}

 void Conv2dLayer::forward() {
	assert(m_isInputLinked);
	assert(m_device != DEVICE_UNSPECIFIED);
	if (m_device == DEVICE_CPU) {
		operator_conv2d_cpu(inputImage->getData(), weigths->getData(), biases->getData(), batch_size, h, w, c, filters, size, stride, padding, out_h, out_w, useBias, outputImage->getData());
	}
	else if (m_device == DEVICE_VULKAN) {
		vulkan_operator_conv2d(inputImage->getDeviceData(), weigths->getDeviceData(), biases->getDeviceData(), batch_size, h, w, c, filters, size, stride, padding, out_h, out_w, useBias, outputImage->getDeviceData());
		outputImage->setDevice(DEVICE_CPU);
		outputImage->setDevice(DEVICE_VULKAN);
	}
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


MaxPooling2dLayer::MaxPooling2dLayer(int _size, int _stride, std::string _padding)
	: size(_size), stride(_stride)
{
	assert(_padding == "same" || _padding == "valid");
	padding = _padding == "valid" ? 0 : size - 1;
}

void MaxPooling2dLayer::init(const std::vector<Tensor*>& inputs) 
{
	assert(inputs.size() == 1);
	m_inputImage = inputs[0];

	assert(m_inputImage->getShape().size() == 4);
	batch_size = m_inputImage->getShape()[0];
	h = m_inputImage->getShape()[1];
	w = m_inputImage->getShape()[2];
	c = m_inputImage->getShape()[3];

	out_h = (h + padding - size) / stride + 1;
	out_w = (w + padding - size) / stride + 1;

	m_outputImage = createOwnedTensor({ batch_size, out_h, out_w, c });
}

int MaxPooling2dLayer::getParamCount() 
{
	return 0;
}

void MaxPooling2dLayer::forward() {
	assert(m_device != DEVICE_UNSPECIFIED);
	if (m_device == DEVICE_CPU) {
		operator_maxpool_cpu(m_inputImage->getData(), batch_size, h, w, c, size, stride, padding, out_h, out_w, m_outputImage->getData());
	}
	else if (m_device == DEVICE_VULKAN) {
		vulkan_operator_maxpool(m_inputImage->getDeviceData(), batch_size, h, w, c, size, stride, padding, out_h, out_w, m_outputImage->getDeviceData());
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

FCLayer::FCLayer(int numNeurons, bool applyRelu)
	: m_numNeurons(numNeurons), m_applyRelu(applyRelu) {}

void FCLayer::init(const std::vector<Tensor*>& inputs) 
{
	assert(inputs.size() == 1);
	m_input = inputs[0];
	m_batch_size = m_input->getShape().front(); // first dim is the batch size
	m_numNeuronsLastLayer = m_input->getSize() / m_batch_size;
	m_weights = createOwnedTensor({ m_numNeuronsLastLayer, m_numNeurons });
	m_biases = createOwnedTensor({ m_numNeurons });
	m_output = createOwnedTensor({ m_batch_size, m_numNeurons });
}

int FCLayer::getParamCount() 
{
	assert(m_isInputLinked);
	return m_weights->getSize() + m_biases->getSize();
}

 void FCLayer::readWeigthsFromFile(FILE* file) 
{
	readWeights(file);
	readBiases(file);
}

void FCLayer::readWeights(const char* binFile) {
	m_weights->readData(binFile);
}

void FCLayer::readBiases(const char* binFile) {
	m_biases->readData(binFile);
}

void FCLayer::readWeights(FILE* file) {
	m_weights->readData(file);
}

void FCLayer::readBiases(FILE* file) {
	m_biases->readData(file);
}

void FCLayer::forward() {
	assert(m_device != DEVICE_UNSPECIFIED);
	if (m_device == DEVICE_CPU) {
		operator_FC_cpu(m_input->getData(), m_weights->getData(), m_biases->getData(), m_output->getData(),
			m_numNeuronsLastLayer, m_numNeurons, m_batch_size, m_applyRelu);
	}
	else if (m_device == DEVICE_VULKAN) {
		vulkan_operator_FC(m_input->getDeviceData(), m_weights->getDeviceData(), m_biases->getDeviceData(),
			m_output->getDeviceData(), m_numNeuronsLastLayer, m_numNeurons, m_batch_size, m_applyRelu);
	}
}

void FCLayer::randomWeightInit(const float factor)
{
	m_weights->randomWeightInit(factor);
	m_biases->randomWeightInit(factor);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void ReLULayer::init(const std::vector<Tensor*>& inputs) 
{
	assert(inputs.size() == 1);
	m_input = inputs[0];
}

 void ReLULayer::forward() {
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

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void LeakyReLULayer::init(const std::vector<Tensor*>& inputs) 
{
	assert(inputs.size() == 1);
	m_input = inputs[0];
}

void LeakyReLULayer::forward() {
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

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void UpSampling2DLayer::init(const std::vector<Tensor*>& inputs) 
{
	assert(inputs.size() == 1);
	m_input = inputs[0];

	assert(m_input->getShape().size() == 4);
	batch_size = m_input->getShape()[0];
	h = m_input->getShape()[1];
	w = m_input->getShape()[2];
	c = m_input->getShape()[3];
	out_h = h * size;
	out_w = w * size;
	m_output = createOwnedTensor({ batch_size, out_h, out_w, c });
}

void UpSampling2DLayer::forward() {
	assert(m_device != DEVICE_UNSPECIFIED);
	if (m_device == EnumDevice::DEVICE_CPU) {
		for (int b = 0; b < batch_size; b++) {
			for (int i = 0; i < out_h; i++) {
				for (int j = 0; j < out_w; j++) {
					for (int k = 0; k < c; k++) {
						m_output->at({ b,i,j,k }) = m_input->at({ b, i / size, j / size, k });
					}
				}
			}
		}
	}
	else if (m_device == EnumDevice::DEVICE_VULKAN) {
		vulkan_operator_upSampling_2D(m_input->getDeviceData(), m_output->getDeviceData(), batch_size, h, w, c, size);
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void ConcatenateLayer::init(const std::vector<Tensor*>& inputs) 
{
	assert(inputs.size() == 2);
	m_input1 = inputs[0];
	m_input2 = inputs[1];

	assert(m_input1->getShape().size() == 4); // batch_size, h, w, c
	assert(m_input2->getShape().size() == 4);
	batch_size = m_input1->getShape()[0];
	int h1 = m_input1->getShape()[1];
	int w1 = m_input1->getShape()[2];
	c1 = m_input1->getShape()[3];

	assert(batch_size == m_input2->getShape()[0]);
	int h2 = m_input2->getShape()[1];
	int w2 = m_input2->getShape()[2];
	c2 = m_input2->getShape()[3];

	assert(h1 == h2);
	assert(w1 == w2);
	h = h1; w = w1;
	m_output = createOwnedTensor({ batch_size, h, w, c1 + c2 });
}

void ConcatenateLayer::forward() {
	assert(m_device != DEVICE_UNSPECIFIED);
	if (m_device == EnumDevice::DEVICE_CPU) {
		for (int b = 0; b < batch_size; b++) {
			for (int i = 0; i < h; i++) {
				for (int j = 0; j < w; j++) {
					for (int k = 0; k < c1 + c2; k++) {
						m_output->at({ b,i,j,k }) = k < c1 ? m_input1->at({ b, i, j, k }) : m_input2->at({ b, i, j, k - c1 });
					}
				}
			}
		}
	}
	else if (m_device == EnumDevice::DEVICE_VULKAN) {
		vulkan_operator_concatenate(m_input1->getDeviceData(), m_input2->getDeviceData(), m_output->getDeviceData(), batch_size, h, w, c1, c2);
	}
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void BatchNormalizationLayer::init(const std::vector<Tensor*>& inputs) 
{
	assert(inputs.size() == 1);
	m_input = inputs[0];

	int filters = m_input->getShape().back(); // assuming data_format == channels_last
	m_params = createOwnedTensor({ 4, filters });
}

int BatchNormalizationLayer::getParamCount() 
{
	assert(m_isInputLinked);
	return m_params->getSize();
}

void BatchNormalizationLayer::forward() {
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

void BatchNormalizationLayer::readWeigthsFromFile(FILE* file) 
{
	readParams(file);
}

void BatchNormalizationLayer::readParams(const char* filename)
{
	m_params->readData(filename);
	//validate
	for (int i = 0; i < m_input->getShape().back(); i++) {
		float variance = m_params->at({ 3, i });
		assert(variance >= 0);
	}
}

void BatchNormalizationLayer::readParams(FILE* file)
{
	m_params->readData(file);
	//validate
	for (int i = 0; i < m_input->getShape().back(); i++) {
		float variance = m_params->at({ 3, i });
		assert(variance >= 0);
	}
}

void BatchNormalizationLayer::setParams(const std::vector<float>& inParams)
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

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////