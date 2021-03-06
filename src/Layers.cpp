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
#include "operator_softmax_cpu.h"
#include "vulkan_operator_softmax.h"
#include "GlobalTimeTracker.h"

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
	m_batch_size = inputImage->getShape()[0];
	h = inputImage->getShape()[1];
	w = inputImage->getShape()[2];
	c = inputImage->getShape()[3];


	out_h = (h + padding - size) / stride + 1;
	out_w = (w + padding - size) / stride + 1;

	outputImage = createOwnedTensor({ m_batch_size, out_h, out_w, filters });
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
	SCOPE_TIMER("conv_2d_forward");
	assert(m_isInputLinked);
	assert(m_device != DEVICE_UNSPECIFIED);
	if (m_device == DEVICE_CPU) {
		operator_conv2d_cpu(inputImage->getData(), weigths->getData(), biases->getData(), m_batch_size, h, w, c, filters, size, stride, padding, out_h, out_w, useBias, outputImage->getData());
	}
	else if (m_device == DEVICE_VULKAN) {
		vulkan_operator_conv2d(inputImage->getDeviceData(), weigths->getDeviceData(), biases->getDeviceData(), m_batch_size, h, w, c, filters, size, stride, padding, out_h, out_w, useBias, outputImage->getDeviceData());
	}
}

 void Conv2dLayer::backprop()
 {
	 SCOPE_TIMER("conv_2d_backprop");
	 Tensor* prev_derivatives = inputImage->getParentLayer()->getDerivativesTensor();
	 if (m_device == DEVICE_CPU) {
		 operator_conv2d_cpu_backprop(inputImage->getData(), weigths->getData(), biases->getData(), m_batch_size, h, w, c, filters, size, stride, padding, 
			 out_h, out_w, useBias, m_learning_rate, outputImage->getData(), m_derivatives->getData(), prev_derivatives ? prev_derivatives->getData() : nullptr);
	 }
	 else if (m_device == DEVICE_VULKAN) {
		 vulkan_operator_conv2d_backprop(inputImage->getDeviceData(), weigths->getDeviceData(), biases->getDeviceData(), m_batch_size, h, w, c, filters, size, stride, padding, 
			 out_h, out_w, useBias, m_learning_rate, outputImage->getDeviceData(), m_derivatives->getDeviceData(), prev_derivatives ? prev_derivatives->getDeviceData() : nullptr);
	 }
 }

 void Conv2dLayer::randomWeightInit()
 {
	 float standardDeviation = sqrt(2. / (m_batch_size * h * w));
	 float mean = 0.f;
	 weigths->randomWeightInit(mean, standardDeviation);
	 biases->randomWeightInit(mean, standardDeviation);
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
	m_batch_size = m_inputImage->getShape()[0];
	h = m_inputImage->getShape()[1];
	w = m_inputImage->getShape()[2];
	c = m_inputImage->getShape()[3];

	out_h = (h + padding - size) / stride + 1;
	out_w = (w + padding - size) / stride + 1;

	m_outputImage = createOwnedTensor({ m_batch_size, out_h, out_w, c });
}

int MaxPooling2dLayer::getParamCount() 
{
	return 0;
}

void MaxPooling2dLayer::forward() {
	SCOPE_TIMER("max_pool_forward");
	assert(m_device != DEVICE_UNSPECIFIED);
	if (m_device == DEVICE_CPU) {
		operator_maxpool_cpu(m_inputImage->getData(), m_batch_size, h, w, c, size, stride, padding, out_h, out_w, m_outputImage->getData());
	}
	else if (m_device == DEVICE_VULKAN) {
		vulkan_operator_maxpool(m_inputImage->getDeviceData(), m_batch_size, h, w, c, size, stride, padding, out_h, out_w, m_outputImage->getDeviceData());
	}
}

void MaxPooling2dLayer::backprop() {
	SCOPE_TIMER("max_pool_backprop");
	assert(m_device != DEVICE_UNSPECIFIED);
	if (m_device == DEVICE_CPU) {
		operator_maxpool_cpu_backprop(m_inputImage->getData(), m_batch_size, h, w, c, size, stride, padding, out_h, out_w, 
			m_outputImage->getData(), m_derivatives->getData(), m_inputImage->getParentLayer()->getDerivativesTensor()->getData());
	}
	else if (m_device == DEVICE_VULKAN) {
		vulkan_operator_maxpool_backprop(m_inputImage->getDeviceData(), m_batch_size, h, w, c, size, stride, padding, out_h, out_w,
			m_outputImage->getDeviceData(), m_derivatives->getDeviceData(), m_inputImage->getParentLayer()->getDerivativesTensor()->getDeviceData());
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
	m_numNeuronsLastLayer = m_input->getSampleSize();

	// if batch size not known yet (==1), cannot allocate the output tensor yet but can allocate the weights and the biases
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
	SCOPE_TIMER("fc_forward");
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

void FCLayer::randomWeightInit()
{
	float standardDeviation = sqrt(2. / m_input->getSampleSize());
	float mean = 0.f;
	m_weights->randomWeightInit(mean, standardDeviation);
	m_biases->randomWeightInit(mean, standardDeviation);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void FCLayer::backprop()
{
	SCOPE_TIMER("fc_backprop");
	assert(m_derivatives);
	Tensor* prev_derivatives = m_input->getParentLayer()->getDerivativesTensor();

	if (m_device == DEVICE_CPU) {
		float* prev_derivativeData = prev_derivatives ? prev_derivatives->getData() : nullptr;
		float* derivativeData = m_derivatives->getData();
		float* weightsData = m_weights->getData();
		float* biasesData = m_biases->getData();
		float* inputData = m_input->getData();

		if (m_applyRelu) {
			for (int i = 0; i < m_output->getSize(); i++) {
				derivativeData[i] = m_output->getData()[i] == 0 ? 0 : derivativeData[i];
			}
		}

		if (prev_derivativeData) {
			for (int b = 0; b < m_batch_size; b++) {
				for (int i = 0; i < m_numNeuronsLastLayer; i++) {
					prev_derivativeData[b * m_numNeuronsLastLayer + i] = 0;
					for (int j = 0; j < m_numNeurons; j++) {
						//prev_derivatives->at({ b, i }) += m_derivatives->at({ b, j }) * m_weights->at({i, j});
						prev_derivativeData[b * m_numNeuronsLastLayer + i] += derivativeData[b * m_numNeurons + j] * weightsData[i * m_numNeurons + j];
					}
				}
			}
		}
		
		float deriv = 0;
		for (int i = 0; i < m_numNeuronsLastLayer; i++) {
			for (int j = 0; j < m_numNeurons; j++) {
				deriv = 0;
				for (int b = 0; b < m_batch_size; b++) {
					deriv += derivativeData[b * m_numNeurons + j] * inputData[b * m_numNeuronsLastLayer + i];
				}
				weightsData[i * m_numNeurons + j] -= m_learning_rate * deriv;
			}
		}

		for (int j = 0; j < m_numNeurons; j++) {
			deriv = 0;
			for (int b = 0; b < m_batch_size; b++) {
				deriv += derivativeData[b * m_numNeurons + j];
			}
			biasesData[j] -= m_learning_rate * deriv;
		}
	}
	else if (m_device == DEVICE_VULKAN) {
		vulkan_operator_FC_backprop(m_input->getDeviceData(), m_weights->getDeviceData(), m_biases->getDeviceData(),
			m_derivatives->getDeviceData(), prev_derivatives ? prev_derivatives->getDeviceData() : nullptr, m_output->getDeviceData(), m_numNeuronsLastLayer, m_numNeurons, m_batch_size, m_applyRelu, m_learning_rate);
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void ReLULayer::init(const std::vector<Tensor*>& inputs) 
{
	assert(inputs.size() == 1);
	m_input = inputs[0];
	m_output = createOwnedTensor(m_input->getShape());
}

 void ReLULayer::forward() {
	SCOPE_TIMER("relu_forward");
	assert(m_device != DEVICE_UNSPECIFIED);
	if (m_device == EnumDevice::DEVICE_CPU) {
		for (int i = 0; i < m_input->getSize(); i++) {
			m_output->getData()[i] = (std::max)(m_input->getData()[i], 0.f);
		}
	}
	else if (m_device == EnumDevice::DEVICE_VULKAN) {
		vulkan_operator_relu(m_input->getDeviceData(), m_output->getDeviceData(), m_input->getSize());
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

 void ReLULayer::backprop()
 {
	 SCOPE_TIMER("relu_backprop");
	 assert(m_derivatives);
	 Tensor* prev_derivatives = m_input->getParentLayer()->getDerivativesTensor();

	 if (m_device == DEVICE_CPU) {
		 for (int i = 0; i < m_input->getSize(); i++) {
			 prev_derivatives->getData()[i] = m_input->getData()[i] < 0 ? 0 : m_derivatives->getData()[i];
		 }
	 }
	 else if (m_device == DEVICE_VULKAN) {
		 vulkan_operator_relu_backprop(m_input->getDeviceData(), m_derivatives->getDeviceData(), prev_derivatives->getDeviceData(), m_input->getSize());
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
	m_batch_size = m_input->getShape()[0];
	h = m_input->getShape()[1];
	w = m_input->getShape()[2];
	c = m_input->getShape()[3];
	out_h = h * size;
	out_w = w * size;
	m_output = createOwnedTensor({ m_batch_size, out_h, out_w, c });
}

void UpSampling2DLayer::forward() {
	assert(m_device != DEVICE_UNSPECIFIED);
	if (m_device == EnumDevice::DEVICE_CPU) {
		for (int b = 0; b < m_batch_size; b++) {
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
		vulkan_operator_upSampling_2D(m_input->getDeviceData(), m_output->getDeviceData(), m_batch_size, h, w, c, size);
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
	m_batch_size = m_input1->getShape()[0];
	int h1 = m_input1->getShape()[1];
	int w1 = m_input1->getShape()[2];
	c1 = m_input1->getShape()[3];

	assert(m_batch_size == m_input2->getShape()[0]);
	int h2 = m_input2->getShape()[1];
	int w2 = m_input2->getShape()[2];
	c2 = m_input2->getShape()[3];

	assert(h1 == h2);
	assert(w1 == w2);
	h = h1; w = w1;
	m_output = createOwnedTensor({ m_batch_size, h, w, c1 + c2 });
}

void ConcatenateLayer::forward() {
	assert(m_device != DEVICE_UNSPECIFIED);
	if (m_device == EnumDevice::DEVICE_CPU) {
		for (int b = 0; b < m_batch_size; b++) {
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
		vulkan_operator_concatenate(m_input1->getDeviceData(), m_input2->getDeviceData(), m_output->getDeviceData(), m_batch_size, h, w, c1, c2);
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

void SoftmaxLayer::init(const std::vector<Tensor*>& inputs)
{
	assert(inputs.size() == 1);
	m_input = inputs[0];

	m_output = createOwnedTensor(m_input->getShape());
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void SoftmaxLayer::forward()
{
	SCOPE_TIMER("softmax_forward");
	assert(m_device != DEVICE_UNSPECIFIED);
	if (m_device == EnumDevice::DEVICE_CPU) {
		operator_softmax_cpu(m_input->getData(), m_output->getData(), m_batch_size, m_input->getSampleSize());
	}
	else if (m_device == EnumDevice::DEVICE_VULKAN) {
		vulkan_operator_softmax(m_input->getDeviceData(), m_output->getDeviceData(), m_batch_size, m_input->getSampleSize());
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void SoftmaxLayer::backprop(Tensor* ground_truth)
{
	SCOPE_TIMER("softmax_backprop");
	// assume loss = sparse categorical cross entropy and ground_truth format is NOT one-hot
	Tensor* derivatives = m_input->getParentLayer()->getDerivativesTensor();
	int sample_size = derivatives->getSampleSize();

	if (m_device == EnumDevice::DEVICE_CPU) {
		derivatives->reset();
		float inv_batch_size = 1.f / m_batch_size;
		for (int b = 0; b < m_batch_size; b++) {
			for (int i = 0; i < sample_size; i++) {
					derivatives->at({ b,i }) += inv_batch_size * (m_output->at({ b, i }) - (round(ground_truth->getData()[b]) == i ? 1 : 0));
			}
		}
	}
	else if (m_device == EnumDevice::DEVICE_VULKAN) {
		vulkan_operator_softmax_backprop(derivatives->getDeviceData(), m_output->getDeviceData(), ground_truth->getDeviceData(), m_batch_size, sample_size);
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void ArgmaxLayer::init(const std::vector<Tensor*>& inputs)
{
	assert(inputs.size() == 1);
	m_input = inputs[0];
	m_batch_size = m_input->getShape().front();
	m_output = createOwnedTensor({ m_batch_size });
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void ArgmaxLayer::forward()
{
	SCOPE_TIMER("argmax_forward");
	assert(m_device != DEVICE_UNSPECIFIED);
	if (m_device == EnumDevice::DEVICE_CPU) {
		int sample_size = m_input->getSampleSize();
		float* inputData = m_input->getData();
		float* outputData = m_output->getData();
		for (int b = 0; b < m_batch_size; b++) {
			float mx = -FLT_MAX;
			int mxIdx = -1;
			int startIdx = b * sample_size;
			int endIdx = startIdx + sample_size;
			for (int i = startIdx; i < endIdx; i++) {
				if (inputData[i] > mx) {
					mx = inputData[i];
					mxIdx = i - startIdx;
				}
			}
			outputData[b] = mxIdx;
		}
	}
	else if (m_device == EnumDevice::DEVICE_VULKAN) {
		vulkan_operator_argmax(m_input->getDeviceData(), m_output->getDeviceData(), m_batch_size, m_input->getSampleSize());
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////