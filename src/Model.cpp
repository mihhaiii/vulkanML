#include "Model.h"
#include "OperatorFunctionInterface.h"

Model::Model(Tensor* input, Tensor* output, EnumDevice device)
	: m_device(device),
	m_inputs({ input }), m_output(output)
{
	topologicalSort();
	setDevice();

	showInfo();
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

Model::Model(const std::vector<Tensor*>& inputs, Tensor* output, EnumDevice device)
	: m_device(device),
	m_inputs(inputs), m_output(output)
{
	topologicalSort();
	setDevice();

	showInfo();
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

Tensor* Model::run()
{
	setDevice();
	for (Layer* l : sortedLayers) {
		l->forward();
	}
	return m_output;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

Tensor* Model::run(const float* inputData, const int size)
{
	assert(m_inputs.size() == 1);
	batch_size = size / m_inputs[0]->getSampleSize();
	setBatchSize();
	m_inputs[0]->setData(inputData, size);
	return run();
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


Tensor* Model::run(const std::vector<float>& input)
{
	return run(&input[0], input.size());
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

Tensor* Model::run(Tensor* inputTensor)
{
	return run(inputTensor->getData(), inputTensor->getSize());
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

Tensor* Model::run(const std::vector<std::vector<float>>& inputs)
{
	setDevice();
	assert(inputs.size() == m_inputs.size());
	for (int i = 0; i < inputs.size(); i++) {
		m_inputs[i]->setData(&inputs[i][0], inputs[i].size());
	}
	for (Layer* l : sortedLayers) {
		l->forward();
	}
	return m_output;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void Model::setBatchSize()
{
	for (Layer* l : sortedLayers) {
		l->setBatchSize(batch_size);
		l->getOutputTensor()->setBatchSize(batch_size);
		l->getDerivativesTensor()->setBatchSize(batch_size);
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void Model::setLearningRate()
{
	for (Layer* l : sortedLayers) {
		l->setLearningRate(learning_rate);
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void Model::readWeights(const char* filename, int ignoreFirst)
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

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void Model::readDarknetWeights(const char* filename, int ignoreFirst)
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
			Layer* nxt = conv->getOutgoingLayers().front();
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

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int Model::getParamCount()
{
	int res = 0;
	for (Layer* l : sortedLayers) {
		res += l->getParamCount();
	}
	return res;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void Model::showInfo()
{
	for (Layer* l : sortedLayers)
	{
		l->showInfo();
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void Model::fit(Tensor* x, Tensor* y, int epochs, Tensor* test_x, Tensor* test_y, bool printAccuracy)
{
	int samples = x->getShape().front();
	assert(samples == y->getShape().front());
	int sample_size = x->getSize() / samples;

	learning_rate = 0.01;
	setLearningRate();

	randomWeightInit();

	// apply softmax to compute accuracy
	Tensor* inputArgmax = Input({ m_output->getSampleSize() });
	Tensor* argmax = Argmax()(inputArgmax);
	Model* argMaxModel = new Model(inputArgmax, argmax, m_device);

	y->setDevice(m_device);
	for (int e = 0; e < epochs; e++)
	{
		batch_size = 32;
		setBatchSize();
		const int iterations = (samples + batch_size - 1) / batch_size;

		for (int iter = 0; iter < iterations; iter++)
		{
			int firstSampleFromBatch = iter * batch_size;
			int lastSampleFromBatch = (std::min)(firstSampleFromBatch + batch_size, samples);
			int current_batch_size = lastSampleFromBatch - firstSampleFromBatch;

			Tensor* out = run(&x->getData()[firstSampleFromBatch * sample_size], current_batch_size * sample_size);

			assert(m_output == out);
			m_output->getParentLayer()->backprop(y, firstSampleFromBatch);
			for (int l = sortedLayers.size() - 1; l >= 0; l--)
			{
				if (sortedLayers[l] == m_output->getParentLayer())
					continue;
				sortedLayers[l]->backprop();
			}
		}
		if (printAccuracy) {
			// compute train and test accuracy
			Tensor* pred_train = run(x);
			pred_train = argMaxModel->run(pred_train);

			int train_correct_pred = 0, test_correct_pred = 0;
			float* pred_train_data = pred_train->getData();
			float* y_data = y->getData();
			for (int i = 0; i < pred_train->getSize(); i++) {
				if (pred_train_data[i] == y_data[i])
					train_correct_pred++;
			}

			std::cout << "epoch " << e << "   train accuracy: " << (float)train_correct_pred / pred_train->getSize();

			Tensor* pred_test = run(test_x);
			pred_test = argMaxModel->run(pred_test);
			float* pred_test_data = pred_test->getData();
			float* test_y_data = test_y->getData();
			for (int i = 0; i < pred_test->getSize(); i++) {
				if (pred_test_data[i] == test_y_data[i])
					test_correct_pred++;
			}
			std::cout << "   test accuracy: " << (float)test_correct_pred / pred_test->getSize() << "\n";
		}	
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void Model::randomWeightInit()
{
	for (Layer* l : sortedLayers) {
		l->randomWeightInit();
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void Model::topologicalSort()
{
	sortedLayers.clear();
	std::set<Layer*> visited;

	for (Tensor* t : m_inputs) {
		dfs(t->getParentLayer(), visited);
	}

	reverse(sortedLayers.begin(), sortedLayers.end());

	assert(visited.find(m_output->getParentLayer()) != visited.end());
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void Model::setDevice()
{
	for (Layer* l : sortedLayers) {
		l->setDevice(m_device);
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void Model::dfs(Layer* l, std::set<Layer*>& visited)
{
	if (visited.find(l) != visited.end())
		return;

	visited.insert(l);

	for (Layer* n : l->getOutgoingLayers()) {
		dfs(n, visited);
	}
	sortedLayers.push_back(l);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////