#include "Model.h"
#include "OperatorFunctionInterface.h"
#include <time.h>
#include "TensorUtils.h"
#include "GlobalTimeTracker.h"

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

Tensor* Model::run(float* inputData, const int size, bool training)
{
	clock_t start = clock();
	assert(m_inputs.size() == 1);
	setBatchSize(size / m_inputs[0]->getSampleSize(), training);
	float t1 = (float)(clock() - start) / CLOCKS_PER_SEC;
	start = clock();
	m_inputs[0]->setData(inputData, size);
	float t2 = (float)(clock() - start) / CLOCKS_PER_SEC;
	start = clock();
	Tensor* res = run();
	float t3 = (float)(clock() - start) / CLOCKS_PER_SEC;
	return res;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


Tensor* Model::run(std::vector<float>& input, bool training)
{
	return run(&input[0], input.size(), training);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//Tensor* Model::run(Tensor* inputTensor, bool training)
//{
//	return run(inputTensor->getData(), inputTensor->getSize(), training);
//}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

Tensor* Model::run(Tensor* inputTensor, bool training)
{
	{
		SCOPE_TIMER("pre-run");
		assert(m_inputs.size() == 1);
		assert(inputTensor->getSampleSize() == m_inputs[0]->getSampleSize());
		setBatchSize(inputTensor->getShape()[0], training);
		inputTensor->setDevice(m_device);
		m_inputs[0]->shallowCopy(inputTensor);
	}
	return run();
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

Tensor* Model::run(std::vector<std::vector<float>>& inputs)
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

void Model::setBatchSize(float batch_size, bool training)
{
	m_batch_size = batch_size;
	for (Layer* l : sortedLayers) {
		l->setBatchSize(m_batch_size, training);
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void Model::setLearningRate(float learning_rate)
{
	m_learning_rate = learning_rate;
	for (Layer* l : sortedLayers) {
		l->setLearningRate(m_learning_rate);
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
	const bool has_test_data = test_x != nullptr && test_y != nullptr;
	int train_samples = x->getShape().front();
	int test_samples = has_test_data ? test_x->getShape().front() : 0;
	assert(train_samples == y->getShape().front());
	assert(!has_test_data || test_samples == test_y->getShape().front());
	int sample_size = x->getSampleSize();
	int target_size = y->getSampleSize();

	setLearningRate(0.01);
	setBatchSize(32);
	randomWeightInit();

	assert(train_samples >= m_batch_size);
	assert(!has_test_data || test_samples >= m_batch_size);

	const int train_iterations = (train_samples + m_batch_size - 1) / m_batch_size;
	const int test_iterations = (test_samples + m_batch_size - 1) / m_batch_size;

	std::vector<std::pair<Tensor*, Tensor*>> train_minibatches(train_iterations);
	std::vector<std::pair<Tensor*, Tensor*>> test_minibatches;
	Shape minibatch_x_shape = x->getShape(); minibatch_x_shape[0] = m_batch_size;
	Shape minibatch_y_shape = y->getShape(); minibatch_y_shape[0] = m_batch_size;
	for (int i = 0; i < train_iterations; i++)
	{
		int firstSampleFromBatch = i * m_batch_size;
		int lastSampleFromBatch = (std::min)(firstSampleFromBatch + m_batch_size, train_samples);
		firstSampleFromBatch = lastSampleFromBatch - m_batch_size; // complete the last incomplete batch with previous samples

		train_minibatches[i].first = new Tensor(minibatch_x_shape, m_device);
		train_minibatches[i].second = new Tensor(minibatch_y_shape, m_device);
		train_minibatches[i].first->setData(&x->getData()[firstSampleFromBatch * sample_size], m_batch_size * sample_size);
		train_minibatches[i].second->setData(&y->getData()[firstSampleFromBatch * target_size], m_batch_size * target_size);
	}

	if (has_test_data) {
		test_minibatches.resize(test_iterations);
		for (int i = 0; i < test_iterations; i++)
		{
			int firstSampleFromBatch = i * m_batch_size;
			int lastSampleFromBatch = (std::min)(firstSampleFromBatch + m_batch_size, test_samples);
			firstSampleFromBatch = lastSampleFromBatch - m_batch_size; // complete the last incomplete batch with previous samples

			test_minibatches[i].first = new Tensor(minibatch_x_shape, m_device);
			test_minibatches[i].second = new Tensor(minibatch_y_shape, m_device);
			test_minibatches[i].first->setData(&test_x->getData()[firstSampleFromBatch * sample_size], m_batch_size * sample_size);
			test_minibatches[i].second->setData(&test_y->getData()[firstSampleFromBatch * target_size], m_batch_size * target_size);
		}
	}

	float t1 = 0, t2 = 0, t3 = 0, t4 = 0, t5 = 0;
	clock_t start;

	Model* argMaxModel = nullptr;
	if (printAccuracy) {
		// apply softmax to compute accuracy
		Tensor* inputArgmax = Input({ m_output->getSampleSize() });
		Tensor* argmax = Argmax()(inputArgmax);
		argMaxModel = new Model(inputArgmax, argmax, m_device);
	}

	for (int e = 0; e < epochs; e++)
	{	
		clock_t epochStartTime = clock();
		GlobalTimeTracker::getInstance().reset();
		for (int i = 0; i < train_iterations; i++)
		{
			start = clock();
			Tensor* out = run(train_minibatches[i].first, true /* training */);

			t2 += (float)(clock() - start) / CLOCKS_PER_SEC;
			start = clock();

			assert(m_output == out);
			m_output->getParentLayer()->backprop(train_minibatches[i].second);
			t3 += (float)(clock() - start) / CLOCKS_PER_SEC;
			start = clock();
			for (int l = sortedLayers.size() - 1; l >= 0; l--)
			{
				if (sortedLayers[l] == m_output->getParentLayer())
					continue;
				sortedLayers[l]->backprop();
			}
			t4 += (float)(clock() - start) / CLOCKS_PER_SEC;
		}
		start = clock();
		if (printAccuracy) {
			// compute train accuracy
			int train_correct_pred = 0;
			for (int i = 0; i < train_iterations; i++)
			{
				Tensor* out = run(train_minibatches[i].first);
				out = argMaxModel->run(out);
				int eq = CountEqual(out, train_minibatches[i].second); //! on gpu, changes device data of 'out'
				train_correct_pred += eq;
			}
			float train_accuracy = (float)train_correct_pred / (train_iterations * m_batch_size);
			std::cout << "epoch " << e << "   train accuracy: " << train_accuracy;

			// compute test accuracy
			if (has_test_data) {
				int test_correct_pred = 0;
				for (int i = 0; i < test_iterations; i++)
				{
					Tensor* out = run(test_minibatches[i].first);
					out = argMaxModel->run(out);
					int eq = CountEqual(out, test_minibatches[i].second);
					test_correct_pred += eq;
				}
				float test_accuracy = (float)test_correct_pred / (test_iterations * m_batch_size);
				std::cout << "   test accuracy: " << test_accuracy; 
			}
			std::cout << "\n";
		}
		t5 += (float)(clock() - start) / CLOCKS_PER_SEC;
		
		float epochDuration = (float)(clock() - epochStartTime) / CLOCKS_PER_SEC;
		std::cout << "epoch duration: " << epochDuration << "s\n";
		std::cout << t1 << " " << t2 << " " << t3 << " " << t4 << " " << t5 << "\n";
		GlobalTimeTracker::getInstance().show();
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