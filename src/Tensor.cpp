#include "TensorUtils.h"
#include "MemoryManager.h"
#include <stdlib.h>
#include "Tensor.h"
#include "InstanceManager.h"
#include <random>
#include <time.h>

int getShapeSize(const Shape& shape) {
	int total = 1;
	assert(shape.size() != 0);
	for (auto& dim : shape) {
		total *= dim;
	}
	return total;
}

std::ostream& operator<<(std::ostream& out, const Shape& shape)
{
	out << "(";
	out << shape[0];
	for (int i = 1; i < shape.size(); i++) {
		out << ", " << shape[i];
	}
	out << ")";
	return out;
}

void readBinFile(const char* fileName, float* out, int count) {
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

void readBinFile(FILE* file, float* out, int count)
{
	int read = fread(out, sizeof(float), count, file);
	assert(read == count);
}


Tensor::Tensor(const Shape& shape, EnumDevice device, const char* binFile)
	: m_shape(shape),
	m_device(device),
	m_data(nullptr),
	m_deviceData(nullptr)
{
	Shape::iterator findIt = std::find(m_shape.begin(), m_shape.end(), -1);
	m_hasUnknownDim = findIt != m_shape.end();
	if (m_hasUnknownDim) {
		m_unknownDimIdx = findIt - m_shape.begin();
		m_size = -1;
	}
	else {
		m_unknownDimIdx = -1;
		m_size = getShapeSize(shape);
	}
	
	setDevice(device);

	if (!m_hasUnknownDim)
	{
		if (binFile) {
			readData(binFile);
		}
	}

	createShapeSuffixProduct();
}

Tensor::~Tensor() {
	delete[] m_data;
	delete m_deviceData;
	m_deviceData = nullptr;
	m_data = nullptr;
}

int Tensor::getSampleSize()
{
	// assuming first dim is the batch_size, return the product of all the other dims
	return m_shapeSuffixProduct[1];
}

void Tensor::reshape(Shape newShape)
{
	assert(!m_hasUnknownDim);
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


void Tensor::randomWeightInit(const float mean, const float standardDeviation)
{
	std::default_random_engine generator(time(0));
	std::normal_distribution<float> distribution(mean, standardDeviation);

	assert(!m_hasUnknownDim);
	if (m_data == nullptr) {
		m_data = new float[m_size];
	}
	for (int i = 0; i < m_size; i++) {
		m_data[i] = distribution(generator);
	}
	if (m_device & DEVICE_VULKAN) {
		m_deviceData->fromHost(m_data, m_data + m_size);
	}
}


float& Tensor::at(const Shape& loc)
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


void Tensor::createShapeSuffixProduct()
{
	m_shapeSuffixProduct.resize(m_shape.size() + 1);
	m_shapeSuffixProduct.back() = 1;
	for (int i = m_shapeSuffixProduct.size() - 2; i >= 0; i--) {
		m_shapeSuffixProduct[i] = m_shapeSuffixProduct[i + 1] * m_shape[i];
	}
}

void Tensor::setDevice(EnumDevice device) {

	EnumDevice prevDevice = m_device;
	m_device = device;
	
	if (m_hasUnknownDim) {
		return; // if we don't know entirely the shape of the tensor delay the allocations until we do
	}
	if (m_device & DEVICE_CPU)
	{
		if (m_data == nullptr) {
			m_data = new float[m_size];
			reset();
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

void Tensor::reset()
{
	for (int i = 0; i < m_size; i++)
		m_data[i] = 0;
}

bool Tensor::inc(Shape& it)
{
	assert(!m_hasUnknownDim);
	for (int i = it.size() - 1; i >= 0; i--) {
		if (it[i] + 1 < m_shape[i]) {
			it[i]++;
			return true;
		}
		it[i] = 0;
	}
	return false;
}

void Tensor::readData(const char* binFile) {
	assert(m_device != DEVICE_UNSPECIFIED);
	assert(!m_hasUnknownDim);
	if (!m_data)
		m_data = new float[m_size];
	readBinFile(binFile, m_data, m_size);
	if (m_device & DEVICE_VULKAN) {
		m_deviceData->fromHost(m_data, m_data + m_size);
	}
}

void Tensor::readData(FILE* file) {
	assert(m_device != DEVICE_UNSPECIFIED);
	assert(!m_hasUnknownDim);
	if (!m_data)
		m_data = new float[m_size];
	readBinFile(file, m_data, m_size);
	if (m_device & DEVICE_VULKAN) {
		m_deviceData->fromHost(m_data, m_data + m_size);
	}
}

void Tensor::setBatchSize(int batch_size)
{
	int currentBatchSize = m_shape.front();
	if (batch_size == currentBatchSize)
		return;

	m_shape[0] = batch_size;
	m_hasUnknownDim = false;
	m_size = getShapeSize(m_shape);
	delete[] m_data;
	delete m_deviceData;
	m_data = nullptr;
	m_deviceData = nullptr;
	setDevice(m_device); // allocate memory now
	createShapeSuffixProduct(); // update the suffix product vector as the shape is entirely known now
}

void Tensor::setData(const float* data, int size)
{
	assert(m_device != DEVICE_UNSPECIFIED);
	if (!m_hasUnknownDim) {
		assert(size == m_size);
	}
	else {
		setUnknownDimBasedOnTotalSize(size);
	}
	if (m_device & DEVICE_CPU) {
		for (int i = 0; i < m_size; i++) {
			m_data[i] = data[i];
		}
	}
	if (m_device & DEVICE_VULKAN) {
		m_deviceData->fromHost(data, data + m_size);
	}
}

float* Tensor::getData() {
	assert(m_device != DEVICE_UNSPECIFIED);
	if (m_device & DEVICE_CPU)
		return m_data;
	else if (m_device & DEVICE_VULKAN)
		return getDataFromDeviceData();
	else assert(false); // unknown device
}

float* Tensor::getDataFromDeviceData() {
	if (!m_data)
		m_data = new float[m_size];
	m_deviceData->toHost(m_data);
	return m_data;
}

void Tensor::setUnknownDimBasedOnTotalSize(int totalSize)
{
	m_size = totalSize;
	int currentSize = 1;
	for (auto x : m_shape) {
		if (x != -1)
			currentSize *= x;
	}
	assert(totalSize % currentSize == 0);
	m_shape[m_unknownDimIdx] = totalSize / currentSize;
	m_hasUnknownDim = false;
	setDevice(m_device); // allocate memory now
	createShapeSuffixProduct(); // update the suffix product vector as the shape is entirely known now
}