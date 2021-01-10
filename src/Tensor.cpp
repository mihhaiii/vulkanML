#include "TensorUtils.h"
#include "MemoryManager.h"
#include <stdlib.h>
#include "Tensor.h"
#include "InstanceManager.h"

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

Tensor::~Tensor() {
	delete[] m_data;
	delete m_deviceData;
	m_deviceData = nullptr;
}

void Tensor::reshape(Shape newShape)
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


void Tensor::randomWeightInit(const float factor)
{
	if (m_data == nullptr) {
		m_data = new float[m_size];
	}
	for (int i = 0; i < m_size; i++) {
		m_data[i] = ((double)rand() / (RAND_MAX)) * factor;
	}
	if (m_device * DEVICE_VULKAN) {
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

bool Tensor::inc(Shape& it)
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

void Tensor::readData(const char* binFile) {
	assert(m_device != DEVICE_UNSPECIFIED);
	if (!m_data)
		m_data = new float[m_size];
	readBinFile(binFile, m_data, m_size);
	if (m_device & DEVICE_VULKAN) {
		m_deviceData->fromHost(m_data, m_data + m_size);
	}
}

void Tensor::readData(FILE* file) {
	assert(m_device != DEVICE_UNSPECIFIED);
	if (!m_data)
		m_data = new float[m_size];
	readBinFile(file, m_data, m_size);
	if (m_device & DEVICE_VULKAN) {
		m_deviceData->fromHost(m_data, m_data + m_size);
	}
}

void Tensor::setData(const std::vector<float>& data)
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