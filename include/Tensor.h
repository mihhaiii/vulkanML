#pragma once

#include <vector>
#include <iostream>
#include "vuh/array.hpp"


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
	Tensor(const Shape& shape, EnumDevice device = EnumDevice::DEVICE_UNSPECIFIED, const char* binFile = nullptr);
	~Tensor();

	void createShapeSuffixProduct();

	void setDevice(EnumDevice device);

	EnumDevice getDevice() { return m_device; }

	Shape begin() { return std::vector<int>(m_shape.size(), 0); }

	bool inc(Shape& it);

	void readData(const char* binFile);

	void readData(FILE* file);

	void setData(const float* data, int size);

	float* getData();
	vuh::Array<float>* getDeviceData() { return m_deviceData; }
	float* getDataFromDeviceData();
	int getSize() { return m_size; }
	int getSampleSize();
	Shape getShape() { return m_shape; }

	float& at(const Shape& loc);

	void reshape(Shape newShape);

	void randomWeightInit(const float mean, const float standardDeviation);

	Layer* getParentLayer() { return m_parentLayer; }

	void setBatchSize(int batch_size);

	void reset();

	friend class Layer;
private:
	void setUnknownDimBasedOnTotalSize(int totalSize);

	bool m_hasUnknownDim;
	int m_unknownDimIdx;
	int m_size;
	Shape m_shape;
	std::vector<int> m_shapeSuffixProduct;
	float* m_data;
	vuh::Array<float>* m_deviceData;
	EnumDevice m_device;

	Layer* m_parentLayer; // the layer that owns this tensor or the last last layer that modifies this tensor
};
