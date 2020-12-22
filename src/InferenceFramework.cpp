#include "InferenceFramework.h"
#include "TensorUtils.h"
#include "MemoryManager.h"

int getShapeSize(const Shape& shape) {
	int total = 1;
	assert(shape.size() != 0);
	for (auto& dim : shape) {
		total *= dim;
	}
	return total;
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
