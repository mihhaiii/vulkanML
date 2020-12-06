#include <iostream>
#include <cassert>
#include "InferenceFramework.h"
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_resize.h"
#include "stb_image_write.h"

using namespace std;

const int imageSize = 416;


std::vector<float> loadTestImage(const char* filename) {
	int width, height, channels;
	unsigned char* image = stbi_load(filename,
		&width,
		&height,
		&channels,
		0);
	assert(channels == 3);
	unsigned char* resizedImage = new unsigned char[imageSize * imageSize * channels];
	stbir_resize_uint8(image, width, height, 0, resizedImage, imageSize, imageSize, 0, channels);

	std::vector<float> img(imageSize * imageSize * channels);
	for (int i = 0; i < img.size(); i++) {
		img[i] = (float)resizedImage[i] / 255.0;
	}
	stbi_write_png("my_image_output.png", imageSize, imageSize, channels, resizedImage, 0);
	delete[] image;
	delete[] resizedImage;
	return img;
}

int main()
{
	std::cout << "Yolo tiny" << std::endl;

	SequentialModel m(EnumDevice::DEVICE_CPU);

	std::vector<float> img = loadTestImage("data/meme.jpg");
	m.addInputLayer({ imageSize, imageSize, 3 })->fill(img);
	
	// DarknetTiny
	m.addConv2dLayer(16, 3 /*size*/, 1 /*stride*/, "same" /*padding*/, false /*useBias*/);
	m.addBatchNormalizationLayer();
	m.addLeakyReLULayer();
	m.addMaxPooling2dLayer(2 /*size*/, 2 /*stride*/, "same");

	m.addConv2dLayer(32, 3 /*size*/, 1 /*stride*/, "same" /*padding*/, false /*useBias*/);
	m.addBatchNormalizationLayer();
	m.addLeakyReLULayer();
	m.addMaxPooling2dLayer(2 /*size*/, 2 /*stride*/, "same");

	m.addConv2dLayer(64, 3 /*size*/, 1 /*stride*/, "same" /*padding*/, false /*useBias*/);
	m.addBatchNormalizationLayer();
	m.addLeakyReLULayer();
	m.addMaxPooling2dLayer(2 /*size*/, 2 /*stride*/, "same");

	m.addConv2dLayer(128, 3 /*size*/, 1 /*stride*/, "same" /*padding*/, false /*useBias*/);
	m.addBatchNormalizationLayer();
	m.addLeakyReLULayer();
	m.addMaxPooling2dLayer(2 /*size*/, 2 /*stride*/, "same");

	m.addConv2dLayer(256, 3 /*size*/, 1 /*stride*/, "same" /*padding*/, false /*useBias*/);
	m.addBatchNormalizationLayer();
	m.addLeakyReLULayer();
	m.addMaxPooling2dLayer(2 /*size*/, 2 /*stride*/, "same");

	m.addConv2dLayer(512, 3 /*size*/, 1 /*stride*/, "same" /*padding*/, false /*useBias*/);
	m.addBatchNormalizationLayer();
	m.addLeakyReLULayer();
	m.addMaxPooling2dLayer(2 /*size*/, 2 /*stride*/, "same");

	m.addConv2dLayer(1024, 3 /*size*/, 1 /*stride*/, "same" /*padding*/, false /*useBias*/);
	m.addBatchNormalizationLayer();
	m.addLeakyReLULayer();

	// YoloConvTiny
	m.addConv2dLayer(256, 1 /*size*/, 1 /*stride*/, "same" /*padding*/, false /*useBias*/);
	m.addBatchNormalizationLayer();
	m.addLeakyReLULayer();


	
	return 0;
}