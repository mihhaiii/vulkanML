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

	std::vector<float> img = loadTestImage("data/meme.jpg");
	
	Tensor* input = Input({ imageSize,imageSize,3 });
	Tensor* x, *x_8, *output_0, *output_1, *boxes_0, *boxes_1, *outputs;

	// Darknet tiny
	x = Conv2D(16, 3, 1, "same", false)(input);
	x = BatchNorm()(x);
	x = LeakyReLU()(x);
	x = MaxPool2D(2, 2, "same")(x);

	x = Conv2D(32, 3, 1, "same", false)(x);
	x = BatchNorm()(x);
	x = LeakyReLU()(x);
	x = MaxPool2D(2, 2, "same")(x);

	x = Conv2D(64, 3, 1, "same", false)(x);
	x = BatchNorm()(x);
	x = LeakyReLU()(x);
	x = MaxPool2D(2, 2, "same")(x);

	x = Conv2D(128, 3, 1, "same", false)(x);
	x = BatchNorm()(x);
	x = LeakyReLU()(x);
	x = MaxPool2D(2, 2, "same")(x);

	x = Conv2D(256, 3, 1, "same", false)(x);
	x = BatchNorm()(x);
	x = LeakyReLU()(x);
	x = MaxPool2D(2, 2, "same")(x);

	x = Conv2D(512, 3, 1, "same", false)(x);
	x = BatchNorm()(x);
	x = LeakyReLU()(x);
	x = MaxPool2D(2, 2, "same")(x);

	x = Conv2D(1024, 3, 1, "same", false)(x);
	x = BatchNorm()(x);
	x = LeakyReLU()(x);

	// YoloConvTiny
	int filters = 256;
	int anchors = 3, classes = 80;
	x = Conv2D(filters, 1, 1, "same", false)(x);
	x = BatchNorm()(x);
	x_8 = x = LeakyReLU()(x);
	// Yolo output
	output_0 = Conv2D(filters * 2, 3, 1, "same", false)(x);
	output_0 = BatchNorm()(output_0);
	output_0 = LeakyReLU()(output_0);
	output_0 = Conv2D(anchors * (classes + 5), 1, 1, "same", true)(output_0);
	
	// YoloConvTiny of (x, x_8)
	filters = 128;
	x = Conv2D(filters, 1, 1, "same", false)(x);
	x = BatchNorm()(x);
	x = LeakyReLU()(x);
	//x = UpSampling2D(2)(x); // TODO
	//x = Concat({ x, x_8 }); // TODO
	// Yolo output
	output_1 = Conv2D(filters * 2, 3, 1, "same", false)(x);
	output_1 = BatchNorm()(output_1);
	output_1 = LeakyReLU()(output_1);
	output_1 = Conv2D(anchors * (classes + 5), 1, 1, "same", true)(output_1);

	// yolo boxes
	//boxes_0 = Lambda([&](Tensor* t) {})(output_0); // TODO
	//boxes_1 = Lambda([&](Tensor* t) {})(output_1); // TODO

	// outputs = Lambda(yolo_nms)({boxes_0, boxes_1}); // TODO

	Model* m = new Model(input, outputs);
	m->run(img);

	// show outputs TODO
	// draw outputs TODO
	
	return 0;
}