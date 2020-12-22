#include <iostream>
#include <cassert>
#include "InferenceFramework.h"
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_resize.h"
#include "stb_image_write.h"
#include "OperatorFunctionInterface.h"
#include <fstream>
#include "TensorUtils.h"

using namespace std;

const int imageSize = 416;
const int anchors = 3;
const int classes = 80;


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

void yolo_boxes(Tensor* t)
{
	t->reshape({ t->getShape()[0], t->getShape()[1], anchors, classes + 5 });
	// t.shape = (grid, grid, anchors, (x, y, w, h, obj, ...classes))
	int h = t->getShape()[0];
	int w = t->getShape()[1];

	t->setDevice(DEVICE_CPU);
	auto tensors = split(t, { 2,2,1,classes }, -1);
	Tensor* box_xy = tensors[0]; 
	Tensor* box_wh = tensors[1];
	Tensor* objectness = tensors[2];
	Tensor* class_probs = tensors[3];
	sigmoid(box_xy);
	sigmoid(objectness);
	sigmoid(class_probs);
}

int main()
{
	std::cout << "Yolo tiny" << std::endl;

	std::vector<float> img = loadTestImage("data/meme.jpg");
	
	Tensor* input = Input({ imageSize,imageSize,3 });
	Tensor* x, *x_8, *output_0, *output_1, *boxes_0, *boxes_1;

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
	x_8 = x = LeakyReLU()(x);
	x = MaxPool2D(2, 2, "same")(x);

	x = Conv2D(512, 3, 1, "same", false)(x);
	x = BatchNorm()(x);
	x = LeakyReLU()(x);
	x = MaxPool2D(2, 1, "same")(x);

	x = Conv2D(1024, 3, 1, "same", false)(x);
	x = BatchNorm()(x);
	x = LeakyReLU()(x);

	// YoloConvTiny
	int filters = 256;
	int anchors = 3, classes = 80;
	x = Conv2D(filters, 1, 1, "same", false)(x);
	x = BatchNorm()(x);
	x = LeakyReLU()(x);
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
	x = UpSampling2D(2)(x); 
	x = Concatenate()({ x, x_8 });
	// Yolo output
	output_1 = Conv2D(filters * 2, 3, 1, "same", false)(x);
	output_1 = BatchNorm()(output_1);
	output_1 = LeakyReLU()(output_1);
	output_1 = Conv2D(anchors * (classes + 5), 1, 1, "same", true)(output_1);

	// yolo boxes
	//boxes_0 = Lambda([&](Tensor* t) {})(output_0); // TODO
	//boxes_1 = Lambda([&](Tensor* t) {})(output_1); // TODO

	// outputs = Lambda(yolo_nms)({boxes_0, boxes_1}); // TODO

	Model* m = new Model(input, output_1, DEVICE_CPU);
	m->readWeights("data/yolov3-tiny.weights", 5); // major, minor, revision, seen, _
	std::cout << "Model params: " << m->getParamCount() << "\n";
	m->run(img);
	
	yolo_boxes(output_0);
	yolo_boxes(output_1);

	std::ifstream in_file("data/yolov3-tiny.weights", std::ios::binary);
	in_file.seekg(0, std::ios::end);
	int file_size = in_file.tellg();
	std::cout << "Size of the file is " << file_size << " " << "bytes which is " << file_size / 4 << " params\n";
	in_file.close();

	// show outputs TODO
	// draw outputs TODO
	
	return 0;
}