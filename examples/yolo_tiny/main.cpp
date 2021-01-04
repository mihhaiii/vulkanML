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

std::vector<std::vector<float>> tiny_yolo_anchors0;
std::vector<std::vector<float>> tiny_yolo_anchors1;

std::vector<std::string> classNames = { 
 "person			"
,"bicycle			"
,"car				"
,"motorbike			"
,"aeroplane			"
,"bus				"
,"train				"
,"truck				"
,"boat				"
,"traffic light		"
,"fire hydrant		"
,"stop sign			"
,"parking meter		"
,"bench				"
,"bird				"
,"cat				"
,"dog				"
,"horse				"
,"sheep				"
,"cow				"
,"elephant			"
,"bear				"
,"zebra				"
,"giraffe			"
,"backpack			"
,"umbrella			"
,"handbag			"
,"tie				"
,"suitcase			"
,"frisbee			"
,"skis				"
,"snowboard			"
,"sports ball		"
,"kite				"
,"baseball bat		"
,"baseball glove	"
,"skateboard		"
,"surfboard			"
,"tennis racket		"
,"bottle			"
,"wine glass		"
,"cup				"
,"fork				"
,"knife				"
,"spoon				"
,"bowl				"
,"banana			"
,"apple				"
,"sandwich			"
,"orange			"
,"broccoli			"
,"carrot			"
,"hot dog			"
,"pizza				"
,"donut				"
,"cake				"
,"chair				"
,"sofa				"
,"pottedplant		"
,"bed				"
,"diningtable		"
,"toilet			"
,"tvmonitor			"
,"laptop			"
,"mouse				"
,"remote			"
,"keyboard			"
,"cell phone		"
,"microwave			"
,"oven				"
,"toaster			"
,"sink				"
,"refrigerator		"
,"book				"
,"clock				"
,"vase				"
,"scissors			"
,"teddy bear		"
,"hair drier		"
,"toothbrush		"
};

void initAnchors()
{
	tiny_yolo_anchors0 = { {81 / 416.f, 82 / 416.f}, {135 / 416.f, 169 / 416.f},  {344 / 416.f, 319 / 416.f} };
	tiny_yolo_anchors1 = { {10 / 416.f, 14 / 416.f}, {23 / 416.f, 27 / 416.f}, {37 / 416.f, 58 / 416.f} };
}

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

void yolo_boxes(Tensor* t, std::vector<std::vector<float>>& yolo_anchors)
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
	tensor_sigmoid(box_xy);

	std::vector<float> obj;
	float* rawData = objectness->getData();
	obj.assign(rawData, rawData + objectness->getSize());

	tensor_sigmoid(objectness);
	tensor_sigmoid(class_probs);

	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {
			for (int a = 0; a < anchors; a++) {
				float& x = box_xy->at({ i,j,a,0 });
				float& y = box_xy->at({ i,j,a,1 });
				x = (x + i) / h;
				y = (y + j) / w;
			}
		}
	}

	tensor_exp(box_wh);
	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {
			for (int a = 0; a < anchors; a++) {
				for (int c = 0; c < 2; c++) {
					float& x = box_wh->at({ i,j,a,c });
					x *= yolo_anchors[a][c];
				}
			}
		}
	}

	Tensor* box_x1y1 = new Tensor(box_xy->getShape(), DEVICE_CPU);
	Tensor* box_x2y2 = new Tensor(box_xy->getShape(), DEVICE_CPU);

	for (int i = 0; i < h; i++) {
		for (int j = 0; j < w; j++) {
			for (int a = 0; a < anchors; a++) {
				for (int c = 0; c < 2; c++) {
					float& x1y1 = box_x1y1->at({ i,j,a,c });
					float& x2y2 = box_x2y2->at({ i,j,a,c });
					float& wh = box_wh->at({ i,j,a,c });
					float& xy = box_xy->at({ i,j,a,c });
					x1y1 = xy - wh / 2.f;
					x2y2 = xy + wh / 2.f;
				}
			}
		}
	}
	box_x1y1->reshape({ -1, 2 }); // a list of pairs
	box_x2y2->reshape({ -1, 2 }); // a list of pairs
	objectness->reshape({ -1 });  // of list of floats
	class_probs->reshape({ -1, classes }); // a list of tuples of 80 elements

	for (int i = 0; i < objectness->getSize(); i++) {
		if (objectness->getData()[i] > 0.2f) {
			float mx = -1;
			int mxIdx = 0;
			for (int j = 0; j < classes; j++) {
				if (class_probs->at({ i, j }) > mx) {
					mx = class_probs->at({ i, j });
					mxIdx = j;
				}
			}
			std::cout << classNames[mxIdx] << class_probs->at({i, mxIdx}) << " -> " << class_probs->at({ i, mxIdx }) * objectness->getData()[i] << "\n";
		}
	}
}

void inspect(Tensor* x)
{
	float* data = x->getData();
	std::vector<float> v;
	v.assign(data, data + x->getSize());
}

int main()
{
	std::cout << "Yolo tiny" << std::endl;
	initAnchors();

	std::vector<float> img = loadTestImage("data/street.jpg");
	
	Tensor* x, *x_8, *output_0, *output_1, *boxes_0, *boxes_1, *x_tmp, *x_tmp1, *input;
	input = x_tmp = Input({ imageSize,imageSize,3 });

	// Darknet tiny
	x_tmp1 = x = Conv2D(16, 3, 1, "same", false)(input);
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
	output_0 = x;
	
	
	// YoloConvTiny of (x, x_8)
	filters = 128;
	x = Conv2D(filters, 1, 1, "same", false)(x);
	x = BatchNorm()(x);
	x = LeakyReLU()(x);
	x = UpSampling2D(2)(x); 
	x = Concatenate()({ x, x_8 });

	// Yolo output
	filters = 256;
	output_0 = Conv2D(filters * 2, 3, 1, "same", false)(output_0);
	output_0 = BatchNorm()(output_0);
	output_0 = LeakyReLU()(output_0);
	output_0 = Conv2D(anchors * (classes + 5), 1, 1, "same", true)(output_0);

	// Yolo output
	filters = 128;
	output_1 = Conv2D(filters * 2, 3, 1, "same", false)(x);
	output_1 = BatchNorm()(output_1);
	output_1 = LeakyReLU()(output_1);
	output_1 = Conv2D(anchors * (classes + 5), 1, 1, "same", true)(output_1);

	// yolo boxes
	//boxes_0 = Lambda([&](Tensor* t) {})(output_0); // TODO
	//boxes_1 = Lambda([&](Tensor* t) {})(output_1); // TODO

	// outputs = Lambda(yolo_nms)({boxes_0, boxes_1}); // TODO

	Model* m = new Model(input, output_1, DEVICE_VULKAN);
	m->readDarknetWeights("data/yolov3-tiny.weights", 5); // major, minor, revision, seen, _
	std::cout << "Model params: " << m->getParamCount() << "\n";
	m->run(img);
	
	inspect(x_tmp);
	inspect(x_tmp1);

	yolo_boxes(output_0, tiny_yolo_anchors0);
	yolo_boxes(output_1, tiny_yolo_anchors1);

	std::ifstream in_file("data/yolov3-tiny.weights", std::ios::binary);
	in_file.seekg(0, std::ios::end);
	int file_size = in_file.tellg();
	std::cout << "Size of the file is " << file_size << " " << "bytes which is " << file_size / 4 << " params\n";
	in_file.close();

	// show outputs TODO
	// draw outputs TODO
	
	return 0;
}