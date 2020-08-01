#pragma once

void operator_maxpool_cpu(float * inputImage, int height, int width, int channels,
	 int kernelHeight, int kernelWidth, float * outputImage);