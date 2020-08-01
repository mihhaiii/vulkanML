#include "operator_maxpool_cpu.h"

void operator_maxpool_cpu(float * inputImage,int height, int width, int channels,
	 int kernelHeight, int kernelWidth, float * outputImage)
{
	// input image is flattened version of   height x width x channels
	// output image is flattened version of (height / kernelHeight) x (width / kernelWidth) x channels,
	int outWidth = width / kernelWidth;
	int outHeight = height / kernelHeight;
	
	for (int i = 0; i < outHeight; i++) {
		for (int j = 0; j < outWidth; j++) {
			for (int c = 0; c < channels; c++) {
				int startI = i * kernelHeight;
				int startJ = j * kernelWidth;
				float maxValue = inputImage[startI*width*channels + startJ*channels + c];
				for (int ii = startI; ii < startI + kernelHeight; ii++) {
					for (int jj = startJ; jj < startJ + kernelWidth; jj++) {
						float currentValue = inputImage[ii*width*channels + jj*channels + c]; 
						if (currentValue > maxValue)
							maxValue = currentValue;
					}
				}
				outputImage[i*outWidth*channels + j*channels + c] = maxValue;
			}
		}
	}
}
