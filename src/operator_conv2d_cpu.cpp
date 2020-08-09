#include "operator_conv2d_cpu.h"
#include <algorithm>

void operator_conv2d_cpu(float * inputImage, int height, int width, int channels,
	float * kernels, int kernelHeight, int kernelWidth, float* kernelBiases, float * outputImage, int numOutputChannels)
{
	// input image is flattened version of   height x width x channels
	// kernels is flattened version of	kernelHeight x kernelWidth x channels x numOutputChannels
	// kernelBiases has a size of	numOutputChannels
	// output image is flattened version of		(height - kernelHeight + 1)	x (width - kernelWidth + 1) x numOutputChannels

	// numOutputChannels is the number of neurons in this conv2d layer
	for (int outputChannel = 0; outputChannel < numOutputChannels; outputChannel++) {
		

		for (int i = 0; i < height - kernelHeight + 1; i++) {
			for (int j = 0; j < width - kernelWidth + 1; j++) {
				float res = kernelBiases[outputChannel];
				for (int kI = 0; kI < kernelHeight; kI++) {
					for (int kJ = 0; kJ < kernelWidth; kJ++) {
						for (int channel = 0; channel < channels; channel++) {
							float imageValue = inputImage[(i+kI)*width*channels+(j+kJ)*channels+channel]; // [i + kI][j + kJ][channel];
							float kernelValue = kernels[kI*kernelWidth*channels*numOutputChannels + 
								kJ*channels*numOutputChannels + channel*numOutputChannels + outputChannel]; // [kI][kJ][channel][outputChannel];
							res += imageValue * kernelValue;
						}
					}
				}
				res = std::max(res, 0.f);
				outputImage[i*(width-kernelWidth+1)*numOutputChannels+j*numOutputChannels+outputChannel] = res; // outputImage[i][j][outputChannel]
			}
		}
		
	}
}
