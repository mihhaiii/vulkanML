#include "operator_maxpool_cpu.h"
#include <float.h>
#include <algorithm>

void operator_maxpool_cpu(float* inputImage,int h, int w, int c, int size, int stride, int padding, int out_h, int out_w, float* outputImage)
{
	// input image is flattened version of   h x w x c
	// output image is flattened version of out_h x out_w x c
	
	int h_offset = -padding / 2;
	int w_offset = -padding / 2;
	for (int i = 0; i < out_h; i++) {
		for (int j = 0; j < out_w; j++) {
			for (int k = 0; k < c; k++) {
				int startI = i * stride + h_offset;
				int startJ = j * stride + w_offset;
				float maxValue = -FLT_MAX;
				for (int ii = std::max(startI,0); ii < std::min(startI + size, h); ii++) {
					for (int jj = std::max(startJ,0); jj < std::min(startJ + size, w); jj++) {
						float currentValue = inputImage[ii*w*c + jj*c + k]; 
						if (currentValue > maxValue)
							maxValue = currentValue;
					}
				}
				outputImage[i*out_w*c + j*c + k] = maxValue;
			}
		}
	}
}
