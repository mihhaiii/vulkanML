#include "operator_conv2d_cpu.h"
#include <algorithm>

void operator_conv2d_cpu(float* inputImage, float* weights, float* biases, int h, int w, int c, int filters, int size, int stride, int padding, int out_h, int out_w, bool useBias, float* outputImage)
{
	// input image is flattened version of   h x w x c
	// weights is flattened version of	size x size x c x filters
	// biases has a size of filters
	// output image is flattened version of out_h x out_w x filters

	int h_offset = -padding / 2;
	int w_offset = -padding / 2;
	for (int out_c = 0; out_c < filters; out_c++) {
		for (int i = 0; i < out_h; i++) {
			for (int j = 0; j < out_w; j++) {
				int startI = i * stride + h_offset;
				int startJ = j * stride + w_offset;
				float res = useBias ? biases[out_c] : 0;
				for (int ii = std::max(startI,0); ii < std::min(startI + size, h); ii++) {
					for (int jj = std::max(startJ,0); jj < std::min(startJ + size, w); jj++) {
						for (int k = 0; k < c; k++) {
							float imageValue = inputImage[ii*w*c + jj*c + k];
							int kI = ii - startI;
							int kJ = jj - startJ;
							float kernelValue = weights[kI*size*c*filters +  
								kJ*c*filters + k*filters + out_c];
							res += imageValue * kernelValue;
						}
					}
				}
				outputImage[i*out_w*filters + j*filters + out_c] = res;
			}
		}	
	}
}
