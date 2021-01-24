#include "operator_conv2d_cpu.h"
#include <algorithm>

void operator_conv2d_cpu(float* inputImage, float* weights, float* biases, int batch_size, int h, int w, int c, int filters, int size, int stride, int padding, int out_h, int out_w, bool useBias, float* outputImage)
{
	// input image is flattened version of   h x w x c
	// weights is flattened version of	size x size x c x filters
	// biases has a size of filters
	// output image is flattened version of out_h x out_w x filters

	int h_offset = -padding / 2;
	int w_offset = -padding / 2;
	for (int b = 0; b < batch_size; b++) {
		for (int out_c = 0; out_c < filters; out_c++) {
			for (int i = 0; i < out_h; i++) {
				for (int j = 0; j < out_w; j++) {
					int startI = i * stride + h_offset;
					int startJ = j * stride + w_offset;
					float res = useBias ? biases[out_c] : 0;
					for (int ii = std::max(startI, 0); ii < std::min(startI + size, h); ii++) {
						for (int jj = std::max(startJ, 0); jj < std::min(startJ + size, w); jj++) {
							for (int k = 0; k < c; k++) {
								float imageValue = inputImage[b * h * w * c + ii * w * c + jj * c + k];
								int kI = ii - startI;
								int kJ = jj - startJ;
								float kernelValue = weights[kI * size * c * filters +
									kJ * c * filters + k * filters + out_c];
								res += imageValue * kernelValue;
							}
						}
					}
					outputImage[b * out_h * out_w * filters + i * out_w * filters + j * filters + out_c] = res;
				}
			}
		}
	}
	
}

void operator_conv2d_cpu_backprop(float* inputImage, float* weights, float* biases, int batch_size, int h, int w, int c, int filters, int size, int stride, int padding, int out_h, int out_w, bool useBias, float learning_rate, float* outputImage, float* derivatives, float* prev_derivatives)
{
	// input image is flattened version of   h x w x c
	// weights is flattened version of	size x size x c x filters
	// biases has a size of filters
	// output image is flattened version of out_h x out_w x filters
	// prev_derivatives is the same as input  h x w x c
	// derivatives is the same as output out_h x out_w x filters

	// compute input image derivatives
	int h_offset = -padding / 2;
	int w_offset = -padding / 2;

	if (prev_derivatives) {
		for (int b = 0; b < batch_size; b++) {
			for (int i = 0; i < h; i++) {
				for (int j = 0; j < w; j++) {
					for (int k = 0; k < c; k++) {
						float res = 0;
						int startI = (i - size + 1 - h_offset) / stride;
						int startJ = (j - size + 1 - w_offset) / stride;
						for (int ii = std::max(startI, 0); ii < std::min(startI + size, out_h); ii++) {
							for (int jj = std::max(startJ, 0); jj < std::min(startJ + size, out_w); jj++) {
								for (int kk = 0; kk < filters; kk++) {
									float derivativesValue = derivatives[b * out_h * out_w * filters + ii * out_w * filters + jj * filters + kk];
									int kI = ii - startI;
									int kJ = jj - startJ;
									kI = size - 1 - kI;
									kJ = size - 1 - kJ;
									float kernelValue = weights[kI * size * c * filters +
										kJ * c * filters + k * filters + kk];
									res += derivativesValue * kernelValue;
								}
							}
						}
						prev_derivatives[b * h * w * c + i * w * c + j * c + k] = res;
					}
				}
			}
		}
	}
	
	// update weights and biases
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			for (int k = 0; k < c; k++) {
				for (int f = 0; f < filters; f++) {
					float deriv = 0;

					for (int b = 0; b < batch_size; b++) {
						for (int ii = 0; ii < out_h; ii++) {
							for (int jj = 0; jj < out_w; jj++) {
								int startI = ii * stride + h_offset;
								int startJ = jj * stride + w_offset;
								startI += i;
								startJ += j;
								if (startI < 0 || startJ < 0 || startI >= h || startJ >= w)
									continue;
								deriv += derivatives[b * out_h * out_w * filters + ii * out_w * filters + jj * filters + f] *
									inputImage[b * h * w * c + startI * w * c + startJ * c + k];
							}
						}
					}
					
					weights[i * size * c * filters + j * c * filters + k * filters + f] -= learning_rate * deriv;
				}
			}
		}
	}

	for (int f = 0; f < filters; f++) {
		float deriv = 0;
		for (int b = 0; b < batch_size; b++) {
			for (int ii = 0; ii < out_h; ii++) {
				for (int jj = 0; jj < out_w; jj++) {
					deriv += derivatives[b * out_h * out_w * filters + ii * out_w * filters + jj * filters + f];
				}
			}
		}
		biases[f] -= learning_rate * deriv;
	}

}
