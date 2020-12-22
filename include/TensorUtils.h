#pragma once
#include <vector>
class Tensor;

std::vector<Tensor*> split(Tensor* t, const std::vector<int>& dims, int axis = -1);
void sigmoid(Tensor* t); // in-place