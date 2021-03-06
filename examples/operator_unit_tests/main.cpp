#include <iostream>
#include <cassert>
#include "unit_tests.h"


using namespace std;


int main()
{
	std::cout << "Running unit tests..." << std::endl;
	
	test_batch_norm();	
	
	test_UpSampling2D();

	test_concatenate();

	test_tensor_split();

	return 0;
}