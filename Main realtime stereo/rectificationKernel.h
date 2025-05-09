#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <limits.h>

 __global__ void rectificationKernel(unsigned char* left, unsigned char* right);


