#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include "imageUtils.h"
#include <climits> // For INT_MAX
#include <cuda_runtime.h>

__global__ void stereoKernel(unsigned char* left, unsigned char* right, unsigned char* depth, int maxDisparity, int rows, int cols) {

    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    float disparityThreshold = 4;
    
    // === STEP 4: Disparity parameters ===
    int windowSize = 13;
    int halfWindow = (windowSize-1) / 2;

    // Camera parameters (from calibration)
    float focalLength = 578.86f;        // pixels
    float baseline = 0.0589621f;        // meters

    // === STEP 5: Compute disparity using SAD block matching ===
    //std::cout << "[INFO] Computing disparity and depth...\n";
  
            float bestDisparity = 0;
            int minSAD = INT_MAX;

            for (int d = 0; d < maxDisparity; ++d) {
                int xr = x - d;
                if (xr - halfWindow < 0)
                    break;

                int SAD = 0;
                for (int wy = -halfWindow; wy <= halfWindow; ++wy) {
                    for (int wx = -halfWindow; wx <= halfWindow; ++wx) {
                        int leftPixel =(int) left[(y + wy) * cols + (x + wx)];
                        int rightPixel =(int) right[(y + wy) * cols + (x + wx) - d];
                        SAD += std::abs(leftPixel - rightPixel);
			
                    }
                }

                if (SAD < minSAD) {
                    minSAD = SAD;
                    bestDisparity = d;
                }
            }
	if (bestDisparity > 0) {
           // if (bestDisparity > 0 && bestDisparity > disparityThreshold) {
                //depth[y * cols + x] = (unsigned char)(bestDisparity);
		depth[y * cols + x]  = (unsigned char)((focalLength * baseline) / (float)(bestDisparity));
            } else {
                depth[y * cols + x] =(unsigned char)(0);
            }

	    
     printf("at  %d, %d, bestDisparty is: %.3f, disparityThreshold is: %.3f\n", x, y, bestDisparity, disparityThreshold);   

    /* if (bestDisparity > disparityThreshold) {
     	printf("bestDisparity is greater than disparityThreshold\n");
     } else {
     	printf("bestDisparity is greater than disparityThreshold\n");
     }*/

}
