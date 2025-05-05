#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include "imageUtils.h"
#include <climits> // For INT_MAX
#include <cuda_runtime.h>

__global__ void stereoKernel(unsigned char* left, unsigned char* right, unsigned char* depth, int maxDisparity, int rows, int cols) {

        int x = blockIdx.x*blockDim.x + threadIdx.x;
        int y = blockIdx.y*blockDim.y + threadIdx.y;
    
    // === STEP 4: Disparity parameters ===
    int windowSize = 11;
    int halfWindow = windowSize / 2;

    // Camera parameters (from calibration)
    float focalLength = 578.86f;        // pixels
    float baseline = 0.0589621f;        // meters

    // === STEP 5: Compute disparity using SAD block matching ===
    //std::cout << "[INFO] Computing disparity and depth...\n";

  //  for (int y = halfWindow; y < rows - halfWindow; ++y) {
  //      for (int x = halfWindow; x < cols - halfWindow; ++x) {
            int bestDisparity = 0;
            int minSAD = INT_MAX;

            for (int d = 0; d < maxDisparity; ++d) {
                int xr = x - d;
                if (xr - halfWindow < 0)
                    break;

                int SAD = 0;
                for (int wy = -halfWindow; wy <= halfWindow; ++wy) {
                    for (int wx = -halfWindow; wx <= halfWindow; ++wx) {
                        int leftPixel = left[(y + wy) * cols + (x + wx)];
                        int rightPixel = right[(y + wy) * cols + (x + wx) - d];
                        SAD += std::abs(leftPixel - rightPixel);
                    }
                }

                if (SAD < minSAD) {
                    minSAD = SAD;
                    bestDisparity = d;
                }
            }

            if (bestDisparity > 0) {
                depth[y * cols + x]  = (focalLength * baseline) / static_cast<float>(bestDisparity);
            } else {
                depth[y * cols + x] = 0;
            }
     //   }
   // }


}
