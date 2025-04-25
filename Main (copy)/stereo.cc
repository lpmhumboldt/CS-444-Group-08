#include "imageUtils.h"
#include "readParams.h"
#include "utils.h"
#include <iostream>
#include <cmath>
#include <climits>
#include <vector>
#include <algorithm>

#define WINDOW_SIZE 5  // Increased block matching window size for better clarity
#define MAX_DISPARITY 50  // Maximum disparity to search
#define PARAM_COUNT  9 // Number of calibration parameters

using namespace std;

// Compute Sum of Absolute Differences (SAD)
int computeSAD(PPMImage* left, PPMImage* right, int x, int y, int d) {
    int halfWindow = WINDOW_SIZE / 2;
    int sum = 0;

    for (int i = -halfWindow; i <= halfWindow; i++) {
        for (int j = -halfWindow; j <= halfWindow; j++) {
            int lx = x + i;
            int ly = y + j;
            int rx = lx - d; // Shift right image by disparity d

            if (lx >= 0 && ly >= 0 && lx < left->width && ly < left->height &&
                rx >= 0 && rx < right->width) {
                int leftPixel = left->data[(ly * left->width + lx) * 3];
                int rightPixel = right->data[(ly * right->width + rx) * 3];

                sum += abs(leftPixel - rightPixel);
            }
        }
    }
    return sum;
}

// Compute depth map using disparity
void computeDepthMap(PPMImage* left, PPMImage* right, unsigned char* depthMap, double focalLength, double baseline) {
    for (int y = WINDOW_SIZE / 2; y < left->height - WINDOW_SIZE / 2; y++) {
        for (int x = WINDOW_SIZE / 2; x < left->width - WINDOW_SIZE / 2; x++) {
            int minSAD = INT_MAX;
            int bestDisparity = 0;

            for (int d = 0; d < MAX_DISPARITY; d++) {
                int SAD = computeSAD(left, right, x, y, d);
                if (SAD < minSAD) {
                    minSAD = SAD;
                    bestDisparity = d;
                }
            }

            // Convert disparity to depth (inverse relationship)
            if (bestDisparity > 0) {
                double depth = (focalLength * baseline) / bestDisparity;
                depthMap[y * left->width + x] = (unsigned char)(255 - (depth * 255 / MAX_DISPARITY));
            } else {
                depthMap[y * left->width + x] = 0; // Assign black if disparity is zero
            }
        }
    }
}

// Apply median filter for noise reduction
void applyMedianFilter(unsigned char* depthMap, int width, int height) {
    int filterSize = 5;  // 5x5 median filter
    int halfSize = filterSize / 2;
    vector<unsigned char> temp(depthMap, depthMap + (width * height));

    for (int y = halfSize; y < height - halfSize; y++) {
        for (int x = halfSize; x < width - halfSize; x++) {
            vector<unsigned char> window;
            for (int dy = -halfSize; dy <= halfSize; dy++) {
                for (int dx = -halfSize; dx <= halfSize; dx++) {
                    window.push_back(temp[(y + dy) * width + (x + dx)]);
                }
            }
            nth_element(window.begin(), window.begin() + window.size() / 2, window.end());
            depthMap[y * width + x] = window[window.size() / 2];
        }
    }
}

// Normalize depth values for better contrast
void normalizeDepthMap(unsigned char* depthMap, int width, int height) {
    int minDepth = 255, maxDepth = 0;
    for (int i = 0; i < width * height; i++) {
        if (depthMap[i] > 0) {  // Ignore zero disparity areas
            minDepth = min(minDepth, (int)depthMap[i]);
            maxDepth = max(maxDepth, (int)depthMap[i]);
        }
    }

    // Normalize depth values to 0-255 range
    for (int i = 0; i < width * height; i++) {
        if (depthMap[i] > 0) {
            depthMap[i] = (depthMap[i] - minDepth) * 255 / (maxDepth - minDepth);
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        cout << "Usage: " << argv[0] << " <left_image.jpg> <right_image.jpg> <output_depth.ppm>" << endl;
        return 1;
    }

    // Convert input JPEG images to PPM format
    convertJPGToPPM(argv[1], "left_temp.ppm");
    convertJPGToPPM(argv[2], "right_temp.ppm");

    // Load images
    PPMImage* leftImage = readPPM("left_temp.ppm", 1);
    PPMImage* rightImage = readPPM("right_temp.ppm", 1);
    if (!leftImage || !rightImage) {
        cerr << "Error loading images!" << endl;
        return 1;
    }

    // Allocate depth map
    unsigned char* depthMap = new unsigned char[leftImage->width * leftImage->height];

    // Camera parameters
    double focalLength = 700.0;  // Example focal length
    double baseline = 0.1;       // Example distance between cameras in meters

    // Compute depth map
    computeDepthMap(leftImage, rightImage, depthMap, focalLength, baseline);

    // Apply median filter
    applyMedianFilter(depthMap, leftImage->width, leftImage->height);

    // Normalize depth values
    normalizeDepthMap(depthMap, leftImage->width, leftImage->height);

    // Save depth map as grayscale
    writePPM(argv[3], leftImage->width, leftImage->height, 255, 0, depthMap);

    // Cleanup
    delete[] depthMap;
    freePPM(leftImage);
    freePPM(rightImage);

    // Remove temporary PPM files
    remove("left_temp.ppm");
    remove("right_temp.ppm");

    cout << "Depth map saved successfully to " << argv[3] << endl;
    return 0;
}
