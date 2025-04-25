#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include "imageUtils.h"
#include <climits> // For INT_MAX


int main() {
    std::cout << "[INFO] Starting stereo.cc (disparity + depth computation)\n";

    // === STEP 1: Load rectified stereo images ===
    cv::Mat leftColor = cv::imread("left_rectified.ppm");
    cv::Mat rightColor = cv::imread("right_rectified.ppm");

    if (leftColor.empty() || rightColor.empty()) {
        std::cerr << "[ERROR] Failed to load left_rectified.ppm or right_rectified.ppm!\n";
        return -1;
    }
    std::cout << "[INFO] Successfully loaded rectified images.\n";

    // === STEP 2: Convert to grayscale ===
    cv::Mat grayLeft, grayRight;
    try {
        cv::cvtColor(leftColor, grayLeft, cv::COLOR_BGR2GRAY);
        cv::cvtColor(rightColor, grayRight, cv::COLOR_BGR2GRAY);
        std::cout << "[INFO] Grayscale conversion successful.\n";
    } catch (cv::Exception& e) {
        std::cerr << "[FATAL] cvtColor failed: " << e.what() << std::endl;
        return -1;
    }

    int rows = grayLeft.rows;
    int cols = grayLeft.cols;

    // === STEP 3: Prepare output images ===
    cv::Mat disparityMap(rows, cols, CV_8U, cv::Scalar(0));
    cv::Mat depthMap(rows, cols, CV_32F, cv::Scalar(0.0f));

    // === STEP 4: Disparity parameters ===
    int windowSize = 5;
    int halfWindow = windowSize / 2;
    int maxDisparity = 64;

    // Camera parameters (from calibration)
    float focalLength = 578.86f;        // pixels
    float baseline = 0.0589621f;        // meters

    // === STEP 5: Compute disparity using SAD block matching ===
    std::cout << "[INFO] Computing disparity and depth...\n";

    for (int y = halfWindow; y < rows - halfWindow; ++y) {
        for (int x = halfWindow; x < cols - halfWindow; ++x) {
            int bestDisparity = 0;
            int minSAD = INT_MAX;

            for (int d = 0; d < maxDisparity; ++d) {
                int xr = x - d;
                if (xr - halfWindow < 0)
                    break;

                int SAD = 0;
                for (int wy = -halfWindow; wy <= halfWindow; ++wy) {
                    for (int wx = -halfWindow; wx <= halfWindow; ++wx) {
                        int leftPixel = grayLeft.at<uchar>(y + wy, x + wx);
                        int rightPixel = grayRight.at<uchar>(y + wy, xr + wx);
                        SAD += std::abs(leftPixel - rightPixel);
                    }
                }

                if (SAD < minSAD) {
                    minSAD = SAD;
                    bestDisparity = d;
                }
            }

            // === STEP 6: Save disparity (scaled) and compute depth ===
            disparityMap.at<uchar>(y, x) = static_cast<uchar>((bestDisparity * 255) / maxDisparity);

            if (bestDisparity > 0) {
                float depth = (focalLength * baseline) / static_cast<float>(bestDisparity);
                depthMap.at<float>(y, x) = depth;
            } else {
                depthMap.at<float>(y, x) = 0.0f;
            }
        }
    }

    // === STEP 7: Prepare final output images (convert to 3-channel BGR) ===
    cv::Mat disparityColor, depthVis, depthColor;
    cv::normalize(depthMap, depthVis, 0, 255, cv::NORM_MINMAX, CV_8U);

    cv::cvtColor(disparityMap, disparityColor, cv::COLOR_GRAY2BGR);
    cv::cvtColor(depthVis, depthColor, cv::COLOR_GRAY2BGR);

    // === STEP 8: Save results as PPM ===
    cv::imwrite("disparity_custom.ppm", disparityColor);
    cv::imwrite("depth_map.ppm", depthColor);
    std::cout << "[INFO] Saved disparity_custom.ppm and depth_map.ppm\n";

    return 0;
}
