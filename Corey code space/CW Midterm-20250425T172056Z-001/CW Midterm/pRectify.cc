#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main() {
    // Load color stereo images
    Mat leftImageColor  = imread("left.jpg");
    Mat rightImageColor = imread("right.jpg");

    if (leftImageColor.empty() || rightImageColor.empty()) {
        cerr << "[ERROR] Failed to load input stereo images!" << endl;
        return -1;
    }

    // Convert to grayscale for rectification
    Mat leftImage, rightImage;
    cvtColor(leftImageColor, leftImage, COLOR_BGR2GRAY);
    cvtColor(rightImageColor, rightImage, COLOR_BGR2GRAY);

    // Camera intrinsics
    Mat cameraMatrix1 = (Mat_<double>(3,3) << 578.86, 0, 329.12,
                                              0, 772.98, 245,
                                              0, 0, 1);

    Mat cameraMatrix2 = (Mat_<double>(3,3) << 576.05, 0, 303.22,
                                              0, 768.94, 252.27,
                                              0, 0, 1);

    // Distortion coefficients
    Mat distCoeffs1 = (Mat_<double>(1,5) << -0.166, 1.16, 0.00, 0.00, -2.54);
    Mat distCoeffs2 = (Mat_<double>(1,5) << -0.179, 1.49, 0.00, 0.00, -3.89);

    // Rotation and translation between cameras
    Mat R = (Mat_<double>(3,3) << 0.9998996653444937, -0.007885144718599264, -0.01176791131579057,
                                  0.007832711413248565, 0.9999592206338627, -0.004495075220148332,
                                  0.0118028757464582, 0.004402449555051375, 0.9999206521329724);

    Mat T = (Mat_<double>(3,1) << -58.96213584809675,
                                  -0.3678605528979815,
                                  -0.6298580023378643);

    // Rectification results
    Mat R1, R2, P1, P2, Q;
    stereoRectify(cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, leftImage.size(),
                  R, T, R1, R2, P1, P2, Q, CALIB_ZERO_DISPARITY, -1, leftImage.size());

    // Undistort and rectify maps
    Mat map1x, map1y, map2x, map2y;
    initUndistortRectifyMap(cameraMatrix1, distCoeffs1, R1, P1, leftImage.size(), CV_32FC1, map1x, map1y);
    initUndistortRectifyMap(cameraMatrix2, distCoeffs2, R2, P2, rightImage.size(), CV_32FC1, map2x, map2y);

    // Rectify the images
    Mat rectifiedLeft, rectifiedRight;
    remap(leftImage, rectifiedLeft, map1x, map1y, INTER_LINEAR);
    remap(rightImage, rectifiedRight, map2x, map2y, INTER_LINEAR);

    // Convert grayscale rectified images to BGR before writing as .ppm
    Mat rectifiedLeftColor, rectifiedRightColor;
    cvtColor(rectifiedLeft, rectifiedLeftColor, COLOR_GRAY2BGR);
    cvtColor(rectifiedRight, rectifiedRightColor, COLOR_GRAY2BGR);

    // Save rectified images as PPM
    imwrite("left_rectified.ppm", rectifiedLeftColor);
    imwrite("right_rectified.ppm", rectifiedRightColor);
    cout << "[INFO] Saved: left_rectified.ppm and right_rectified.ppm\n";

    // Optional: Save the rectification maps (can be reused)
    FileStorage fs("lookupTables.xml", FileStorage::WRITE);
    fs << "Map1x" << map1x;
    fs << "Map1y" << map1y;
    fs << "Map2x" << map2x;
    fs << "Map2y" << map2y;
    fs.release();

    return 0;
}
