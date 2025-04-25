#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include "imageUtils.h"
#include "matrixUtils.h"
#include "utils.h"

int main() {

    int cols = 640;
    int rows = 480;
    int searchWidth = 5;
    int searchHeight = 5;
    int blurWidth = 5;
    int blurheight = 5;

    const char* leftBW = "leftBW.ppm";
    const char* rightBW = "rightBW.ppm";
    const char* depthImageName = "depth.ppm";
    const char* disparityImageName = "disparity.ppm";    

    unsigned char* depthImage = (unsigned char*) malloc(rows * cols * sizeof(unsigned char));
    unsigned char* disparityImage = (unsigned char*) malloc(rows * cols * sizeof(unsigned char));

    PPMImage* leftImg;
    PPMImage* rightImg;
    leftImg = readPPM(leftBW, 0);
    rightImg = readPPM(rightBW, 0);

    double leftAvg = 0.0;
    double rightAvg = 0.0;
    double depthAvg = 0.0;
    double sumAvg = 0.0;
    
    int maxColor = 255;
    int black = 0;
    int white = 255;

    int acceptableRange = 1;
    printf("start\n");
    //Create disparity image
    for (int row = 0; row < rows; row= row + searchWidth) {
        for (int col = 0; col <= cols - searchHeight; col = col + searchHeight) {
            int ID = col + row * cols;
            for (int i = 0; i < searchWidth; i++) {
                for (int y = 0; y < searchHeight; y++) {
                    int rangeID = ID + i + (cols * y);
                    leftAvg = leftAvg + leftImg->data[rangeID];
                    rightAvg = rightAvg + rightImg->data[rangeID];
                }

                leftAvg = leftAvg / (searchHeight * searchWidth);
                rightAvg = rightAvg / (searchHeight * searchWidth);
                sumAvg = ((leftAvg + rightAvg) / 2);

                for (int y = 0; y < searchHeight; y++) {
                    int rangeID = ID + i + (cols * y);
                    
                    if (rightAvg > 10 && leftAvg > 10 && (fabs(rightAvg - leftAvg)) < 7) {  
                        //disparityImage[rangeID] = black;
                        disparityImage[rangeID] = black + sumAvg;
                    } else {
                        //disparityImage[rangeID] = white;
                        disparityImage[rangeID] = white - sumAvg;
                    }
                }
                rightAvg = 0.0;
                leftAvg = 0.0;
                sumAvg = 0.0;
            }
        }
    }

    //create depth map
    for (int row = 0; row < rows; row= row + searchWidth) {
        for (int col = 0; col <= cols - searchHeight; col = col + searchHeight) {

            int ID = col + row * cols;
            for (int i = 0; i < searchWidth; i++) {
                for (int y = 0; y < searchHeight; y++) {
                    int rangeID = ID + i + (cols * y);
                    depthAvg = depthAvg + disparityImage[rangeID];
                }
            }

            depthAvg = depthAvg / (blurheight * blurWidth);

            for (int i = 0; i < searchWidth; i++) {
                for (int y = 0; y < searchHeight; y++) {
                    int rangeID = ID + i + (cols * y);
                    depthImage[rangeID] = depthAvg;
                }
            }
            depthAvg = 0.0;
        }
    }


    writePPM(depthImageName, cols, rows, maxColor, 0, depthImage);
    writePPM(disparityImageName, cols, rows, maxColor, 0, disparityImage);
    printf("end\n");
    return 0;
}