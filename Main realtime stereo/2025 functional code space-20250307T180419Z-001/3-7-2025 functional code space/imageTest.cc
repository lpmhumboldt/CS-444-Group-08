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

    const char* leftBW = "leftBW.ppm";
    const char* depthImageName = "depth.ppm";

    unsigned char* depthImage = (unsigned char*) malloc(rows * cols * sizeof(unsigned char));

    PPMImage* leftImg;
    leftImg = readPPM(leftBW, 0);

    double avg = 0.0;
    int maxColor = 255;

    printf("start\n");
    for (int row = 0; row < rows; row= row + searchWidth) {
        for (int col = 0; col <= cols - searchHeight; col = col + searchHeight) {

            printf("row: %i\n", row);
            printf("rows: %i\n", rows);
            printf("col: %i\n", col);
            printf("cols: %i\n", cols);
            int ID = col + row * cols;
            printf("ID: %i\n", ID);
                printf("==============================\n");
            for (int i = 0; i < searchWidth; i++) {

                for (int y = 0; y < searchHeight; y++) {

                    int rangeID = ID + i + (cols * y);


                    if (rangeID <= 10) {
                        printf("00%i ",rangeID);
                    }else if (rangeID < 100 && rangeID > 10) {
                        printf("0%i ", rangeID);
                    } else {
                        printf("%i ", rangeID);
                    }  

                    avg = avg + leftImg->data[rangeID];

                }

                avg = avg / searchHeight * searchWidth;

                for (int y = 0; y < searchHeight; y++) {

                    int rangeID = ID + i + (cols * y);


                    if (rangeID <= 10) {
                        printf("depth 00%i ",rangeID);
                    }else if (rangeID < 100 && rangeID > 10) {
                        printf("depth 0%i ", rangeID);
                    } else {
                        printf("depth %i ", rangeID);
                    }  
                    
                    depthImage[rangeID] = avg;


                }

                avg = 0.0;

            }
            printf("\n");
        }
        printf("***********************************************************************************************************************=\n");
    }

    writePPM(depthImageName, cols, rows, maxColor, 0, depthImage);
    printf("end");
    return 0;
}