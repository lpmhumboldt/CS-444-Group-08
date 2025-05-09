#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <vector>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include "stereoDepth.h"
#include <time.h>

using namespace cv;
using namespace std;

int main(){

	int rows = 480;
	int cols = 640;
	int maxDisparity;

	// just used for debugging
	// clock_t start,end;
	// double time_used;

	Mat left = imread("left", IMREAD_GRAYSCALE);
	Mat right = imread("right", IMREAD_GRAYSCALE);
	Mat depth = Mat::zeros(rows, cols, CV_8UCI);

	// this is going to read 
	// the rectification lookup tables
	Mat map1x,maply,map2x,map2y;
	
	FileStorage fs("lookupTables.xml", FileStorage::READ);
	fs["Map1x"]>>map1x;
	fs["Map1y"]>>map1y;
	fs["Map2x"]>>map2x;
	fs["Map2y"]>>map2y;
	fs.release();

	float offset = 0.0;
	float currentRow;
	for(int row = 0; row < rows; row++){
		for(int col = 0; col < cols; col++){
			currentRow = map2y.at<float>(row,col);
			if(currentRow+offset < 0 || 
			currentRow+offset > rows){
				map2y.at<float>(row,col) = currentRow;
			}else{
				map2y.at<float>(row,col) = currentRow + offset;
			}
		}
	}

	Mat rectifiedLeft, rectifiedRight, both;
	remap(left, rectifiedLeft, map1x, map1y, INTER_LINEAR);
	remap(right ,rectifiedRight, map2x, map2y, INTER_LINEAR);

	// start = clock();
	stereoDepth(&rectifiedLeft, &rectifiedRight, &depth, maxDisparity, rows, cols);
	// end = clock();
}