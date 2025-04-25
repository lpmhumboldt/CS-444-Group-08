#ifndef UTILS_H
#define UTILS_H


using namespace std;

void vectorPrint(float* u , int rows); 

void vectorScale(float* u, int rows, float alpha , float* v); 

// class changes
void matrixMult(float* a, int rows_a, int cols_a, float* b, int rows_b, int cols_b, float* c);





// Hw1 Changes

void vectorDotProduct(float* a, float* b, int size, float& result);
// takes the float sum and sets to 0.0, uses a loop 
// to cycle through the size(rows) of the matrix,
// and changes sum to sum + u[i] * v[i], populating result into the answer

void vectorScaling();
// I think this is already added? under vectorScale()

void vectorSubtract(float* a, int rows, float* b, float* sum);
// loops through a and b while subtracting a[i] - b[i] equaling result[i]
// updates result[i] as it loops

void vectorNorm(float* u, int rows, float& norm);
// has float sum = 0.0, loops through rows and sets sum to sum+= a[i] * a[i],
// then sets float& norm to the square root of sum


#endif