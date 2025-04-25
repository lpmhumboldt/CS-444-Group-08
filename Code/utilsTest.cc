#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <ctime>
#include <string>
#include <sstream>
#include "utils.h"

using namespace std;

int main(){

float u[] = 
	    {1.0,
	     2.0,
	     3.0};

float v[] = {
             4.0,
             5.0,
             6.0}; 
float a[] = 
            {2, 1,
             1, 2};

int rows_a = 2;
int cols_a = 2;

float b[] = 
            {1,2,3,4,
            5,6,7,8};

int rows_b = 2;
int cols_b = 4;

// --------------------
float c[] = 
            {0,0,0,0,
             0,0,0,0};
int rows_c = 2;
int cols_c = 4;

// --------------------

int rows = 3;
float alpha = 2.0;

// --------------------

float result[rows];
float norm = 0.0;
float dotProductResult = 0.0;

// --------------------
cout << "This should print the vector u, " << endl;
vectorPrint(u, rows);

cout << "This should print the vector v, " << endl;
vectorPrint(v,rows);
// prints rows

//vectorScale(u,rows,alpha,v);
// commented out to show functionallity of vectorSubtract,
// using original vectors

matrixMult(a,rows_a,cols_a,b,rows_b,cols_b,c);
// added in lecture

// --------------------
// added for hw1

cout<< "This will print vectorDotProduct(), " << endl;
cout<< "(the answer should be 32.0) " << endl;
vectorDotProduct(u, v, rows, dotProductResult);
vectorPrint(&dotProductResult, 1);
// should print out 32.0

cout<< "This will vectorSubtract(), " << endl;
cout<< "Should print -3.0, -3.0, -3.0 " << endl;
vectorSubtract(u, rows, v, result);
vectorPrint(result,rows);
// should be [-3.0, -3.0, -3.0]

cout << "This will assign vector normArray[] and it should print 3.7" << endl;
vectorNorm(u, rows, norm);
float normArray[] = {norm};
vectorPrint(normArray, 1);
// should return 3.7



return 0;
}