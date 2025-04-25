#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <ctime>
#include <string>
#include <sstream>

using namespace std;

void vectorPrint(double* u , int rows){
  for(int i=0;i<rows;i++){
    printf( "%7.1f \n",u[i]);
  }
  printf("\n");
}

void vectorScale(double* u, int rows, double alpha , double* v){
  for(int i=0; i<rows; i++){
    v[i]=alpha*u[i];
  }
}

double vectorDotProduct(double* vectA, int sizeOfA, double* vectB, int sizeOfB) {
  double sum = 0.0;
  double returnedValue;

  if (sizeOfA == sizeOfB) {
    for (int i = 0; i < sizeOfA; i++) {
      sum = sum + vectA[i] * vectB[i];
    }
    returnedValue = sum;
  } else {
    printf("Can't find dot product. Size of vectors don't match.\n");
  }

  return returnedValue;
}
//vectorSubtract(numerator,rows,projection,rows,numerator);
void vectorSubtract(double* vectA, int sizeOfA, double* vectB, int sizeOfB, double* returnedValue) {
  if (sizeOfA == sizeOfB) {
    for (int i = 0; i < sizeOfA; i++) {
      returnedValue[i] = vectA[i] - vectB[i];
    }
  } else {
    printf("Can't find result of vector A minus vector B. Size of vector A, B, or return vector don't match.\n");
  }
}

double vectorNorm(double* vect, int sizeOfVect) {
  double sum = 0.0;
  double returnedValue;
  for (int i = 0;i < sizeOfVect; i++){
    sum = sum + vect[i] * vect[i];
  }
  returnedValue = sqrt(sum);

  return returnedValue;
}