#ifndef UTILS_H
#define UTILS_H


using namespace std;

void vectorPrint(double* u , int rows); 

void vectorScale(double* u, int rows, double alpha , double* v); 
//vectorDotProduct(double*, int, double*, int)
double vectorDotProduct(double* vectA, int sizeOfA, double* vectB, int sizeOfB);

//   vectorSubtract(numerator,rows,projection,rows,numerator);
void vectorSubtract(double* vectA, int sizeOfA, double* vectB, int sizeOfB, double* returnedValue);

double vectorNorm(double* vect, int sizeOfVect);

#endif

