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

void vectorPrint(double* u , int rows)
{
  for(int i=0;i<rows;i++)
  {
    printf( "%7.1f \n",u[i]);
  }
  printf("\n");

}

void vectorScale(double* u, int rows, double alpha , double* v)
{
  for(int i=0; i<rows; i++)
  {
    v[i]=alpha*u[i];
  }
}

// added in lecture V
void matrixMult(double*a, int rows_a, int cols_a, double*b, int rows_b, int cols_b, double*c)
{
  int index = 0;
  int rows = rows_a;
  int cols = cols_b;

  if(cols_a == rows_b)
  {
    for(int row = 0; row < rows; row++)
    {
      for(int col = 0; col < cols_b; col++)
      {
        index = row*cols + col;
        c[index] = 0.0;
        for(int k = 0; k < cols_a; k++)
        {
          c[index]=c[index]+a[row*cols_a + k] *b[col+k*cols_b];
        }
      }
    }
  }
  else
  {
    printf("Can't multiply. Dimensions are incorrect\n");
    exit(0);
  }

}

// added for hw1

//---------------------------------------------------------------
void vectorDotProduct(double* a, double* b, int size, double& sum)
{
  sum = 0.0;
  for(int i = 0; i < size; i++)
  {
    sum += a[i] * b[i];
  }
}
//---------------------------------------------------------------

void vectorSubtract(double* a, int rows, double* b, double* result)
{
  for (int i = 0; i < rows; i++)
  {
    result[i] = a[i] - b[i];
  }
}
//---------------------------------------------------------------
void vectorNorm(double*a, int rows, double& norm)
{
 double sum = 0.0;
 for(int i = 0; i < rows; i++)
 {
  sum+= a[i] * a[i];
 } 
 norm = sqrt(sum);
}
//---------------------------------------------------------------
