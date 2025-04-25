#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <ctime>
#include <string>
#include <sstream>
#include <lapacke.h>
#include "utils.h"
#include "matrixUtils.h"

using namespace std;

void matrixPrint(double* matrix, int rows, int cols) {
    printf("number of rows and cols,%d %d \n", rows, cols);
    for (int row = 0; row < rows; row++) {

        for (int col = 0; col < cols; col++) {
            printf("%1.3e ",matrix[row*cols + col]);
        }
        printf("\n");

    }
    printf("\n");
}

void matrixProduct(double* a, int rows_a, int cols_a, double* b, int rows_b, int cols_b, double* c){
 int index = 0;
 int rows  = rows_a;
 int cols  = cols_b;

 if(cols_a == rows_b){
 for(int row = 0; row < rows; row++){
   for(int col = 0; col < cols; col++){
       index = row*cols + col;
       c[index] = 0.0;
       for(int k = 0; k < cols_a; k++){
         c[index]=c[index]+a[row*cols_a + k]*b[col+k*cols_b];
       }
   }

 }

 }else{
	 printf("Can't multiply. Dimensions incorrect \n");
	 exit(0);
 }

}

void matrixTimesVector(double* a, int rows, int cols, double* v, int rows_v, double* w){
if(cols==rows_v){
   for( int row = 0; row < rows; row++){
                w[row]=0.0;
		for( int col = 0;col < cols;col++){
                   w[row]=w[row]+a[row*cols+col]*v[col];
		}
   }
}else{
	printf(" dimensions don't match \n");
	exit(0);
}

}

void matrixTranspose(double* a, int rows, int cols, double* aTranspose){
 for( int row = 0; row < rows; row++){
		for( int col = 0;col < cols;col++){

		    aTranspose[col*rows+row] = a[row*cols+col];
		}

 }
}

void matrixExtractCol(double*a, int rows, int cols, int col, double* column){
    int index;
    for(int row=0;row<rows;row++){
	index = col + row*cols;
        column[row] = a[index];
    }
}


void matrixBackSubstitution(double* R, int rows, int cols, double* d, double* p){
//solve Rp=d for an uppertriangular matrix R
// using back substitution

  for (int j = cols-1; j > -1; j--){
        p[j]=d[j];
	for(int k = j+1; k < cols; k++){
		p[j]= p[j] - R[j*cols + k]*p[k];
	}
	if(R[j*cols+j] != 0.0){
	       p[j]=p[j]/R[j*cols+j];
	}else{
		printf("Backsubstitution failed \n");
		exit(0);
	}
  }

}

void matrixUpperTriangularInverse(double* A, int rows, int cols, double* invA){
// This code computes the inverse of a non-singular upper triangular square matrix.
  if(rows != cols){
	printf(" Only square full rank upper triangle matricies can be inverted with this code\n");
       	exit(0);
  }

  for (int row=0; row<rows; row++){
    //check for singularity
    if(A[row*cols+row] == 0.0){
       printf("A must be non-singular \n");
       exit(0);
     }
  }

// initialize the inverse to zero
 for(int row = 0; row < rows; row++){
  for(int col= 0;col < cols; col++){
    invA[ row*cols + col]  = 0.0;
  }
 }

// compute the rows of the inverse matirx starting with the last row
for (int row = rows - 1; row > -1; row--){

 //compute only the diagonal and upper triangular elements of each row. all others are zero.
 for(int col = row; col < cols; col++){

  // compute the diagonal element
  if(row == col){

      invA[row*cols+col] = 1.0/A[row*cols+row];

 }else{
	  // Compute the above diagonal elements
	  // col-row terms are needed to compute each element
    for(int k = 0;k < col-row; k++){

             invA[row*cols+col] = invA[row*cols+col]-
	                         A[row*cols+(col-k)]*invA[(col-k)*cols+col]/A[row*cols+row];
     }
 }

 }
}

}

void matrixInternalCameraParameters(double* p,int rows,int cols,double* k){

double pSub[9];
double r[9];
double rInv[9];
double q[9];
double qT[9];
double b[9];

// extract a 3x3 sub matrix of p
for (int row=0; row < rows; row++){
	for (int col=0; col<cols-1;col++){
            pSub[row*(cols-1)+col]=p[row*cols+col];
	}
}
printf("3x3 submatrix of p\n");
matrixPrint(pSub,3,3);

// find the inverse of pSub with QR decomposition
// Its inverse is B=RInv*QTranspose
matrixQR(pSub,3,3,q,r);
matrixTranspose(q,3,3,qT);
matrixUpperTriangularInverse(r,3,3,rInv);
matrixProduct(rInv,3,3,qT,3,3,b);

//QR decompose the matrix B
matrixQR(b,3,3,q,r);

int rows_k = rows;
int cols_k = rows_k;

int rows_r = rows;
int cols_r = rows_r;

//diagonals of calibration matirx
k[0*cols_k +0] = 1/r[0*cols_r+0];
k[1*cols_k +1] = 1/r[1*cols_r+1];
k[2*cols_k +2] = 1.0;
// off diagonals of the calibration matrix
k[0*cols_k +2] = -r[0*cols_r +2]/r[0*cols_r+0];
k[1*cols_k +2] = -r[1*cols_r +2]/r[1*cols_r+1];

}

void matrixSVD(double* A, int rows, int cols, double* U, double* S, double* VT){

    int m = rows; // rows of A
    int n = cols;
    int lda = rows; //leading dimension of matrix A
    int ldu = rows; //leading dimension of U
    int ldvt = cols; //leading dimension of VT
    int min = rows;
    double s[rows];

    if(cols<rows) min = cols;
    double superb[min-1]; // super-diagonal matrix elements

    // 'A' = compute full matrix U
    // 'A' = compute full matrix V
    // Compute SVD
    int info = LAPACKE_dgesvd(LAPACK_ROW_MAJOR, 'A', 'A', m, n, A, lda, s, U, ldu, VT, ldvt, superb);
// creeate the S matrix
   S[0*cols+0]=s[0];
   S[1*cols+1]=s[1];
   S[2*cols+2]=s[2];

    // Check for success
    if (info > 0) {
        printf("SVD failed to converge.\n");
        exit(0);
    }


}

void matrixTR(double* u, int rows_u, int cols_u, double* s, int rows_s,int cols_s, double* vT, int rows_vT, int cols_vT, double* t, double* r){
double w[9]={0.0, -1.0, 0.0,
	     1.0,  0.0, 0.0,
	     0.0,  0.0, 1.0};
double wT[9];
double tmp1[9];
double tmp2[9];
double tx[9];
double uT[9];


//extract t according to wikipedia t=u*w*s*uT
matrixTranspose(u,3,3,uT);
matrixProduct(u,3,3,w,3,3,tmp1);
matrixProduct(s,3,3,uT,3,3,tmp2);
matrixProduct(tmp1,3,3,tmp2,3,3,tx);
printf("tx matrix \n");
matrixPrint(tx,3,3);

t[0]=tx[2*3+1];
t[1]=-tx[2*3+0];
t[2]=tx[1*3+0];

//extract R according to wikipedia R=u*wT*vT (Since wT = wInv)
matrixTranspose(w,3,3,wT);
matrixProduct(u,3,3,wT,3,3,tmp1);
matrixProduct(tmp1,3,3,vT,3,3,r);

//must check which has +1 det !!!!!! There are 4 possible solutions
// +/- t and R with det(R)=+/-1 Only the correct one keeps all image
// points in from of the cameras.

}
void matrix2X2Inverse(double* a, int rows, int cols, double* aInv){
double det;
det = a[0*cols+0]*a[1*cols+1]-a[0*cols+1]*a[1*cols+0];
if(det == 0.0){
	printf("singular matrix in 2x2 Inverse \n");
	exit(0);
}

aInv[0*cols+0]=a[1*cols+1]/det;
aInv[1*cols+1]=a[0*cols+0]/det;
aInv[0*cols+1]=-a[1*cols+0]/det;
aInv[1*cols+0]=-a[0*cols+1]/det;

}

void matrixRectify(double* e1, double* e2, double* e3, double* Rect){
int cols=3;
	Rect[0*cols+0]=e1[0];
	Rect[0*cols+1]=e1[1];
	Rect[0*cols+2]=e1[2];

	Rect[1*cols+0]=e2[0];
	Rect[1*cols+1]=e2[1];
	Rect[1*cols+2]=e2[2];

	Rect[2*cols+0]=e3[0];
	Rect[2*cols+1]=e3[1];
	Rect[2*cols+2]=e3[2];
}
