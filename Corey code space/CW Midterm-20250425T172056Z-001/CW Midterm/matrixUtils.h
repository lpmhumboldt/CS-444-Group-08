#ifndef MATRIXUTILS_H
#define MATRIXUTILS_H


using namespace std;

void matrixPrint(double* matrix, int rows, int cols);  

void matrixProduct(double* a, int rows_a, int cols_a, double* b, int rows_b, int cols_b, double* c); 

void matrixTimesVector(double* a, int rows, int cols, double* v, int rows_v, double* w);

void matrixTranspose(double* a, int rows, int cols, double* aTranspose);

void matrixExtractCol(double*a, int rows, int cols, int col, double* column);

void matrixQR(double* a, int rows, int cols, double* q, double* r);

void matrixBackSubstitution(double* R, int rows, int cols, double* d, double* p);

void matrixUpperTriangularInverse(double* A, int rows, int cols, double* invA);

void matrixInternalCameraParameters(double* p,int rows,int cols,double* k);

void matrixSVD(double* a, int rows, int cols, double* u, double* s, double* vT);

void matrixTR(double* u, int rows_u, int cols_u, double* s, int rows_s, int cols_s, double* vT, int rows_vT, int cols_vT, double* t, double* r);

void matrixRectify(double* e1, double * e2, double* e3, double* Rect);

double matrix2X2Det(double* a,int rows,int cols);

double matrix3X3Det(double* m, int rows, int cols);

void matrixScale(double* m,int rows,int cols, double scale);

#endif
