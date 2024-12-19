#ifndef KPART
#define KPART


/* ########## Arithmetic Operations ########## */

/* kpartSum (sum of two kpart numbers) */
void sumParts(const double *x, double *y, const unsigned int k);
/* kpartAddD (add double to kpart number) */
void sumPartsD(const double xval, double *y, const unsigned int k);
/* kpartProd (product of two kpart numbers) */
void prodParts(const double *x, double *y, const unsigned int k);
/* kpartMulD (multipy kpart number by double) */
void prodPartsD(const double xval, double *y, const unsigned int k);
/* kpartFMA (fused multiply-add with kpart numbers) */
void FMAParts(const double *x, const double *y, double *z, const unsigned int k);
/*kpartFMA (fused multiply-add: {z} = {x}y + {z})*/
void FMAPartsD(const double* x, const double y, double* z, const unsigned int k);
/* kpartNeg (negative kpart number) */
void negParts(double* x,const unsigned int k);
/* kpartPrint */
void printParts(const double *x, const unsigned int k);
/* kpartSet */
void setParts(const double *x,double *y,const unsigned int k);
/* kpartSetD */
void setPartsD(double *x, const double xval, const unsigned int k);
/* kpartNewDiv */
void divParts(double* x, const double* b, const unsigned int k);
/* kpartDivD */
void divPartsD(double* x, const double b, const unsigned int k);
/* kpartSplitD*/
void splitParts(double x, double* res, int k);


/* ########## Linear Algebra Operations ########## */

/* Display a Matrix that is in K Parts */
void printMatParts(const unsigned int m, const unsigned int n, const unsigned int k, 
const double** kA, char* desc);
/* Dot Product in fl_kk from floating-point vectors */
void dotPartsD(const double* u, const double* v, double* d, const unsigned int n, const unsigned int k);
/* Dot Product in K-fold Precision and Stored in K-parts */
void dotParts(const double** u, const double** v, double b, double* d, const unsigned int n, const unsigned int k);
/* Dot Product in fl_kk where inputs are in l and m parts */
void dotPartsLM(const double** u, const double** v, double* d, const unsigned int n, 
const unsigned int l, const unsigned int m, const unsigned int k);
/* Gaussian Elimination with Partial Pivoting s.t. {x} --> A{x} = b */
void GEPPParts(const double** A, const double* b, double** x, const int* ipiv, const unsigned int n, const unsigned int k);
/* Gaussian Elimination with Partial Pivoting s.t. {x} --> A{x} = b with an intermediate variable */
void GEPPPartsO(const double** A, double** b, const int* ipiv, const unsigned int n, const unsigned int k);
/*LU Decomposition in K Parts from K Parts */
void LUParts(double** A, int* ipiv, const unsigned int n, const unsigned int k);
/* Iterative Refinement in K Parts */
void itrefParts(const double** A, const double** LU, double** x, const double* b, const int* ipiv, 
const unsigned int n, const unsigned int k, const double TOL, const unsigned int max_count);

#endif