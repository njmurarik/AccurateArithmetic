#ifndef GENTEST
#include <mpc.h>
#include <mpfr.h>
#define GENTEST


/* ########## Auxillary Functions ########## */

/* Make a permutation of two vectors vector using a permutation rule */
void permutate(double* v, double* u, int n);
/* Applies abs() to each entry of a specified vector */
void abs_vec(double* vec, double* abs_vec, int n);
/* Generates a double in the interval [a, b] */
double double_rand(double a, double b);
/* Returns value of Binomial Coefficient C(n, k) - stolen from the web*/
double binomialCoeff(int n, int k);
/* Returns LCM of array elements - stolen from the web */
int findlcm(int n);
/* Splits a double into k-parts by manipulating the significand */
void double_split(double x, double* res, int k);
/* Dot Product in Double Precision */
void dotDP(const double* u, const double* v, double* dot, const unsigned int n);
/* Dot Product in MPFR Precision */
void dotMPFR(const double* u, const double* v, mpfr_t* dot, int size, long prec);
/* Matrix Product with MPFR Precision */
void matmultMPFR(const double* A, const double* B, double* C, int n, long prec);
/* Vector Inf Norm */
void vecInfNormMPFR(const mpfr_t* x, const unsigned int n, mpfr_t* norm);
/* Vector Inf Norm for Double Input */
void vecInfNormMPFRD(const double* x, const unsigned int n, mpfr_t* norm, const unsigned long prec);
/* linear solve (solve linear system in multi-precision using MPFR) */
void linearSolveMPFR(const double* a,const double* b, mpfr_t* x, const unsigned int n,const unsigned int prec);
/* Estimates the Condition Number of a Matrix with MPFR */
void condMPFR(const double* A, const unsigned int n, double* cond, const unsigned long prec);

/* ########## Test Generation Functions ########## */

/* Creates two vectors with a high dot product condition number */
void gendot(double* v, double* u, int vec_size, double* cond);
/* Generates two vectors, both in k-parts, with a high dot product condition number */
void gendotLM(double* v, double* u, double** vk, double** uk, int vec_size, int l, int m, double* cond);
/* Generates a Hilbert Matrix */
void hilbmatrix(double** A, const unsigned int n, const unsigned int k, double* cond);
/* Generates a Random Matrix using a Singular Value Decomposition based on the input condition number */
void svdmatrix(double* A, const unsigned int n, double* cond);

#endif