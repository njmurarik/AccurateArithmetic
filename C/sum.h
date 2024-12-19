#ifndef SUM
#define SUM
/* Sumk (summation in k-fold precision) */
double sumk(double* p,const unsigned int n,const unsigned int k);
/* Sumkk (summation in k-fold precision stored in k-parts) */
double* sumkk(double* p,const unsigned int n,const unsigned int k);
/* Sum MPFR (sum in multi-precision using MPFR) */
double sumMPFR(double* p,const unsigned int n,const unsigned int prec);
#endif