#include "eft.h"
#include "sum.h"
#include <math.h>
#include <mpfr.h>
#include <stdlib.h>
/* Sumk (summation in k-fold precision rounded to the working precision) */
double sumk(double* p,const unsigned int n,const unsigned int k){
	for(unsigned int i=1; i<k; ++i){
		vec_sum(p,n);
	}
	double res = p[0];
	for(unsigned int i=1; i<n; ++i){
		res += p[i];
	}
	return res;
}
/* Sumkk (summation in k-fold precision stored in k-parts) */
double* sumkk(double* p,const unsigned int n,const unsigned int k){
	double* res = (double*)malloc(k*sizeof(double));
	for(unsigned int i=0; i<k-1; ++i){
		vec_sum(p,n-i);
		res[i] = p[n-i-1];
	}
	res[k-1] = p[0];
	for(unsigned int i=1; i<n-k+1; ++i){
		res[k-1] += p[i];
	}
	return res;
}
/* Sum MPFR (sum in multi-precision using MPFR) */
double sumMPFR(double* p,const unsigned int n,const unsigned int prec){
	mpfr_t x[n];
	mpfr_ptr xp[n];
	for(unsigned int i=0; i<n; ++i){
		mpfr_init2(x[i],prec);
		xp[i] = x[i];
	}
	
	for(unsigned int i=0; i<n; ++i){
		mpfr_set_d(x[i],p[i],MPFR_RNDN);
	}
	mpfr_t sum;
	mpfr_init2(sum,prec);
	mpfr_sum(sum,xp,n,MPFR_RNDN);
	double res = mpfr_get_d(sum,MPFR_RNDN);
	
	mpfr_clear(sum);
	for(unsigned int i=0; i<n; ++i){
		mpfr_clear(x[i]);
	}
	return res;
}