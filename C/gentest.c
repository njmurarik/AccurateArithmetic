#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <math.h>
#include <time.h>
#include <mpfr.h>
#include <mpc.h>
#include <lapacke.h>
#include <lapack.h>
#include <cblas.h>
#include "gentest.h"
#include "kpart.h"
#include "sum.h"

/* ########## Auxillary Functions ########## */

/* Make a permutation of two vectors vector using a permutation rule */
void permutate(double* v, double* u, int n){
	
	int perm[n];
	
	// Make the permutation
	for(int i = 0; i < n; i++){
		int cont;
		do {
			cont = 0;
			perm[i] = rand() % n;
			for(int j = 0; j < i; j++){
				if(perm[j] == perm[i]) { cont = 1; break; }
			}
			if(cont == 0) break;
		}while(1 == 1);
	}
	
	// Apply the same permutation to v and u
	for(int i = 0; i < n; i++){
		int perm_loc = perm[i];
		double vtemp = v[perm_loc];
		double utemp = u[perm_loc];
		
		v[perm_loc] = v[i];
		v[i] = vtemp;
		
		u[perm_loc] = u[i];
		u[i] = utemp;
	}
	
}
/* Applies abs() to each entry of a specified vector */
void abs_vec(double* vec, double* abs_vec, int n){
	for(int i = 0; i < n; i++) abs_vec[i] = fabs(vec[i]);
}
/* Generates a double in the interval [a, b] */
double double_rand(double a, double b){
	return (double)(b - a)*((double)rand() / RAND_MAX) + a;
}
/* Splits a double into k-parts by manipulating the significand */
void double_split(double x, double* res, int k){
	
	// If the user asks for the double to be represented as 1 part
	if(k == 1){
		res[0] = x;
		return;
	}
	// If the user asks for the double to be represented as 2 parts
	else if(k == 2){
		res[0] = x / 2;
		res[1] = x / 2;
		return;
	}
	
    // Defines a double and its C representation in bytes
    typedef union {
      double val;
      unsigned char c[sizeof(double)];
    } PART;

    // Defines the byte and bit location on a Double
    typedef struct {
      int byte;
      int bit;
    } BoI;

    // Assign the k - 1 parts
    PART parts[k - 1];
    for (int i = 0; i < k - 1; i++) parts[i].val = x;

    // Determine the exponent of x
    int exp = (parts[0].c[7] & ~0x80);
    exp <<= 4;
    exp |= (parts[0].c[6] & (0x0F << 4)) >> 4;
    exp -= 1023;

    // Now to find the bits of interest
    BoI marks[k];

    // Evenly mark different bits on the double for bitmasking
    int unit = 52 / (k - 1);
    for (int m = 0; m < k; m++) {
      int mark;
      if (m == k - 1) mark = 52;
      else mark = m * unit;
      marks[m].byte = mark / 8;
      marks[m].bit = mark % 8;
    }

    /*
      Remove bits where information is not wanted and keep bits where
      information is wanted. This part makes the k - 1 doubles which
      sum to input +/- pow(2, exp).
    */
    for (int i = 0; i < k - 1; i++) {

      BoI start = marks[i];
      BoI end;
	  
	  // Edit the marks so that our bitmask gets the wanted bits
      if (marks[i + 1].bit - 1 < 0) {
        end.byte = marks[i + 1].byte - 1;
        end.bit = 7;
      } 
	  else {
        end.byte = marks[i + 1].byte;
        end.bit = marks[i + 1].bit - 1;
      }
	  
      for (int j = 0; j < 7; j++) {
		  
		// This is the byte where both the exponent and significand reside
        if(j == 6){
			
		  // If outside the byte interval made by the start and end mark
          if((j < start.byte) || (j > end.byte)) 
            parts[i].c[j] &= (0x0F << 4);
		  
		  // If in the start of the byte interval
          else if(j == start.byte){
            int exp_part = 0xF0 & parts[i].c[j];
            parts[i].c[j] &= (0xFF << start.bit);
            parts[i].c[j] &= (0xFF >> (8 - (end.bit + 1)));
            parts[i].c[j] |= exp_part; // Save the exponent
          }
		  
		  // If in the end of the byte interval
          else if(j == end.byte){
            int exp_part = 0xF0 & parts[i].c[j];
            parts[i].c[j] &= (0xFFF >> (8 - (end.bit + 1)));
            parts[i].c[j] |= exp_part; // Save the exponent
          }
        }
		
		// Every byte in which the significand resides
        else {
			
		  // If outside the byte interval made by the start and end mark
          if ((j < start.byte) || (j > end.byte)) parts[i].c[j] &= 0;
          
		  // If in the start byte which is the same as the end byte
		  else if((j == start.byte) && (j == end.byte)){
            parts[i].c[j] &= (0xFF << start.bit);
            parts[i].c[j] &= (0xFF >> (8 - (end.bit + 1)));
          }
		  
		  // If in the start of the byte interval
          else if(j == start.byte) parts[i].c[j] &= (0xFF << start.bit);
		  
		  // If in the end of the byte interval
          else if(j == end.byte) parts[i].c[j] &= (0xFF >> (8 - (end.bit + 1)));
        }
      }
      res[i] = parts[i].val;
    }
	
    // Add the correction term into res
    if (x > 0) res[k - 1] = (2 - k) * pow(2, exp);
    else res[k - 1] = (k - 2) * pow(2, exp);
}
/* Dot Product in Double Precision */
void dotDP(const double* u, const double* v, double* d, const unsigned int n){
	*d = 0;
	for(int i = 0; i < n; i++) *d += v[i] * u[i];
}
/* Dot Product with MPFR Precision */
void dotMPFR(const double* u, const double* v, mpfr_t* dot, int size, long prec){
	
	mpfr_init2(*dot, prec);
	mpfr_set_d(*dot, 0.00, MPFR_RNDN);
	
	mpfr_t v_mp; mpfr_init2(v_mp, prec);
	mpfr_t u_mp; mpfr_init2(u_mp, prec);
	mpfr_t mult_mp; mpfr_init2(mult_mp, prec);
	
	for(int i = 0; i < size; i++){
		mpfr_set_d(v_mp, v[i], MPFR_RNDN);
		mpfr_set_d(u_mp, u[i], MPFR_RNDN);
		mpfr_mul(mult_mp, v_mp, u_mp, MPFR_RNDN);
		mpfr_add(*dot, *dot, mult_mp, MPFR_RNDN);
	}
	mpfr_clears(v_mp, u_mp, mult_mp, NULL);
}
/* Matrix Product with MPFR Precision */
void matmultMPFR(const double* A, const double* B, double* C, int n, long prec){

	mpfr_t sum; mpfr_init2(sum, prec);
	mpfr_t mult; mpfr_init2(mult, prec);

	for(int i = 0; i < n; i++){
		for(int j = 0; j < n; j++){
			mpfr_set_d(sum, 0.00, MPFR_RNDN);
			for(int k = 0; k < n; k++){
				mpfr_set_d(mult, A[k*n + i], MPFR_RNDN);
				mpfr_mul_d(mult, mult, B[j*n + k], MPFR_RNDN);
				mpfr_add(sum, sum, mult, MPFR_RNDN);
			}
			C[j*n + i] = mpfr_get_d(sum, MPFR_RNDN);
		}
	}
	mpfr_clear(sum);
	mpfr_clear(mult);
}
/* Matrix Inf Norm */
void matInfNormMPFR(const double* A, mpfr_t* norm, const unsigned int n, const unsigned long prec){
	mpfr_t sum;
	mpfr_init2(sum, prec);
	mpfr_set_d(*norm, 0.00, MPFR_RNDN);

	for(int i = 0; i < n; i++){
		mpfr_set_d(sum, 0.00, MPFR_RNDN);
		for(int j = 0; j < n; j++){
			mpfr_add_d(sum, sum, fabs(A[j*n + i]), MPFR_RNDN);
		}
		if(mpfr_cmp_abs(sum, *norm) > 0) mpfr_set(*norm, sum, MPFR_RNDN);
	}
}
/* Vector Inf Norm */
void vecInfNormMPFR(const mpfr_t* x, const unsigned int n, mpfr_t* norm){
	mpfr_set(*norm, x[0], MPFR_RNDN); 
	for(int i = 1; i < n; i++){
		if(mpfr_cmpabs(x[i], *norm) > 0) mpfr_set(*norm, x[i], MPFR_RNDN);
	}
	mpfr_abs(*norm, *norm, MPFR_RNDN);
}
/* Vector Inf Norm for Double Input */
void vecInfNormMPFRD(const double* x, const unsigned int n, mpfr_t* norm, const unsigned long prec){
	mpfr_t xp[n];
	for(int i = 0; i < n; i++) {
		mpfr_init2(xp[i], prec);
		mpfr_set_d(xp[i], x[i], MPFR_RNDN); 
	}
	vecInfNormMPFR(xp, n, norm);
	for(int i = 0; i < n; i++) mpfr_clear(xp[i]);
}
/* linear solve (solve linear system in multi-precision using MPFR) */
void linearSolveMPFR(const double* a,const double* b, mpfr_t* x, const unsigned int n,const unsigned int prec){
	// store mpfr variables
    mpfr_t ap[n*n], bp[n];
    for(unsigned int i=0; i<n; ++i){
        mpfr_init2(bp[i],prec);
        mpfr_set_d(bp[i],b[i],MPFR_RNDN);
        for(unsigned int j=0; j<n; ++j){
            mpfr_init2(ap[n*j + i],prec);
            mpfr_set_d(ap[n*j + i],a[n*j + i],MPFR_RNDN);
        }
    }
    // Gaussian Elimination (with partial pivoting)
    mpfr_t lp;
    mpfr_init2(lp,prec);
    for(unsigned int j=0; j<n-1; ++j){
        // select pivot in jth column
        unsigned int m = j; 
        for(unsigned int i=j+1; i<n; ++i){
            if(mpfr_cmpabs(ap[n*j + i],ap[n*j + m]) > 0){
                m = i;
            }
        }
        // swap m and j rows (over columns j to n-1)
        if(m > j){
            for(unsigned int k=j; k<n; ++k){
                mpfr_swap(ap[n*k + m],ap[n*k + j]);
            }
            mpfr_swap(bp[m],bp[j]);
        }
        // apply row operations
        for(unsigned int i=j+1; i<n; ++i){
            mpfr_div(lp,ap[n*j + i],ap[n*j + j],MPFR_RNDN);
			mpfr_neg(lp,lp,MPFR_RNDN);											// lp = -a_{ij}/a_{jj}
			mpfr_set_zero(ap[n*j + i],1);
            for(unsigned int k=j+1; k<n; ++k){
                mpfr_fma(ap[n*k + i],lp,ap[n*k + j],ap[n*k + i],MPFR_RNDN);		// a_{ik} = a_{ik} + lp*u_{jk}
            }
            mpfr_fma(bp[i],lp,bp[j],bp[i],MPFR_RNDN);							// b[i] = b[i] + lp*b[j]
        }
    }
    // backward substitution
    for(int i=n-1; i>=0; --i){
        mpfr_set(x[i],bp[i],MPFR_RNDN);
        for(int j=n-1; j>i; --j){
			mpfr_neg(lp,ap[n*j + i],MPFR_RNDN);									// lp = -a_{ij}
            mpfr_fma(x[i],lp,x[j],x[i],MPFR_RNDN);							// x[i] = x[i] + lp*x[j]
        }
        mpfr_div(x[i],x[i],ap[n*i + i],MPFR_RNDN);							// x[i] = x[i]/a_{ii}
    }
    // clear mpfr variables
    mpfr_clear(lp);
    for(unsigned int i=0; i<n; ++i){
        mpfr_clear(bp[i]);
        for(unsigned int j=0; j<n; ++j){
            mpfr_clear(ap[n*j + i]);
        }
    }
}
/* Estimates the Condition Number of a Matrix with MPFR */
void condMPFR(const double* A, const unsigned int n, double* cond, const unsigned long prec){
	
	// Declare and initialize intermediate operation variables
	mpfr_t x[n];
	mpfr_t xnorm, bnorm, anorm, max;
	mpfr_init2(xnorm, prec); mpfr_init2(bnorm, prec); mpfr_init2(max, prec); mpfr_init2(anorm, prec);
	mpfr_set_d(max, 0.00, MPFR_RNDN);

	// Allocate memory for random RHS to solve
	double* b = (double*)malloc(n*sizeof(double));
	for(int i = 0; i < n; i++) mpfr_init2(x[i], prec);

	// Compute an Estimate for ||A^{-1}|| which is ||x|| / ||b||
	for(int i = 0; i < 1; i++){
		for(int j = 0; j < n; j++){
			if(j == i) b[j] = 1;
			else b[j] = 0;
		}

		// Solve Ax = b so that x approx A^{-1}b
		linearSolveMPFR(A, b, x, n, prec); 

		// Compute ||x|| <= ||A^{-1}||*||b||
		vecInfNormMPFR(x, n, &xnorm); 

		// Compute ||b||
		vecInfNormMPFRD(b, n, &bnorm, prec); 

		// Compute ||x|| / ||b|| ~ ||A^{-1}||
		mpfr_div(xnorm, xnorm, bnorm, MPFR_RNDN); 

		// Use max{ ||x|| / ||b|| }
		if(mpfr_cmp_abs(xnorm, max) > 0) mpfr_set(max, xnorm, MPFR_RNDN); 
	}

	// Compute ||A||
	matInfNormMPFR(A, &anorm, n, prec);

	// Compute ||A|| * max{ ||x|| / ||b|| }
	mpfr_mul(max, max, anorm, MPFR_RNDN);
	*cond = mpfr_get_d(max, MPFR_RNDN);

	// Free allocated variables
	mpfr_clear(xnorm);
	mpfr_clear(bnorm);
	mpfr_clear(anorm);
	mpfr_clear(max);
	for(int i = 0; i < n; i++) mpfr_clear(x[i]);
	free(b);
}
// Returns value of Binomial Coefficient C(n, k)
double binomialCoeff(int n, int k)
{
    double res = 1; 
  
    // Since C(n, k) = C(n, n-k) 
    if (k > n - k) 
        k = n - k; 
  
    // Calculate value of 
    // [n * (n-1) *---* (n-k+1)] / [k * (k-1) *----* 1] 
    for (int i = 0; i < k; ++i) { 
        res *= (n - i); 
        res /= (i + 1); 
    } 
  
    return res;
}

/* ########## Test Generation Functions ########## */

/* Creates two vectors with a high dot product condition number */
void gendot(double* u, double* v, int size, double* cond) {
	
	int half_size = round(size/2);
	
	// Set v and u to be zero vectors
	memset(u, 0, size*sizeof(double));
	memset(v, 0, size*sizeof(double));
	
	double b = log2(*cond);
	
	// Compute exponents for the first half of vectors v and u
	int expo[half_size];
	for(int i = 0; i < half_size; i++) expo[i] = round(double_rand(0, 1)*(b/2));
	expo[0] = round(b/2) + 1;
	expo[half_size - 1] = 0;
	
	// First half of vectors v and u
	for(int i = 0; i < half_size; i++) {
		u[i] = (2*double_rand(0, 1) - 1)*(pow(2, expo[i]));
		v[i] = (2*double_rand(0, 1) - 1)*(pow(2, expo[i]));
	}
	
	// Create evenly spaced numbers for the  exponents for the second half of v and u
	for(int i = 0; i < size - half_size; i++) expo[i] = round((b/2) - i*(b/(2*(size - half_size - 1))));
	
	// Second half of vectors v and u
	mpfr_t* dot_MPFR = (mpfr_t*)malloc(sizeof(mpfr_t));
	for(int i = half_size; i < size; i++){
		u[i] = (2*double_rand(0, 1)-1)*(pow(2, expo[i - half_size]));
		dotMPFR(u, v, dot_MPFR, size, 32768);
		v[i] = ((2*double_rand(0, 1)-1)*(pow(2, expo[i - half_size])) - mpfr_get_d(*dot_MPFR, MPFR_RNDN)) / u[i];
	}
	free(dot_MPFR);
	
	// Permutate v and u for randomization
	permutate(u, v, size);
	
	// Compute the actual condition number of v and u
	mpfr_t* res_MPFR = (mpfr_t*)malloc(sizeof(mpfr_t));
	dotMPFR(u, v, res_MPFR, size, 32768);
	double res_MPFR_d = mpfr_get_d(*res_MPFR, MPFR_RNDN);
	
	double abs_v[size];
	double abs_u[size];
	abs_vec(u, (double*)&abs_u, size);
	abs_vec(v, (double*)&abs_v, size);
	
	double abs_dot;
	dotDP((double*)&abs_u, (double*)&abs_v, &abs_dot, size);
	
	*cond = (2*(abs_dot))/fabs(res_MPFR_d);
	free(res_MPFR);
}
/* Generates two vectors, one in l parts and the other in m parts, with a high dot product condition number */
void gendotLM(double* u, double* v, double** uk, double** vk, int n, int l, int m, double* cond){
	
	gendot(u, v, n, cond);
	for(int i = 0; i < n; i++){
		double_split(u[i], uk[i], l);
		double_split(v[i], vk[i], m);
	}
}
/* Generates a k-part Hilbert Matrix */
void hilbmatrix(double** A, const unsigned int n, const unsigned int k, double* cond){
	int ipiv[n], info;
	double norm;
	double* a = (double*)malloc(n*n*sizeof(double));
	double* b = (double*)malloc(n*sizeof(double));
	double sgn, scale, coeff1, coeff2, coeff3;

	do {
		for(int i = 0; i < n; i++){
			for(int j = 0; j < n; j++){
				setPartsD(b, 0, k);
				setPartsD(A[j*n + i], 1.0 / (i + j + 1), k);
				divPartsD(A[j*n + i], i + j + 1, k);
				setParts(A[j*n + i], b, k);
				a[j*n + i] = sumk(b, n, k);
			}
		}
		
		norm = LAPACKE_dlange(LAPACK_COL_MAJOR, 'I', n, n, a, n);
		info = LAPACKE_dgetrf(LAPACK_COL_MAJOR, n, n, a, n, ipiv);
		if(info < 0) printf("The %dth argument had an illegal value.\n", -info);
		else if(info > 0) printf("U(%d, %d) is exactly zero, so U is a singular matrix. Generating a new matrix... \n", info, info);
	}while(info != 0);

	condMPFR(a, n, cond, 32768);
	free(a);
	free(b);
}
/* Generates a Random Matrix using a Singular Value Decomposition based on the input condition number */
void svdmatrix(double* A, const unsigned int n, double* cond){
	
    int ipiv[n], info;
	double Utau[n], Vtau[n], norm, srand, urand, vrand;

	double* a = (double*)malloc(n*n*sizeof(double));
	double* U = (double*)malloc(n*n*sizeof(double));
	double* V = (double*)malloc(n*n*sizeof(double));
	double* C = (double*)malloc(n*n*sizeof(double));
	do{
		for(int i = 0; i < n; i++){
			for(int j = 0; j < n; j++){

				do{
					urand = double_rand(-1.0, 1.0);
					vrand = double_rand(-1.0, 1.0);
					srand = pow(urand, 2) + pow(vrand, 2);
				} while(srand >= 1);

				U[j*n + i] = urand*sqrt(-2.0*log(srand) / srand);
				V[j*n + i] = vrand*sqrt(-2.0*log(srand) / srand);
				
				if((j == i) && (j == n - 1)) A[j*n + i] = 1.0 / *cond;
				else if (j == i) A[j*n + i] = 1.0; 
				else A[j*n + i] = 0.00;
			}
		}

		info = LAPACKE_dgeqrfp(LAPACK_COL_MAJOR, n, n, U, n, Utau);
		if(info < 0) printf("The %dth argument had an illegal value for Utau.\n", -info);
		info = LAPACKE_dgeqrfp(LAPACK_COL_MAJOR, n, n, V, n, Vtau);
		if(info < 0) printf("The %dth argument had an illegal value for Vtau.\n", -info);
		
		info = LAPACKE_dormqr(LAPACK_COL_MAJOR, 'L', 'N', n, n, n, U, n, Utau, A, n);
		if(info < 0) printf("The %dth argument had an illegal value for Utau.\n", -info);

		info = LAPACKE_dormqr(LAPACK_COL_MAJOR, 'R', 'T', n, n, n, V, n, Vtau, A, n);
		if(info < 0) printf("The %dth argument had an illegal value for Vtau.\n", -info);

		for(int i = 0; i < n; i++){ for(int j = 0; j < n; j++) { a[j*n + i] = A[j*n + i]; } }

		norm = LAPACKE_dlange(LAPACK_COL_MAJOR, 'I', n, n, A, n);	
		
		info = LAPACKE_dgetrf(LAPACK_COL_MAJOR, n, n, a, n, ipiv);
		if(info < 0) printf("The %dth argument had an illegal value.\n", -info);
		else if(info > 0) printf("U(%d, %d) is exactly zero, so U is a singular matrix. Generating a new matrix... \n", info, info);

	}while(info != 0);

	condMPFR(A, n, cond, 32768);
	
	free(a);
	free(U);
	free(C);
	free(V);
}