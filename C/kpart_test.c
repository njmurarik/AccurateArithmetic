#include "eft.h"
#include "sum.h"
#include "kpart.h"
#include "gentest.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <math.h>
#include <time.h>
#include <lapacke.h>
#include <lapack.h>
#include <cblas.h>
#include <mpfr.h>
#include <mpc.h>

#define MAX_EXPONENT 400

/* Calculuate the relative error between k-fold dot product and MPFR dot product */
double dotprod_error(mpfr_t* res_MPFR, double* res_KK, int k, int prec){
	
	// Define and Initialize MPFR sum of k-parts
	mpfr_t resKK_MPFR; mpfr_init2(resKK_MPFR, prec);
	mpfr_set_d(resKK_MPFR, 0.00, MPFR_RNDN);
	
	// Define and Initialize MPFR calculated error
	mpfr_t error; mpfr_init2(error, prec);
	mpfr_set_d(error, 0.00, MPFR_RNDN);
	
	// Sum the k-parts from dotkk
	for(int i = 0; i < k; i++) {
		mpfr_set_d(resKK_MPFR, res_KK[i], MPFR_RNDN);
		mpfr_add(error, error, resKK_MPFR, MPFR_RNDN);	
	}
	
	// Issue with rounding relative error to 0 at specific condition numbers...
	// Compute the relative error expression between dotMPFR and dotKK
	mpfr_sub(error, error, *res_MPFR, MPFR_RNDN);
	mpfr_div(error, error, *res_MPFR, MPFR_RNDN);
	return fabs(mpfr_get_d(error, MPFR_RNDN));
}
/* Compute Rel Error of solution with GEPP */
double linsolv_hilbert_error(double* x, double** xk, const unsigned int n, unsigned int k, const unsigned int prec){

    double error;
	
    // Store mpfr variables
    mpfr_t xp[n], xpk[n], aux, r;
    mpfr_init2(aux, prec); mpfr_init2(r, prec);

    for(unsigned int i = 0; i < n; i++) {
        mpfr_init2(xp[i], prec);
        mpfr_set_d(xp[i], x[i], MPFR_RNDN);
        mpfr_init2(xpk[i], prec);
        mpfr_set_d(xpk[i], 0.00, MPFR_RNDN);
        for(int j = 0; j < k; j++){
            mpfr_add_d(xpk[i], xpk[i], xk[i][j], MPFR_RNDN);
        }
    }

    // Compute the 1-norm of vector r = xpk - xp
    mpfr_set_d(r, 0.00, MPFR_RNDN);
    for(int i = 0; i < n; i++) {
        mpfr_sub(aux, xpk[i], xp[i], MPFR_RNDN);
        mpfr_abs(aux, aux, MPFR_RNDN);
        mpfr_add(r, r, aux, MPFR_RNDN);
    }

    // Compute the 1-norm of xp 
    mpfr_set_d(aux, 0.00, MPFR_RNDN);
    for(int i = 0; i < n; i++){
        mpfr_abs(xp[i], xp[i], MPFR_RNDN);
        mpfr_add(aux, aux, xp[i], MPFR_RNDN);
    }

    // Divide the 1-norm of r by the 1-norm of xp
    mpfr_div(r, r, aux, MPFR_RNDN);

    // Output relative error
    error = mpfr_get_d(r, MPFR_RNDN);    

    // clear mpfr variables
    mpfr_clear(r);
    mpfr_clear(aux);
    for(unsigned int i=0; i<n; ++i) {
        mpfr_clear(xp[i]);
        mpfr_clear(xpk[i]);
    }

    return error;
}
/* Compute Rel. Error of K Parts solution with GEPP */
double linsolv_svd_error(const double* A,const double* b, double** x, mpfr_t* xp,
const unsigned int n, const unsigned int k, const unsigned int prec){
	
    double error;

    // store mpfr variables
    mpfr_t xkp[n], temp[n], mpfr_err, xp_inorm;
    mpfr_init2(mpfr_err, prec); mpfr_init2(xp_inorm, prec);
    for(int i = 0; i < n; i++){
        mpfr_init2(xkp[i], prec);
        mpfr_init2(temp[i], prec);
        mpfr_set(temp[i], xp[i], prec);
    }

    // Sum K parts and compute vector r = xp - xkp
    for(int i = 0; i < n; i++){
        mpfr_set_d(xkp[i], x[i][0], MPFR_RNDN);
        for(int j = 1; j < k; j++) mpfr_add_d(xkp[i], xkp[i], x[i][j], MPFR_RNDN);
        mpfr_sub(xkp[i], temp[i], xkp[i], MPFR_RNDN);
    }

    // Compute the 1-norm of r and the 1-norm of xp
    mpfr_set_d(xp_inorm, 0.00, MPFR_RNDN);
    mpfr_set_d(mpfr_err, 0.00, MPFR_RNDN);
    for(int i = 0; i < n; i++){
        mpfr_abs(xkp[i], xkp[i], MPFR_RNDN);
        mpfr_abs(temp[i], temp[i], MPFR_RNDN);
        mpfr_add(mpfr_err, mpfr_err, xkp[i], MPFR_RNDN);
        mpfr_add(xp_inorm, xp_inorm, temp[i], MPFR_RNDN);
    }

    // Divide the 1-norm of r by the 1-norm of xp
    mpfr_div(mpfr_err, mpfr_err, xp_inorm, MPFR_RNDN);

    // Output relative error
    error = mpfr_get_d(mpfr_err, MPFR_RNDN);    

    // clear mpfr variables
    mpfr_clear(xp_inorm);
    mpfr_clear(mpfr_err);
    for(unsigned int i=0; i<n; i++){
        mpfr_clear(xkp[i]);
        mpfr_clear(temp[i]);
    }

    return error;
}
/* Testing DotParts Against MPFR Dot Product */
void dotprod_acc_test(char** argv){ 
	
	// Initialize test input parameters with user input
	int vec_size = atoi(argv[2]);
	int sample_size = atoi(argv[3]);
	int k_min = atoi(argv[4]);
	int k_max = atoi(argv[5]);
	
	// Initialize/Declare test output
	mpfr_t* res_MPFR = (mpfr_t*)malloc(sizeof(mpfr_t));
	double cond;
	
	// Test vectors v and u as well as parts vectors uk and vk
	double* v = (double*)malloc(vec_size*sizeof(double));
	double* u = (double*)malloc(vec_size*sizeof(double));
	double** uk = (double**)malloc(vec_size*sizeof(double*));
	double** vk = (double**)malloc(vec_size*sizeof(double*));
	
	FILE* file = fopen("../csv/dotprod_acc_test.csv", "w+");
	fprintf(file, "cond, rel_err\n");
	for(int k = k_min; k < k_max + 1; k++){
		
		// Compute precision and allocate k-parts based on k
		double* res_KK = (double*)malloc(k*sizeof(double));
		int prec = 64*k - round(2*log2(64*k)) + 13;
		for(int i = 0; i < vec_size; i++) {
			uk[i] = (double*)malloc(k*sizeof(double));
			vk[i] = (double*)malloc(k*sizeof(double));
            setPartsD(res_KK, 0, k);
		}
		
		for(int s = 0; s < sample_size; s++){
						
			// Make test vectors from input condition number
			cond = (double)pow(2, rand() % MAX_EXPONENT + 1);
			gendotLM(u, v, uk, vk, vec_size, k, k, &cond);
		
			// Calculate K-Fold and MPFR Dot Products
            dotParts((const double**)uk, (const double**)vk, 0, res_KK, vec_size, k);
			dotMPFR(u, v, res_MPFR, vec_size, 2*prec);
			
			// Compute Relative Error and Rounded Result of dotKK and dotMPFR
			double rel_err = dotprod_error(res_MPFR, res_KK, k, 2*prec);
			double res_KK_d = sumk(res_KK, k, k);
			double res_MPFR_d = mpfr_get_d(*res_MPFR, MPFR_RNDN);
			
			// Output test results
			fprintf(file, "%.4e, ", cond);
			if(rel_err >= 1.0) fprintf(file, "%.4e \n", 1.0);
			else fprintf(file, "%.4e \n", rel_err);
		}
		printf("K=%d Complete!\n", k);
		free(res_KK);
		for(int i = 0; i < vec_size; i++) { 
			free(uk[i]);
			free(vk[i]);
		}
	}
	
	// Deallocate memory and resources
	free(uk);
	free(vk);
	free(u);
	free(v);
	free(res_MPFR); 
	fclose(file);
	printf("DotProd Accuracy Testing Complete! \n");
}
/* Testing GEPP and Iterative Refinement with Hilbert Matrix */
void linsolv_hilbert_test(char** argv){

    // Initialize given parameters and open output file
    int n = atoi(argv[2]);
    int k_min = atoi(argv[3]);
    int k_max = atoi(argv[4]);
    FILE* file = fopen("../csv/hilbert.csv", "w+");
    
    // Assign variables needed for test execution
    double sgn, scale, coeff1, coeff2, coeff3, gepp_err, iter_err;
    double cond = 0;
    double** A = (double**)malloc(n*n*sizeof(double*));
    double** LU = (double**)malloc(n*n*sizeof(double*));
    double* b = (double*)malloc(n*sizeof(double));
    double* x = (double*)malloc(n*sizeof(double));
    double** xk = (double**)malloc(n*sizeof(double*));
    int ipiv[n];

    fprintf(file, "cond, k, gepp_error, iter_error\n");
    for(int k = k_min; k < k_max + 1; k++){
        for(int h = 0; h < 1; h++){             // Change this to test different columns in the inverse hilbert matrix
            for(int i = 0; i < n; i++) {

                if(h == i) b[i] = 1;
                else b[i] = 0;

                // Construct the "exact" integer solution
                sgn = pow(-1, h + i + 2);
                scale = h + i + 1;
                coeff1 = binomialCoeff(n + h, n - i - 1);
                coeff2 = binomialCoeff(n + i, n - h - 1);
                coeff3 = binomialCoeff(h + i, h);
                coeff3 = pow(coeff3, 2);
                x[i] = sgn*scale*coeff1*coeff2*coeff3;

                xk[i] = (double*)malloc(k*sizeof(double)); 
                if(h == i) setPartsD(xk[i], 1, k);
                else setPartsD (xk[i], 0, k);

                for(int j = 0; j < n; j++){
                    A[j*n + i] = (double*)malloc(k*sizeof(double));
                    LU[j*n + i] = (double*)malloc(k*sizeof(double));
                }
            }
            
            // Construct the hilbert matrix using k-parts arithmetic
            hilbmatrix(A, n, k, &cond);
            for(int i = 0; i < n; i++){
                for(int j = 0; j < n; j++){ setParts(A[j*n + i], LU[j*n + i], k); }
            }

            // LU Decomposition and GEPP
            LUParts(LU, ipiv, n, k);
            GEPPPartsO((const double**)LU, xk, ipiv, n, k);
            gepp_err = linsolv_hilbert_error(x, xk, n, k, 32768);

            // Iterative Refinement
            itrefParts((const double**)A, (const double**)LU, xk, (const double*)b, (const int*)ipiv, n, k, pow(DBL_EPSILON, k), 5);
            iter_err = linsolv_hilbert_error(x, xk, n, k, 32768);

            // Output error results
            fprintf(file, "%.4e, %d, ", cond, k);
            if(gepp_err >= 1) fprintf(file, "%.4e, ", 1.0);
            else fprintf(file, "%.4e, ", gepp_err);

            if(iter_err >= 1) fprintf(file, "%.4e\n", 1.0);
            else fprintf(file, "%.4e\n", iter_err);
            printf("%dth column testing complete...\n", h);
        }
        for(int i = 0; i < n; i++){
            free(xk[i]);
            for(int j = 0; j < n; j++){
                free(A[j*n + i]);
                free(LU[j*n + i]);
            }
        }
        printf("k=%d complete!\n", k);
    }
    printf("Hilbert Matrix Testing Complete!\n");
    free(A);
    free(b);
    free(x);
    free(xk);
    fclose(file);
}
/* Testing GEPP and Iterative Refinement with SVD Matrix */
void linsolv_svd_test(char** argv){

    // Initialize given parameters and open output file
    int n = atoi(argv[2]);
    int sample_size = atoi(argv[3]);
    int k_min = atoi(argv[4]);
    int k_max = atoi(argv[5]);
    FILE* file = fopen("../csv/svd.csv", "w+");

    // Assign variables needed for test execution
    double cond = 0, gepp_err = 0, iter_err = 0;
    int prec = 32768;
    double* A = (double*)malloc(n*n*sizeof(double));
    double** kA = (double**)malloc(n*n*sizeof(double*));
    double** LU = (double**)malloc(n*n*sizeof(double*));
    double* b = (double*)malloc(n*sizeof(double));
    double** xk = (double**)malloc(n*sizeof(double*));
    mpfr_t* xp = (mpfr_t*)malloc(n*sizeof(mpfr_t));
    int ipiv[n];

    for(int i = 0; i < n; i++){ 
        b[i] = 1;
        mpfr_init2(xp[i], prec);
    }

    fprintf(file, "cond, k, gepp_error, iter_error\n");
    for(int k = k_min; k < k_max + 1; k++){

        for(int i = 0; i < n; i++){
            xk[i] = (double*)malloc(k*sizeof(double)); 
            for(int j = 0; j < n; j++){
                LU[j*n + i] = (double*)malloc(k*sizeof(double));
                kA[j*n + i] = (double*)malloc(k*sizeof(double));
            }
        }

        for(int s = 0; s < sample_size; s++){

            // Make the SVD matrix based on a random condition number <= 10^16
            cond = (double)pow(2, rand() % 55 + 1);
            svdmatrix(A, n, &cond);

            // Solve the linear system in high precision
            linearSolveMPFR(A, b, xp, n, prec);
            
            for(int i = 0; i < n; i++){
                for(int j = 0; j < n; j++){ 
                    setPartsD(LU[j*n + i], A[j*n + i], k); 
                    setPartsD(kA[j*n + i], A[j*n + i], k);
                }
            }

            // LU Decomposition and GEPP
            LUParts(LU, ipiv, n, k);
            GEPPParts((const double**)LU, b, xk, ipiv, n, k);
            gepp_err = linsolv_svd_error(A, b, xk, xp, n, k, prec);

            // Iterative Refinement
            itrefParts((const double**)kA, (const double**)LU, xk, (const double*)b, (const int*)ipiv, n, k, pow(DBL_EPSILON, k), 5);
            iter_err = linsolv_svd_error(A, b, xk, xp, n, k, prec);

            // Output results
            fprintf(file, "%.4e, %d, ", cond, k);
            if(gepp_err >= 1) fprintf(file, "%.4e, ", 1.0);
            else fprintf(file, "%.4e, ", gepp_err);

            if(iter_err >= 1) fprintf(file, "%.4e\n", 1.0);
            else fprintf(file, "%.4e\n", iter_err);
        }
        for(int i = 0; i < n; i++){
            free(xk[i]);
            for(int j = 0; j < n; j++){
                free(kA[j*n + i]);
                free(LU[j*n + i]);
            }
        }
        printf("k=%d complete!\n", k);
    }
    
    printf("SVD Matrix Testing Complete!\n");
    for(int i = 0; i < n; i++) mpfr_clear(xp[i]);
    free(A);
    free(LU);
    free(kA);
    free(b);
    free(xk);
    free(xp);
    fclose(file);
}
/* Main Function */
int main(int argc,char **argv){
    
    // Random seed for the program based on the current time
	srand(time(NULL));

	if(argc < 2){
		printf("No Argument Provided\n");
        printf("SYNTAX: ../tests/kpart_test [program] [vector/matrix size] [sample size] [k min] [k max] \n");
		printf("where [algorithm] = dotprod, hilbert, svd \n");
		return 1;
	}
	
	// Go to one of the tests based on the [algorithm] input
	if(strcmp(argv[1], "dotprod") == 0){
		if(argc != 6){
			printf("INVALID INPUT: dotprod must be ran on the command line as follows \n");
			printf("SYNTAX: ../tests/kpart_test dotprod [vector size] [sample size] [k min] [k max]\n");
			return 1;
		}
		else{
			// Every argument past the [algorithm] must be a non-zero, nonnegative nteger
			for(int i = 2; i < argc; i++){
				if(atoi(argv[i]) < 1){
					printf("INVALID INPUT: All numerical inputs must be non-zero, nonnegative integers.\n");
					return 1;
				}
			}
			printf("Starting DotProd Test...\n");
			dotprod_acc_test(argv);
		}
	}
	else if(strcmp(argv[1], "hilbert") == 0){
		if(argc != 5){
			printf("INVALID INPUT: dotKK must be ran on the command line as follows \n");
			printf("SYNTAX: ../tests/kpart_test hilbert [matrix size] [k min] [k max]\n");
			return 1;
		}
		else{
			// Every argument past the [algorithm] must be a non-zero, nonnegative nteger
			for(int i = 2; i < argc; i++){
				if(atoi(argv[i]) < 1){
					printf("INVALID INPUT: All numerical inputs must be non-zero, nonnegative integers.\n");
					return 1;
				}
			}
			printf("Starting Hibert Matrix Test...\n");
			linsolv_hilbert_test(argv);
		}
	}
    else if(strcmp(argv[1], "svd") == 0){
        if(argc != 6){
            printf("INVALID INPUT: svd must be ran on the command line as follows \n");
            printf("SYNTAX: ../tests/kpart_test svd [matrix size] [sample size] [k min] [k max]\n");
            return 1;
        }
        else{
            // Every argument past the [algorithm] must be a non-zero, nonnegative nteger
            for(int i = 2; i < argc; i++){
                if(atoi(argv[i]) < 1){
                    printf("INVALID INPUT: All numerical inputs must be non-zero, nonnegative integers.\n");
                    return 1;
                }
            }
            printf("Starting SVD Matrix Test...\n");
            linsolv_svd_test(argv);
        }
    }
	else{
		printf("Function not found.\n");
		printf("SYNTAX: ../tests/kpart_test [program] [vector/matrix size] [sample size] [k min] [k max] \n");
		printf("where [algorithm] = dotprod, hilbert, svd \n");
		return 1;
	}
	return 0;
}