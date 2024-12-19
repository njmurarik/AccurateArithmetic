#include "eft.h"
#include "kpart.h"
#include "sum.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>


/* ########## Arithmetic Operations ########## */

/* kpartSum (sum of two kpart numbers) */
void sumParts(const double *x, double *y, const unsigned int k){
    const unsigned int n = 2*k-1;
    double e[n];
    double r = x[0];
    struct eft res;
    for(unsigned int i=1; i<k; ++i){                // e[0:k-2] error in sum of x
        two_sum(r,x[i],&res);
        r = res.fl_res;
        e[i-1] = res.fl_err;
    }
    for(unsigned int i=0; i<k; ++i){                // e[k-1:2k-2] error in sum of y
        two_sum(r,y[i],&res);
        r = res.fl_res;
        e[k + i - 1] = res.fl_err;
    } 
    y[0] = r;                                       // y[0] = fl(x + y)
    for(unsigned int i=1; i<k-1; ++i){
        vec_sum(e,n-i+1);
        y[i] = e[n-i];                              // y[i] = vecsum(e[0:n-i+1]), 1\leq i\leq k-2
    }
    y[k-1] = e[0];                                  // y[k-1] = fl(\sum e[0:k])
    for(unsigned int i=1; i<k+1; ++i){
        y[k-1] += e[i];
    }
}
/* kpartAddD (add double to kpart number) */
void sumPartsD(const double xval, double *y, const unsigned int k){
    double e[k];
    double r = xval;
    struct eft res;
    for(unsigned int i=0; i<k; ++i){                // e[0:k-1] error in sum of xval + x
        two_sum(r,y[i],&res);
        r = res.fl_res;
        e[i] = res.fl_err;
    }
    y[0] = r;                                       // y[0] = fl(xval + x)
    for(unsigned int i=1; i<k-1; ++i){
        vec_sum(e,k-i+1);
        y[i] = e[k-i];                              // y[i] = vecsum(e[0:k-i+1]), 1\leq i\leq k-2
    }
    y[k-1] = e[0] + e[1];                           // y[k-1] = fl(e[0]+e[1])
}
/* kpartProd (product of two kpart numbers) */
void prodParts(const double *x, double *y, const unsigned int k){
    const unsigned int n = 2*k*k;
    double e[n];
    double r = 0;
    struct eft res;
    for(unsigned int i=0; i<k; ++i){                // e[0:k^2-1] error in product x[i]*x[j]
        for(unsigned int j=0; j<k; ++j){            // e[k^2:2*k^2-1] error in r + x[i]*y[j]
            two_prod(x[i],y[j],&res);
            e[k*i + j] = res.fl_err;
            two_sum(r,res.fl_res,&res);
            r = res.fl_res;
            e[n - k*i - j - 1] = res.fl_err;        // e[n-1]=0 since the error in the first sum is zero
        }
    }
    y[0] = r;                                       // y[0] = fl(x*y)
    for(unsigned int i=1; i<k-1; ++i){
        vec_sum(e,n-i);
        y[i] = e[n-i-1];                            // y[i] = vecsum(e[0:n-i-1]), 1\leq i\leq k-2, we ignore e[n-1] since it is zero
    }
    y[k-1] = e[0];                                  // y[k-1] = fl(\sum e[0:n-k])
    for(unsigned int i=1; i<n-k+1; ++i){
        y[k-1] += e[i];
    }
}
/* kpartMulD (multipy kpart number by double) */
void prodPartsD(const double xval, double *y, const unsigned int k){
    const unsigned int n = 2*k;
    double e[n];
    double r = 0;
    struct eft res;
    for(unsigned int i=0; i<k; ++i){                // e[0:k-1] error in product xval*x[i]
        two_prod(xval,y[i],&res);                   // e[k:2*k-1] 
        e[i] = res.fl_err;
        two_sum(r,res.fl_res,&res);
        r = res.fl_res;
        e[n-i-1] = res.fl_err;                      // e[n-1]=0 since the error in the first sum is zero
    }
    y[0] = r;                                       // y[0] = fl(xval*x)
    for(unsigned int i=1; i<k-1; ++i){
        vec_sum(e,n-i);                           
        y[i] = e[n-i-1];                            // y[i] = vecsum(e[0:n-i-1]), 1\leq i\leq k-2, we ignore e[n-1] since it is zero
    }
    y[k-1] = e[0];                                  // y[k-1] = fl(\sum e[0:k])
    for(unsigned int i=1; i<k+1; ++i){
        y[k-1] += e[i];
    }
}
/* kpartFMA (fused multiply-add: {z} = {x}{y} + {z}) */
void FMAParts(const double *x, const double *y, double *z, const unsigned int k){
    double* r = (double*)malloc(k*sizeof(double));
    for(int i = 0; i < k; i++) r[i] = y[i];
    prodParts(x, r, k);
    sumParts(r, z, k);
    free(r);
}
/*kpartFMA (fused multiply-add: {z} = {x}y + {z})
void FMAPartsD(const double* x, const double y, double* z, const unsigned int k){ 
    const unsigned int n = 3*k;
    double e[n];
    double r = 0;
    struct eft res;
    for(unsigned int i=0; i<k; i++){                // e[0:k-2] = error in sum of z
        two_sum(r,z[i],&res);
        r = res.fl_res;
        e[i] = res.fl_err;
    }
    for(unsigned int i = 0; i < k; i++){            // e[k^2+k-1:n-1] = error in sum r + x[i]*y[j]
        two_prod(x[i], y, &res);
        e[i + k] = res.fl_err;
        two_sum(r,res.fl_res,&res);
        r = res.fl_res;                         
        e[i + 2*k] = res.fl_err;        // e[n-1] may not be zero since r is not set to zero on first sum
    }
    z[0] = r;                                       // z[0] = fl(x*y + z)
    for(unsigned int i = 1; i < k - 1; i++){
        vec_sum(e,n - i + 1);
        z[i] = e[n - i];                             // z[i] = vecsum(e[0:n-i]), 1\leq i\leq k-2, we don't ignore e[n-1] since it may not be zero
    }
    z[k - 1] = 0.00;                                  // z[k-1] = fl(\sum e[0:n-k+1])
    for(unsigned int i = 0; i < n - k + 2; i++) z[k - 1] += e[i];
}
*/
/* kpartNeg (negative kpart number) */
void negParts(double* x,const unsigned int k){
    for(unsigned int i=0; i<k; ++i){
        x[i] *= -1.0;
    }
}
/* kpartPrint */
void printParts(const double *x, const unsigned int k){
    for(unsigned int i=0; i<k; ++i){
        printf("%e ",x[i]);
    }
    printf("\n");
}
/* kpartSet */
void setParts(const double *x,double *y,const unsigned int k){
    for(unsigned int i=0; i<k; ++i){
        y[i] = x[i];
    }
}
/* kpartSetD */
void setPartsD(double *x, const double xval, const unsigned int k){
    x[0] = xval;
    memset(x+1,0,(k-1)*sizeof(double));
}
/* kpartDiv (division using Newton's Method where b is in k parts)*/
void divParts(double* x, const double* b, const unsigned int k){

    double* y = (double*)malloc(k*sizeof(double));
    double* z = (double*)malloc(k*sizeof(double));
    setParts(b, z, k);
    negParts(z, k);

    for(int h = 0; h < ceil(log2(k)); h++){
        setParts(x, y, k);
        prodParts(x, y, k);
        for(int i = 0; i < k; i++) x[i] *= 2;
        FMAParts(z, y, x, k);
    }

    free(y);
    free(z);
}
/* kpartDivD (division using Newton's Method where b is in 1 part)*/
void divPartsD(double* x, const double b, const unsigned int k){
    const unsigned int m = k*(k + 1) + 1;
    const unsigned int n = 2*m + k;
    double y[m];
    double err[n];
    struct eft res;
    double p;

    for(int h = 0; h < ceil(log2(k)) + 1; h++){
        p = 0;
        for(int i = 0; i < k; i++){
            two_prod(x[i], x[i], &res);                         // y[0:k-1] = error in the x_i*x_i product
            y[i] = res.fl_err;
            two_sum(p, res.fl_res, &res);
            y[((k*(k + 1)) / 2) + i] = res.fl_err;              // y[1/2(k(k+1)):1/2(k(k+1)) + (k-1)] = error in the sum of x_i^2
            p = res.fl_res;
            for(int j = i + 1; j < k; j++){
                int index = (i + 1)*k - ((i*(i + 1)) / 2) + (j - i) - 1;
                two_prod(2*x[i], x[j], &res);
                y[index] = res.fl_err;                          // y[index] = error in the product of x_i*x_j
                two_sum(p, res.fl_res, &res);
                y[((k*(k + 1)) / 2) + index] = res.fl_err;      // y[((k*(k + 1)) / 2) + index] = error in the sum of x_i^2 + x_i*x_j
                p = res.fl_res;
            }
        }

        y[m - 1] = p;                                           // y[m-1] = fl(x^2)
        p = 0;

        for(int i = 0; i < k; i++){
            two_sum(p, 2*x[i], &res);
            err[i] = res.fl_err;                                // err[0:k-1] = error in adding 2x to p
            p = res.fl_res;
        }
        for(int i = 0; i < m; i++){
            two_prod(-b, y[i], &res);                           // err[k: m(k-1)+(m-1)+k] = error in multiplying b and y
            err[i + k] = res.fl_err;
            two_sum(p, res.fl_res, &res);
            err[m + i + k] = res.fl_err;                        // err[index] = error in adding -by to p
            p = res.fl_res;
        }
        
        x[0] = p;                                               // fl(2x - bx^2)
        for(int i = 0; i < k - 1; i++){
            vec_sum(err, n - i);                                // x[i] = vecsum(err[0:n-i])
            x[i + 1] = err[n - (i + 1)];
        }
        x[k - 1] = 0.00;
        for(int i = 0; i < n - k + 2; i++) x[k - 1] += err[i];  // x[k-1] = fl(\sum err[0:n-k+1])
    }
}
/* Splits a double into k-parts by manipulating the significand */
void splitParts(double x, double* res, int k){
	
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

/* ########## Linear Algebra Operations ########## */

/* Display a Matrix that is in K Parts */
void printMatParts(const unsigned int m, const unsigned int n, const unsigned int k, 
const double** kA, char* desc){

    printf( "\n %s\n", desc );
    for(int i = 0; i < m; i++){
        for(int j = 0; j < n; j++){
            printf("########## a_%d%d ##########\n", i, j);
            printParts(kA[j*n + i], k);
        }
    }
}
/* Dot Product in fl_kk from floating-point vectors */
void dotPartsD(const double* u, const double* v, double* d, const unsigned int n, const unsigned int k){
    double err[2*n];
	struct eft res;
	double r = 0;
	for(int h = 0; h < n; h++){
		two_prod(u[h], v[h], &res);
		err[h] = res.fl_err;                                    // err[0:n-1] = error in the product u_h*v_h
		two_sum(r, res.fl_res, &res);
        r = res.fl_res;                                         // err[n:n-1] = error in the sum of the products
		err[n + h] = res.fl_err;
	}
	d[0] = r;                                                   // d[0] = fl(u * v)
	for(int r = 0; r < k - 2; r++){
		vec_sum(err, 2*n - r);                                  // d[1:k-2] = vecsum(err[0:2n - r])
		d[r + 1] = err[2*n - (r + 1)];
	}
	d[k - 1] = 0.00;
	for(int r = 0; r < 2*n - k + 2; r++) d[k - 1] += err[r];    // d[k-1] = fl(e[0: 2n-k+1])
}
/* Dot Product in fl_kk precision */
void dotParts(const double** u, const double** v, double b, double* d, const unsigned int n, const unsigned int k){
    setPartsD(d, b, k);
    for(int i = 0; i < n; i++){
        FMAParts(u[i], v[i], d, k);
    }
}
/* Dot Product in fl_kk where inputs are in l and m parts */
void dotPartsLM(const double** u, const double** v, double* d, const unsigned int n, 
const unsigned int l, const unsigned int m, const unsigned int k) {
	double err[2*l*m*n];
	struct eft res;
	double r = 0;
	for(int h = 0; h < n; h++){
		for(int i = 0; i < l; i++){
			for(int j = 0; j < m; j++){
				two_prod(u[h][i], v[h][j], &res);
				err[h*l*m + i*m + j] = res.fl_err;                  // err[0:lmn - 1] = error in the product {u_h}*{v_h}
				two_sum(r, res.fl_res, &res);
                r = res.fl_res;
				err[l*m*n + h*l*m + i*m + j] = res.fl_err;          // err[lmn:2lmn - 1] = error in the sum of the products
			}
		}
	}
	d[0] = r;                                                       // d[0] = fl({u} * {v})
	for(int r = 0; r < k - 2; r++){
		vec_sum(err, 2*l*m*n - r);                                  // d[1:k-2] = vecsum(err[0:2lmn - r])
		d[r + 1] = err[2*l*m*n - (r + 1)];
	}
	d[k - 1] = 0.00;
	for(int r = 0; r < 2*l*m*n - k + 2; r++) d[k - 1] += err[r];    // d[k-1] = fl(e[0: 2lmn-k+1])
}
/* LU Decomposition in K Parts from K Parts */
void LUParts(double** A, int* perm, const unsigned int n, const unsigned int k){
    
    // Gaussian Elimination with partial pivoting
    double* lp = (double*)malloc(k*sizeof(double));
    double *ptr;
    for (unsigned int h = 0; h < n; h++) perm[h] = h;
    for(int j=0; j<n-1; ++j){
        setPartsD(lp, 0, k);

        // select pivot in jth column
        int m = j;                                  
        for(int i=j+1; i<n; ++i){
            if(fabs(A[n*j + i][0]) > fabs(A[n*j + m][0])){
                m = i;
            }
        }
        // swap m and j rows (over columns l=0,...,n-1)
        if(m > j){                                  
            int temp = perm[j];
            perm[j] = perm[m];
            perm[m] = temp;
            for(int l=0; l<n; ++l){
                ptr = A[l*n + j];
                A[l*n + j] = A[l*n + m];
                A[l*n + m] = ptr;
            }                                               
        }
        // apply row operations
        for(int i=j+1; i<n; ++i){
            setPartsD(lp, 1.0 / A[j*n + j][0], k);
            divParts(lp, A[j*n + j], k);
            prodParts(A[j*n + i], lp, k);
            setParts(lp, A[j*n + i], k);

            // lp = -ap_{ij}/ap_{jj}
            negParts(lp, k);                        
            for(int l=j+1; l<n; ++l){
                FMAParts(lp, A[l*n + j], A[l*n + i], k);
            }
        }
    }
    free(lp);
}
/* Gaussian Elimination with Partial Pivoting s.t. {x} --> A{x} = b */
void GEPPParts(const double** A, const double* b, double** x, const int* ipiv, const unsigned int n, const unsigned int k){

    // Intermediate arithmetic number
    double* lp = (double*)malloc(k*sizeof(double));
    setPartsD(lp, 0, k);
	
    // Forward Substitution from Ly = b
    for(int i = 0; i < n; i++){
        setPartsD(x[i], b[ipiv[i]], k);
        for(int j = 0; j < i; j++){
            // x[i] -= x[j]*A[j*n + i] --> x_i = x_i - x_j*a_ij
            setParts(A[j*n + i], lp, k);
            negParts(lp, k);
            FMAParts(x[j], lp, x[i], k);
        }
    }
    
    setPartsD(lp, 0, k);
    // Backward Substitution from Ux = y
    for(int i = n - 1; i > -1; i--){
        for(int j = n - 1; j > i; j--){
            // x[i] -= x[j]*A[j*n + i] --> x_i = x_i - x_j*a_ij
            setParts(A[j*n + i], lp, k);
            negParts(lp, k);
            FMAParts(x[j], lp, x[i], k);
        }
        // x[i] /= A[i*n + i] --> x_i = x_i / a_ii
        setPartsD(lp, 1.0 / A[i*n + i][0], k);
        divParts(lp, A[i*n + i], k);
        prodParts(lp, x[i], k);
    }

    free(lp);
}
/* Gaussian Elimination with Partial Pivoting s.t. {x} --> A{x} = b where b is overwritten */
void GEPPPartsO(const double** A, double** b, const int* ipiv, const unsigned int n, const unsigned int k){
	
    // Intermediate Arithmetic Number
    double* z = (double*)malloc(n*sizeof(double));
    setPartsD(z, 0, k);

    // Solution of Ax = b
    double** x = (double**)malloc(n*sizeof(double*));
    for(int i = 0; i < n; i++) {
        x[i] = (double*)malloc(k*sizeof(double));
        setPartsD(x[i], 0, k);
    }

    // Forward Substitution with Ly = b
    for(int i = 0; i < n; i++){
        setParts(b[ipiv[i]], x[i], k);
        for(int j = 0; j < i; j++){
            // x[i] -= x[j]*A[j*n + i] --> x_i = x_i - x_j*a_ij
            setParts(A[j*n + i], z, k);
            negParts(z, k);
            FMAParts(x[j], z, x[i], k);
        }
    }

    // Backward Substution with Ux = y
    for(int i = n - 1; i > -1; i--){
        for(int j = i + 1; j < n; j++){
            // x[i] -= x[j]*A[j*n + i] --> x_i = x_i - x_j*a_ij
            setParts(A[j*n + i], z, k);
            negParts(z, k);
            FMAParts(x[j], z, x[i], k);
        }
        // x[i] /= A[i*n + i] --> x_i = x_i / a_ii
        setPartsD(z, 1.0 / A[i*n + i][0], k);
        divParts(z, A[i*n + i], k);
        prodParts(z, x[i], k);
    }

    // Set b = x and free allocated variables
    for(int i = 0; i < n; i++) {
        setParts(x[i], b[i], k);
        free(x[i]);
    }
    free(x);
    free(z);
}
/* K-Fold Precision Calculation of x = A^{-1}b or equivalently (b - Ax) */
void matresParts(const double** A, const double** x, const double* b, double** z, const unsigned int n, const unsigned int k){
    
    // Temporary variable used in dot product computations
    double** u = (double**)malloc(n*sizeof(double)); 
	double** v = (double**)malloc(n*sizeof(double));

    for(int i = 0; i < n; i++){
        u[i] = (double*)malloc(k*sizeof(double));
        v[i] = (double*)malloc(k*sizeof(double));
        setParts(x[i], v[i], k); // Allocate components of v based on {x}
    }
    // Allocate the components of u based on the row of A and the vector b
	for(int i = 0; i < n; i++){
		for(int j = 0; j < n; j++){
            setParts(A[j*n + i], u[j], k);
            negParts(u[j], k);
		}
        // Compute the dot product of u and v in 2k-fold precision and stored in 2k parts
        dotParts((const double**) u, (const double**) v, b[i], z[i], n, k);
	}

    // Free any allocated resources.
    for(int h = 0; h < n; h++) {
        free(v[h]);
        free(u[h]);
    }
	free(u);
	free(v);
}
/* Iterative Refinement in K Parts */
void itrefParts(const double** A, const double** LU, double** x, const double* b, const int* ipiv, 
const unsigned int n, const unsigned int k, const double TOL, const unsigned int max_count){
	
    // {z} is the k parts vector so that A{z} = b - A{x}
	double** z = (double**)malloc(n*sizeof(double*));
    for(int i = 0; i < n; i++) z[i] = (double*)malloc(k*sizeof(double));
    
    // Variable to compute the 1-norm of x and z
    double xnorm, znorm;

	for(int c = 0; c < max_count; c++){

        // Compute Residual: b - {A}{x}
		matresParts((const double**)A, (const double**)x, (const double*)b, z, n, k);

        // Use GEPP so that A{z} = b - {A}{x}
        GEPPPartsO((const double**)LU, z, (const int*)ipiv, n, k);
		
        // Compute the 1-norm of {z} and {x}
        xnorm = 0.0;
        znorm = 0.0;
        for(int i = 0; i < n; i++){
            for(int j = 0; j < k; j++) {
                znorm += fabs(z[i][j]);
                xnorm += fabs(x[i][j]);
            }
        }

        // If the relative norm of {z} is small enough, then exit the program.
        if((znorm / xnorm) < TOL) { break; }
        else { for(int i = 0; i < n; i++) sumParts(z[i], x[i], k); } // Otherwise, update x accordingly
	}

    // Free Allocated variable for {z}
	for(int i = 0; i < n; i++) free(z[i]);
	free(z);
}