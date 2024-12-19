#include "eft.h"
#include <math.h>
#include <stdlib.h>
/* Two Sum (error free transformation of sum operation )*/
void two_sum(const double a,const double b,struct eft* res){
	res->fl_res = a + b;
	double t = res->fl_res - a;
	res->fl_err = (a - (res->fl_res - t)) + (b - t);
}
/* Two Product (error free transformation of product operation)*/
void two_prod(const double a,const double b,struct eft* res){
	res->fl_res = a*b;
	res->fl_err = fma(a,b,-res->fl_res);
}
/* Vector Sum (distillation algorithm) */
void vec_sum(double* p,const unsigned int n){
	struct eft res;
	for(unsigned int i=1; i<n; ++i){
		two_sum(p[i],p[i-1],&res);
		p[i] = res.fl_res;
		p[i-1] = res.fl_err;
	}
}