#ifndef EFT
#define EFT
/* EFT Data Structure */
struct eft{
	double fl_res, fl_err;
};
/* Two Sum (error free transformation of sum operation) */
void two_sum(const double a,const double b,struct eft* res);
/* Two Product (error free transformation of product operation)*/
void two_prod(const double a,const double b,struct eft* res);
/* Vector Sum (distillation algorithm) */
void vec_sum(double* p,const unsigned int n);
#endif