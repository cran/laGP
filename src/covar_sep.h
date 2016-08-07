#ifndef __COVAR_SEP_H__
#define __COVAR_SEP_H__

void covar_sep_symm(const int col, double **X, const int n, 
		    double *d, double g, double **K);
void covar_sep(const int col, double **X1, const int n1, double **X2,
     const int n2, double *d, double **K);
void diff_covar_sep_symm(const int col, double **X1, const int n1, 
            double *d, double **K, double ***dK);
void diff_covar_sep(const int col, double **X, const int n, 
            double **X2, const int n2, double *d, double **K, 
            double ***dK);
void calc_g_mui_kxy_sep(const int col, double *x, double **X, 
        const int n, double **Ki, double **Xref, 
        const int m, double *d, double g, double *gvec, 
        double *mui, double *kx, double *kxy);

#endif
