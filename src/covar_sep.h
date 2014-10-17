#ifndef __COVAR_SEP_H__
#define __COVAR_SEP_H__

void covar_sep_symm(const int col, double **X, const int n, 
		    double *d, const double g, double **K);
void diff_covar_sep_symm(const int col, double **X, const int n, 
            double *d, double **K, double ***dK);

#endif
