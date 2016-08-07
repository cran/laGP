#ifndef __COVAR_H__
#define __COVAR_H__


void covar_symm(const int col, double **X, const int n, 
                double d, double g, double **K);
void covar(const int col, double **X1, const int n1, double **X2,
	   const int n2, double d, double **K);
void diff_covar_symm(const int col, double **X, const int n, 
                     double d, double **dK, double **d2K);
void diff_covar(const int col, double **X1, const int n1, double **X2,
		const int n2, double d, double **dK, double **d2K);
void distance(double **X1, const unsigned int n1, double **X2,
	      const unsigned int n2, const unsigned int m, double **D);
void distance_R(double *X1_in, int *n1_in, double *X2_in, 
		int *n2_in, int *m_in, double *D_out);
void distance_symm_R(double *X_in, int *n_in, int *m_in, double *D_out);
void dist2covar_R(double *D_in, int *n1_in, int *n2_in, double *d_in,
		  double *g_in, double *K_out);
void dist2covar_symm_R(double *D_in, int *n_in, double *d_in,
		       double *g_in, double *K_out);
void calc_g_mui_kxy(const int col, double *x, double **X, 
		    const int n, double **Ki, double **Xref, 
		    const int m, double d, double g, double *gvec, 
        	    double *mui, double *kx, double *kxy);
#endif

