#ifndef __IECI_H__
#define __IECI_H__


double EI(const double m, const double s2, const int df, const double fmin);
void calc_ecis(const int m, double *ktKik, double *s2p, const double phi, 
	       const double g, double *badj, double *tm, const double tdf, 
	       const double fmin);
double calc_ieci(const int m, double *ktKik, double *s2p, const double phi, 
		 const double g, double *badj, double *tm, const double tdf, 
		 const double fmin, double *w);
double calc_alc(const int m, double *ktKik, double *s2p, const double phi, 
		double *badj, const double tdf, double *w);
void calc_g_mui_kxy(const int col, double *x, double **X, 
		    const int n, double **Ki, double **Xref, 
		    const int m, double d, const double g, double *gvec, 
        double *mui, double *kx, double *kxy);
void calc_ktKikx(double *ktKik, const int m, double **k, const int n,
		 double *g, const double mui, double *kxy, double **Gmui_util,
		 double *ktGmui_util, double *ktKikx);
void rbetter_R(int *n_in, int *m_in, double *rect_in, double *ystar_in, 
	       double *X_out);

#endif

