#ifndef __LAGP_SEP_H__
#define __LAGP_SEP_H__

void aGPsep_R(int *m_in, int *start_in, int *end_in, double *XX_in, int *nn_in,
	int *n_in, double *X_in, double *Z_in, double *d_in, double *darg_in, 
	double *g_in, double *garg_in, int *imethod_in, int *close_in,
    int *ompthreads_in, int *numrays_in, double *rect_in, int *verb_in, 
    int *Xiret_in, int *Xi_out, double *mean_out, double *var_out, 
    double *dmle_out, int *dits_out, double *gmle_out, int *gits_out, 
    double *llik_out);

void laGPsep(const unsigned int m, const unsigned int start, 
             const unsigned int end, double **Xref, const unsigned int nref, 
             const unsigned int n, double **X, double *Z, double *d, double *g, 
             const Method method, unsigned int close, const unsigned int numrays, 
             double **rect, const int verb, int *Xi, double *mean, double *s2, 
             const unsigned int lite, double *df, double *dmle, int *dits, 
             double *gmle, int *gits, double *llik, int fromR);

void laGPsep_R(int *m_in, int *start_in, int *end_in, double *Xref_in, int *nref_in, 
               int *n_in, double *X_in, double *Z_in, double *d_in, double *g_in, 
               int *imethod_in, int *close_in, int *numrays_in, 
               double *rect_in, int *lite_in, int *verb_in, int *Xiret_in, int *Xi_out, 
               double *mean_out, double *s2_out, double *df_out, double *dmle_out, 
               int *dits_out, double *gmle_out, int *gits_out, double *llik_out);

#endif

