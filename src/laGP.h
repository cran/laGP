#ifndef __LAGP_H__
#define __LAGP_H__

typedef enum METHOD {ALC=1001, ALCRAY=1002, MSPE=1003, EFI=1004, NN=1005} Method;

void aGP_R(int *m_in, int *start_in, int *end_in, double *XX_in, int *nn_in,
		int *n_in, double *X_in, double *Z_in,double *d_in, double *darg_in, 
		double *g_in, double *garg_in, int *imethod_in, int *close_in,
		int *ompthreads_in, int *numgpus_in, int *gputhreads_in, int *nngpu_in,
		double *rect_in, int *verb_in, int *Xiret_in, int *Xi_out, double *mean_out, 
    double *var_out, double *dmle_out, int *dits_out, double *gmle_out, int *gits_out, 
		double *llik_out);

void laGP(const unsigned int m, const unsigned int start, const unsigned int end, 
       double **Xref, const unsigned int nref, const unsigned int n, double **X, 
       double *Z, double *d, double *g, const Method method, const unsigned int close, 
       const int alc_gpu, double **rect, const int verb, int *Xi, double *mean, 
       double *s2, double *df, double *dmle, int *dits, double *gmle, int *gits, 
       double *llik);

void laGP_R(int *m_in, int *start_in, int *end_in, double *Xref_in, int *nref_in,
         int *n_in, double *X_in, double *Z_in, double *d_in, double *g_in,
         int *imethod_in, int *close_in, int *alc_gpu_in, double *rect_in, int *verb_in, 
         int *Xiret_in, int *Xi_out, double *mean_out, double *s2_out, double *df_out,
         double *dmle_out, int *dits_out, double *gmle_out, int *gits_out,
         double *llik_out);

#endif

