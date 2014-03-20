#ifndef __GP_H__
#define __GP_H__

typedef struct gp {
  double **X;       /* design matrix */
  double **K;       /* covariance between design points */
  double **Ki;      /* inverse of K */
  double **dK;      /* first derivative of K; optional */
  double **d2K;     /* second derivative of K; also optional */
  double ldetK;     /* log of the determinant of K */
  double *Z;        /* response vector */
  double *KiZ;      /* Ki %*% Z */
  unsigned int m;   /* number of cols in X */
  unsigned int n;   /* number of rows in X; length of Z */
  double d;         /* lengthscale parameter to correlation */
  double g;         /* nugget parameter to correlation */
  double phi;       /* t(Z) %*% Ki %*% Z = t(Z) %*% KiZ, used for s2 */
  double F;         /* approx Fisher info; optional with dK */
} GP;

typedef enum THETA {LENGTHSCALE=2001, NUGGET=2002} Theta;

GP* newGP(const unsigned int m, const unsigned int n, double **X,
	  double *Z, const double d, const double g, const int dK);
GP* copyGP(GP* gp);
unsigned int get_gp(void);
void deletedKGP(GP *gp);
void deletedKGP_R(int *gpi_in);
void deleteGP(GP* gp);
void deleteGP_index(unsigned int i);
void deleteGP_R(int *gp);
void deleteGPs(void);
void deleteGPs_R(void);
double log_determinant_chol(double **M, const unsigned int n);
void dllikGP(GP *gp, double *ab, double *dllik, double *d2llik);
void dllikGP_nug(GP *gp, double *ab, double *dllik, double *d2llik);
void dllikGP_R(int *gpi_in, double *ab_in, double *dllik_out,
        double *d2llik_out);
void dllikGP_nug_R(int *gpi_in, double *ab_in, double *dllik_out,
        double *d2llik_out);
double fishinfoGP(GP *gp);
void calc_ZtKiZ(GP *gp);
void newdKGP(GP *gp);
void buildKGP_R(int *gpi_in);
GP* buildGP(GP *gp, int dK);
GP* newGP(const unsigned int m, const unsigned int n, double **X,
    double *Z, const double d, const double g, const int dK);
GP* newGP_sub(const unsigned int m, const unsigned int n, int *p, 
        double **X, double *Z, const double d, const double g, const int dK);
void newGP_R(int *m_in, int *n_in, double *X_in, double *Z_in,
       double *d_in, double *g_in, int *dK, int *gp_index);
void newparamsGP(GP* gp, const double d, const double g);
void newparamsGP_R(int *gpi_in, double *d_in, double *g_in);
double llikGP(GP *gp, double *dab, double *gab);
void llikGP_R(int *gpi_in, double *dab_in, double *gab_in, double *llik_out);
double Ropt(GP* gp, Theta theta, double tmin, double tmax, double *ab, 
        char *msg, int *its, int verb);
double mleGP(GP* gp, Theta theta, double tmin, double tmax, double *ab, 
        int *its, int verb);
void mleGP_R(int *gpi_in, int *param_in, int *verb_in, double *tmin_in,
       double *tmax_in, double *ab_in, double *mle_out, int * its_out);
void jmleGP(GP *gp, double *drange, double *grange, double *dab, double *gab,
            int *dits, int *gits, int verb);
void jmleGP_R(int *gpi_in, int *verb_in, double *drange_in, double *grange_in,
       double *dab_in, double *gab_in, double *d_out, double *g_out, int *dits_out,
       int *gits_out);
GP* copyGP(GP* gp);
void copyGP_R(int *gpi_in, int *newgpi_out);
void updateGP(GP* gp, unsigned int nn, double **XX, double *ZZ, int verb);
void updateGP_R(int *gpi_in, int *m_in, int *nn_in, double *XX_in, double *ZZ_in,
      int *verb_in);
void predGP(GP* gp, unsigned int nn, double **XX, double *mean, 
      double **Sigma, double *df, double *llik);
void new_predutilGP_lite(GP *gp, unsigned int nn, double **XX, double ***k, 
       double ***ktKi, double **ktKik);
void predGP_lite(GP* gp, unsigned int nn, double **XX, double *mean, 
     double *sigma2, double *df, double *llik);
void predGP_R(int *gpi_in, int *m_in, int *nn_in, double *XX_in,
        int *lite_in, double *mean_out, double *Sigma_out, double *df_out,
        double *llik_out);
void alcGP(GP *gp, unsigned int ncand, double **Xcand, unsigned int nref,
     double **Xref,  int verb, double *alc);
int lalcrayGP(GP *gp, double **Xcand, int ncand, double **Xref, int start, 
  double **rect, int verb);
#ifdef _GPU
void alcGP_gpu(GP *gp, unsigned int ncand, double **Xcand, unsigned int nref,
     double **Xref,  int verb, double *alc, int omp_threadnum);
#endif
#ifdef _OPENMP
void alcGP_omp(GP *gp, unsigned int ncand, double **Xcand, unsigned int nref,
     double **Xref,  int verb, double *alc);
void alcGP_omp_R(int *gpi_in, int *m_in, double *Xcand_in, int *ncand_in,
       double *Xref_in, int *nref_in, int *verb_in, double *alc_out);
#endif
void alcGP_R(int *gpi_in, int *m_in, double *Xcand_in, int *ncand_in, double *Xref_in,
       int *nref_in, int *verb_in, double *alc_out);
void d_ktKi(const int n, double **Ki, const int nn, double **dk, 
      double **dK, double **ktKi, double **dktKi);
void d2_ktKi(const int n, double **Ki, const int nn, double **d2k, 
         double **dK, double **d2K, double **ktKi, double **dktKi,
         double **d2ktKi);
void d_KiZ(const int n, double **Ki, double **dK, double *KiZ, 
     double *dKiZ);
void d2_KiZ(const int n, double **Ki, double **dK, double **d2K,
      double *KiZ, double *dKiZ, double *d2KiZ);
void dmus2GP(GP* gp, unsigned int nn, double **XX, double *mu, double *dmu, 
       double *d2mu, double *s2, double *ds2, double *d2s2);
void dmus2GP_R(int *gpi_in, int *m_in, int *nn_in, double *XX_in, double *mu_out,
        double *dmu_out, double *d2mu_out, double *s2_out, double *ds2_out,
        double *d2s2_out);
void efiGP(GP* gp, unsigned int nn, double **XX, double *efi);
void efiGP_R(int *gpi_in, int *m_in, int *nn_in, double *XX_in, double *efi_out);
void update_fishinfoGP(GP *gp, double *x, double z);
void mspeGP(GP *gp, unsigned int ncand, double **Xcand, unsigned int nref,
      double **Xref, int fi, int verb, double *mspe);
void mspeGP_R(int *gpi_in, int *m_in, double *Xcand_in, int *ncand_in, double *Xref_in,
       int *nref_in, int *fi_in, int *verb_in, double *mspe_out);

#endif

