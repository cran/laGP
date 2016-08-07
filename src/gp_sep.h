/****************************************************************************
 *
 * Local Approximate Gaussian Process Regression
 * Copyright (C) 2013, The University of Chicago
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
 * 02110-1301  USA
 *
 * Questions? Contact Robert B. Gramacy (rbg@vt.edu)
 *
 ****************************************************************************/


#ifndef __GP_SEP_H__
#define __GP_SEP_H__

typedef struct gpsep {
  double **X;       /* design matrix */
  double **K;       /* covariance between design points */
  double **Ki;      /* inverse of K */
  double ***dK;     /* gradient of K */
  double ldetK;     /* log of the determinant of K */
  double *Z;        /* response vector */
  double *KiZ;      /* Ki %*% Z */
  unsigned int m;   /* number of cols in X */
  unsigned int n;   /* number of rows in X; length of Z */
  double *d;        /* separable lengthscale parameter to correlation */
  double g;         /* nugget parameter to correlation */
  double phi;       /* t(Z) %*% Ki %*% Z = t(Z) %*% KiZ, used for s2 */
} GPsep;


GPsep* newGPsep(const unsigned int m, const unsigned int n, double **X,
    double *Z, double *d, const double g, const int dK);
GPsep* newGPsep_sub(const unsigned int m, const unsigned int n, int *p, 
    double **X, double *Z, double *d, const double g, const int dK);
void newGPsep_R(int *m_in, int *n_in, double *X_in, double *Z_in,
    double *d_in, double *g_in, int *dK_in, int *gp_index);
void updateGPsep(GPsep* gpsep, unsigned int nn, double **XX, double *ZZ, 
    int verb);
unsigned int get_gpsep(void);
void deletedKGPsep(GPsep *gpsep);
void deleteGPsep(GPsep* gpsep);
void deleteGPsep_index(unsigned int i);
void deleteGPsep_R(int *gpsep);
void deleteGPseps(void);
void deleteGPseps_R(void);
void calc_ZtKiZ_sep(GPsep *gpsep);
void newdKGPsep(GPsep *gpsep);
GPsep* buildGPsep(GPsep *gpsep, const int dK);
double llikGPsep(GPsep *gpsep, double *dab, double *gab);
void llikGPsep_R(int *gpsepi_in, double *dab_in, double *gab_in,
        double *llik_out);
void dllikGPsep(GPsep *gpsep, double *ab, double *dllik);
void dllikGPsep_nug(GPsep *gpsep, double *ab, double *dllik, double *d2llik);
void dllikGPsep_R(int *gpsepi_in, double *ab_in, double *dllik_out);
void dllikGPsep_nug_R(int *gpsepi_in, double *ab_in, double *dllik_out,
        double *d2llik_out);
void getmGPsep_R(int *gpsepi_in, int *m_out);
void getgGPsep_R(int *gpsepi_in, double *g_out);
void getdGPsep_R(int *gpsepi_in, double *d_out);
void newparamsGPsep(GPsep* gpsep, double *d, const double g);
void newparamsGPsep_R(int *gpsepi_in, double *d_in, double *g_in);
void jmleGPsep(GPsep *gpsep, int maxit, double *dmin, double *dmax, 
      double *grange, double *dab, double *gab, int verb, 
      int *dits, int *gits, int *dconv, int fromR);
void mleGPsep(GPsep* gpsep, double* dmin, double *dmax, double *ab, 
      const unsigned int maxit, int verb, double *p, int *its, 
      char *msg, int *conv, int fromR);
double mleGPsep_nug(GPsep* gpsep, double tmin, double tmax, double *ab, 
      int verb, int *its);
void mleGPsep_nug_R(int *gpsepi_in, int *verb_in, double *tmin_in,
       double *tmax_in, double *ab_in, double *mle_out, int *its_out);
void predGPsep(GPsep* gpsep, unsigned int nn, double **XX, double *mean, 
      double **Sigma, double *df, double *llik);
void new_predutilGPsep_lite(GPsep *gpsep, unsigned int nn, double **XX, 
      double ***k, double ***ktKi, double **ktKik);
void predGPsep_lite(GPsep* gpsep, unsigned int nn, double **XX, double *mean, 
     double *sigma2, double *df, double *llik);
void predGPsep_R(int *gpsepi_in, int *m_in, int *nn_in, double *XX_in,
        int *lite_in, double *mean_out, double *Sigma_out, double *df_out,
        double *llik_out);
void alcGPsep(GPsep *gpsep, unsigned int ncand, double **Xcand, 
        unsigned int nref, double **Xref,  int verb, double *alc);
void alcGP_R(int *gpsepi_in, int *m_in, double *Xcand_in, int *ncand_in, 
        double *Xref_in, int *nref_in, int *verb_in, double *alc_out);
#ifdef _OPENMP
void alcGPsep_omp(GPsep *gpsep, unsigned int ncand, double **Xcand, 
        unsigned int nref, double **Xref,  int verb, double *alc);
void alcGPsep_omp_R(int *gpsepi_in, int *m_in, double *Xcand_in, int *ncand_in,
       double *Xref_in, int *nref_in, int *verb_in, double *alc_out);
#endif
double* alcrayGPsep(GPsep *gpsep, double **Xref, const unsigned int nump, 
  double **Xstart, double **Xend, double *negalc, const unsigned int verb);
int lalcrayGPsep(GPsep *gpsep, double **Xcand, const unsigned int ncand, 
  double **Xref, const unsigned int offset, unsigned int nr, double **rect, 
  int verb);

#endif
