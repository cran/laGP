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
 * Questions? Contact Robert B. Gramacy (rbgramacy@chicagobooth.edu)
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

typedef enum THETA {LENGTHSCALE=3001, NUGGET=3002} Theta;

GPsep* newGPsep(const unsigned int m, const unsigned int n, double **X,
    double *Z, double *d, const double g);
void newGPsep_R(int *m_in, int *n_in, double *X_in, double *Z_in,
    double *d_in, double *g_in, int *gp_index);
unsigned int get_gpsep(void);
void deletedKGPsep(GPsep *gpsep);
void deleteGPsep(GPsep* gpsep);
void deleteGPsep_index(unsigned int i);
void deleteGPsep_R(int *gpsep);
void deleteGPseps(void);
void deleteGPseps_R(void);
void calc_ZtKiZ_sep(GPsep *gpsep);
void newdKGPsep(GPsep *gpsep);
GPsep* buildGPsep(GPsep *gpsep);
GPsep* newGPsep(const unsigned int m, const unsigned int n, double **X,
    double *Z, double *d, const double g);
void newGPsep_R(int *m_in, int *n_in, double *X_in, double *Z_in, 
  double *d_in, double *g_in, int *gpsep_index);
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
double mleGPsep_nug(GPsep* gpsep, double tmin, double tmax, double *ab, 
        int *its, int verb);
void mleGPsep_nug_R(int *gpsepi_in, int *verb_in, double *tmin_in,
       double *tmax_in, double *ab_in, double *mle_out, int *its_out);

#endif