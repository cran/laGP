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


#include "matrix.h"
#include "gp_sep.h"
#include "util.h"
#include "linalg.h"
#include "rhelp.h"
#include <assert.h>
#include <stdlib.h>
#include <math.h>
#ifdef RPRINT
#include <R.h>
#include <Rmath.h>
#endif
#include "covar_sep.h"
#include "ieci.h"
#include "gp.h"
#ifdef _OPENMP
  #include <omp.h>
#endif

#define SDEPS sqrt(DBL_EPSILON)

/*
 * Global variables used to accumulate data on the C-side
 * as passed from initialization and updtating R calls
 */

unsigned int NGPsep = 0;
GPsep **gpseps = NULL;

/*
 * get_gpsep:
 *
 * returns an integer reference to a free separable gp
 *
 */

unsigned int get_gpsep(void)
{
  unsigned int i;
  if(NGPsep == 0) {
    assert(gpseps == NULL);
    gpseps = (GPsep**) malloc(sizeof(GPsep*));
    gpseps[0] = NULL;
    NGPsep = 1;
    return 0;
  } else {
    for(i=0; i<NGPsep; i++) {
      if(gpseps[i] == NULL) return i;
    }
    gpseps = (GPsep**) realloc(gpseps, sizeof(GPsep*) * (2*NGPsep));
    for(i=NGPsep; i<2*NGPsep; i++) gpseps[i] = NULL;
    NGPsep *= 2;
    return NGPsep/2;
  }
}


/*
 * deletedKGPsep:
 *
 * delete the dK components of the gp
 *
 * SIMILAR to deleteGP but multiple dKs to delete
 */

void deletedKGPsep(GPsep *gpsep)
{
  unsigned int k;
  if(gpsep->dK) {
    for(k=0; k<gpsep->m; k++) {
      assert(gpsep->dK[k]);
      delete_matrix(gpsep->dK[k]);
    }
    free(gpsep->dK);
  }
}


/*
 * deletedKGPsep_R:
 *
 * R-interface to code destroying dK information
 * so that they are not updated in future calculations
 */

void deletedKGPsep_R(/* inputs */
      int *gpsepi_in)
{
  GPsep *gpsep;
  unsigned int gpsepi;

  /* get the cloud */
  gpsepi = *gpsepi_in;
  if(gpseps == NULL || gpsepi >= NGPsep || gpseps[gpsepi] == NULL)
    error("gpsep %d is not allocated\n", gpsepi);
  gpsep = gpseps[gpsepi];

  /* check if needed */
  if(! gpsep->dK) error("derivative info not in gpsep");

  /* call real C routine */
  deletedKGPsep(gpsep);
}


/*
 * deleteGPsep:
 *
 * free the memory allocated to a separable gp structure
 *
 * similar to deleteGP except loops over dK
 */


void deleteGPsep(GPsep* gpsep)
{
  assert(gpsep);
  assert(gpsep->X); delete_matrix(gpsep->X);
  assert(gpsep->Z); free(gpsep->Z);
  assert(gpsep->K); delete_matrix(gpsep->K);
  assert(gpsep->Ki); delete_matrix(gpsep->Ki);
  assert(gpsep->KiZ); free(gpsep->KiZ);
  deletedKGPsep(gpsep);
  assert(gpsep->d); free(gpsep->d);
  free(gpsep);
}


/*
 * deleteGPsep_index:
 *
 * delete the i-th gpsep
 *
 * SAME as deleteGP except with gpsep
 */

void deleteGPsep_index(unsigned int i)
{
  if(!(gpseps == NULL || i >= NGPsep || gpseps[i] == NULL)) {
    deleteGPsep(gpseps[i]);
    gpseps[i] = NULL;
  } else error("gpsep %d is not allocated\n", i);
}


/*
 * deleteGPsep_R:
 *
 * R-interface to deleteGPsep
 *
 * SAME as deleteGP_R except with gpsep
 */

void deleteGPsep_R(int *gpsep)
{
  deleteGPsep_index(*gpsep);
}


/*
 * deleteGPseps:
 *
 * delete all of the gpseps stored in
 * the array, and destroy the array
 *
 * SAME as deleteGPs except with gpsep
 */

void deleteGPseps(void)
{
  int i;
  for(i=0; i<NGPsep; i++) {
    if(gpseps[i]) {
      MYprintf(MYstdout, "removing gpsep %d\n", i);
      deleteGPsep(gpseps[i]);
    }
  }
  if(gpseps) free(gpseps);
  gpseps = NULL;
  NGPsep = 0;
}


/*
 * deleteGPseps_R:
 *
 * R interface to deleteGPseps
 *
 * SAME as deleteGPs_R except with gpsels
 */

void deleteGPseps_R(void)
{
  if(gpseps) deleteGPseps();
}


/*
 * calc_ZtKiZ_sep:
 *
 * re-calculates phi = ZtKiZ from Ki and Z stored in
 * the GP object; also update KiZ on which it depends
 *
 * SAME as gp.c but uses GPsep instead
 */

void calc_ZtKiZ_sep(GPsep *gpsep)
{
  assert(gpsep);
  /* phi <- t(Z) %*% Ki %*% Z */
  if(gpsep->KiZ == NULL) gpsep->KiZ = new_vector(gpsep->n);
  linalg_dsymv(gpsep->n,1.0,gpsep->Ki,gpsep->n,gpsep->Z,1,0.0,gpsep->KiZ,1);
  gpsep->phi = linalg_ddot(gpsep->n, gpsep->Z, 1, gpsep->KiZ, 1);
}


/*
 * newdKGPsep:
 *
 * allocate new space for dK and d2K calculations, and
 * calculate derivatives
 *
 * similar to newdKGP except no 2nd derivatives or fishinfo
 */

void newdKGPsep(GPsep *gpsep)
{
  unsigned int j;
  assert(gpsep->dK == NULL);
  gpsep->dK = (double ***) malloc(sizeof(double **) * gpsep->m);
  for(j=0; j<gpsep->m; j++) gpsep->dK[j] = new_matrix(gpsep->n, gpsep->n);
  diff_covar_sep_symm(gpsep->m, gpsep->X, gpsep->n, gpsep->d, gpsep->K,
    gpsep->dK);
}


/*
 * buildKGPsep_R:
 *
 * R-interface to code allocating dK information
 * for future calculations
 */

void buildKGPsep_R(/* inputs */
    int *gpsepi_in)
{
  GPsep *gpsep;
  unsigned int gpsepi;

  /* get the cloud */
  gpsepi = *gpsepi_in;
  if(gpseps == NULL || gpsepi >= NGPsep || gpseps[gpsepi] == NULL)
    error("gpsep %d is not allocated\n", gpsepi);
  gpsep = gpseps[gpsepi];

  /* check if needed */
  if(gpsep->dK) error("derivative info already in gpsep");

  /* call real C routine */
  newdKGPsep(gpsep);
}


/*
 * buildGPsep:
 *
 * intended for newly created separable GPs, e.g., via newGPsep
 * does all of the correlation calculations, etc., after data and
 * parameters are defined
 *
 * similar to buildGP except calculates gradient dK
 */

GPsep* buildGPsep(GPsep *gpsep, const int dK)
{
  double **Kchol, **X;
  unsigned int n, m;
  int info;

  assert(gpsep && gpsep->K == NULL);
  n = gpsep->n;
  m = gpsep->m;
  X = gpsep->X;

  /* build covari ance matrix */
  gpsep->K = new_matrix(n, n);
  covar_sep_symm(m, X, n, gpsep->d, gpsep->g, gpsep->K);

  /* invert covariance matrix */
  gpsep->Ki = new_id_matrix(n);
  Kchol = new_dup_matrix(gpsep->K, n, n);
  info = linalg_dposv(n, Kchol, gpsep->Ki);
  if(info) {
#ifdef UNDEBUG
    printMatrix(gpsep->K, n, n, stdout);
#endif
    MYprintf(MYstdout, "d = ");
    printVector(gpsep->d, m, MYstdout, HUMAN);
    error("bad Cholesky decomp (info=%d), g=%g",
          info, gpsep->g);
  }
  gpsep->ldetK = log_determinant_chol(Kchol, n);
  delete_matrix(Kchol);

  /* phi <- t(Z) %*% Ki %*% Z */
  gpsep->KiZ = NULL;
  calc_ZtKiZ_sep(gpsep);

  /* calculate derivatives ? */
  gpsep->dK = NULL;
  if(dK) newdKGPsep(gpsep);

  /* return new structure */
  return(gpsep);
}


/*
 * newGPsep:
 *
 * allocate a new separable GP structure using the data and parameters
 * provided
 *
 * similar to  newGP except gpseps and pointer to d, and does not have dK
 * flag since gradient is always calculated
 */

GPsep* newGPsep(const unsigned int m, const unsigned int n, double **X,
	  double *Z, double *d, const double g, const int dK)
{
  GPsep* gpsep;

  /* new gp structure */
  gpsep = (GPsep*) malloc(sizeof(GPsep));
  gpsep->m = m;
  gpsep->n = n;
  gpsep->X = new_dup_matrix(X, n, m);
  gpsep->Z = new_dup_vector(Z, n);
  gpsep->d = new_dup_vector(d, m);
  gpsep->g = g;
  gpsep->K = NULL;
  gpsep->dK = NULL;

  return buildGPsep(gpsep, dK);
}


/*
 * newGPsep_sub:
 *
 * allocate a new GPsep structure using the parameters
 * provided, and the subset (rows) of the data specified by p
 */

GPsep* newGPsep_sub(const unsigned int m, const unsigned int n, int *p,
        double **X, double *Z, double *d, const double g, const int dK)
{
  unsigned int i;
  GPsep* gpsep;

  /* new gp structure */
  gpsep = (GPsep*) malloc(sizeof(GPsep));
  gpsep->m = m;
  gpsep->n = n;
  gpsep-> X = new_p_submatrix_rows(p, X, n, gpsep->m, 0);
  gpsep->Z = new_vector(n);
  for(i=0; i<n; i++) gpsep->Z[i] = Z[p[i]];
  gpsep->d = new_dup_vector(d, m);
  gpsep->g = g;
  gpsep->K = NULL;
  gpsep->dK = NULL;

  return buildGPsep(gpsep, dK);
}


/*
 * newGPsep_R:
 *
 * R-interface initializing a new separable GP, allocating and
 * assigning values to the global variables, which are
 * written over if already in use
 *
 * similar to newGP_R except takes vector d_in and does not have dK flag
 * since the gradient is always calculated
 */

void newGPsep_R(/* inputs */
       int *m_in,
       int *n_in,
       double *X_in,
       double *Z_in,
       double *d_in,
       double *g_in,
       int *dK_in,

       /* outputs */
       int *gpsep_index)
{
  double **X;

  /* assign a new gp index */
  *gpsep_index = get_gpsep();

  /* create a new GP; */
  X = new_matrix_bones(X_in, *n_in, *m_in);
  gpseps[*gpsep_index] = newGPsep(*m_in, *n_in, X, Z_in, d_in, *g_in, *dK_in);
  free(X);
}


/*
 * llikGPsep:
 *
 * calculate and return the log marginal likelihood
 *
 * similar to llikGPsep except loops over separable d for
 * prior calculation
 */

double llikGPsep(GPsep *gpsep, double *dab, double *gab)
{
  unsigned int k;
  double llik;

  /* proportional to likelihood calculation */
  llik = 0.0 - 0.5*(((double) gpsep->n) * log(0.5 * gpsep->phi) + gpsep->ldetK);
  // MYprintf(MYstdout, "d=%g, g=%g, phi=%g, llik=%g\n", gpsep->d, gpsep->g, gpsep->phi, llik);
  /* llik += lgamma(0.5*((double) gpsep->n)) - ((double) gpsep->n)*M_LN_SQRT_2PI; */

  /* if priors are being used; for lengthscale */
  if(dab && dab[0] > 0 && dab[1] > 0) {
    for(k=0; k<gpsep->m; k++) {
      if(gpsep->d[k] > 0) llik += dgamma(gpsep->d[k], dab[0], 1.0/dab[1], 1);
    }
  }

  /* if priors are being used; for nugget */
  if(gpsep->g > 0 && gab && gab[0] > 0 && gab[1] > 0)
    llik += dgamma(gpsep->g, gab[0], 1.0/gab[1], 1);

  return(llik);
}


/*
 * llikGPsep_R:
 *
 * R-interface to calculate the marginal likelihood of a
 * separable GP
 *
 * SAME to llikGP except with gpsep
 */

void llikGPsep_R(/* inputs */
        int *gpsepi_in,
        double *dab_in,
        double *gab_in,

        /* outputs */
        double *llik_out)
{
  GPsep *gpsep;
  unsigned int gpsepi;

  /* get the cloud */
  gpsepi = *gpsepi_in;
  if(gpseps == NULL || gpsepi >= NGPsep || gpseps[gpsepi] == NULL)
    error("gpsep %d is not allocated\n", gpsepi);
  gpsep = gpseps[gpsepi];

  /* calculate log likelihood */
  *llik_out = llikGPsep(gpsep, dab_in, gab_in);
}


/*
 * dllikGPsep:
 *
 * batch calculation of the gradient of the log likelihood
 * of a separable gp, with respect to the
 * lengthscale parameter, d; requires that derivatives
 * be pre-calculated
 *
 * substantially changed from dllikGP and removed d2llik
 */

void dllikGPsep(GPsep *gpsep, double *ab, double *dllik)
{
  double *KiZtwo;
  unsigned int i, j, n, k;
  double dn, phirat ;

  /* sanity check */
  assert(gpsep->dK);
  assert(dllik);

  /* copy dims for fast access */
  n = gpsep->n;
  dn = (double) n;

  KiZtwo = new_vector(n);
  for(k=0; k<gpsep->m; k++) {

    /* deal with possible prior */
    if(ab && ab[0] > 0 && ab[1] > 0) {
      dllik[k] = (ab[0] - 1.0)/gpsep->d[k] - ab[1];
    } else dllik[k] = 0.0;

    /* dllik = - 0.5 * tr(Ki %*% dK) */
    for(i=0; i<n; i++) {
      for(j=0; j<i; j++) /* off diagonal */
        dllik[k] -= gpsep->Ki[i][j] * gpsep->dK[k][i][j];

      /* on-diagonal */
      dllik[k] -= 0.5 * gpsep->Ki[i][i] * gpsep->dK[k][i][i];
    }

    /* now third part of the expression, re-using KiZtwo */
    /* KiZtwo = dK %*% KiZ */
    linalg_dsymv(n,1.0,gpsep->dK[k],n,gpsep->KiZ,1,0.0,KiZtwo,1);
    /* now t(KiZ) %*% dK %*% KiZ */
    phirat = linalg_ddot(n, gpsep->KiZ, 1, KiZtwo, 1) / gpsep->phi;
    dllik[k] += 0.5*dn*phirat;
  }

  /* clean up */
  free(KiZtwo);
}


/*
 * dllikGPsep_nug:
 *
 * batch calculation of the first derivative
 * of the log likelihood of a gp, with respect to the
 * NUGGET parameter, g
 *
 */

void dllikGPsep_nug(GPsep *gpsep, double *ab, double *dllik, double *d2llik)
{
  unsigned int i, j, n;
  double *KiZtwo;
  double **two, **dKKidK;
  double dn, phirat, dlp, d2lp;

  /* sanity check */
  assert(dllik);

  /* deal with possible prior */
  if(ab && ab[0] > 0 && ab[1] > 0) {
    dlp = (ab[0] - 1.0)/gpsep->g - ab[1];
    d2lp = 0.0 - (ab[0] - 1.0)/sq(gpsep->g);
  } else dlp = d2lp = 0.0;

  /* copy dims for fast access */
  n = gpsep->n;
  dn = (double) n;

  if(d2llik) {
    two = new_matrix(n, n);
    dKKidK = gpsep->Ki;
  } else two = dKKidK = NULL;

  /* d2llik = - 0.5 * tr(Ki %*% [0.0 - Ki]); the first expression */
  /* dllik = - 0.5 * tr(Ki) */
  if(d2llik) *d2llik = d2lp;
  *dllik = dlp;
  for(i=0; i<n; i++) {
    if(d2llik) {
      for(j=0; j<i; j++) { /* off diagonal */
        *d2llik += gpsep->Ki[i][j] * dKKidK[i][j];
        two[i][j] = two[j][i] = 2.0*dKKidK[i][j];
      }
    }
    /* on-diagonal */
    *dllik -= 0.5 * gpsep->Ki[i][i];
    if(d2llik) {
      *d2llik += 0.5 * gpsep->Ki[i][i] * dKKidK[i][i];
      two[i][i] = 2.0*dKKidK[i][i];
    }
  }

  /* now the second part of the expression: */
  /* d2llik -= 0.5 * KiZ %*% two %*% KiZ */
  if(d2llik) {
    KiZtwo = new_vector(n);
    linalg_dsymv(n,1.0,two,n,gpsep->KiZ,1,0.0,KiZtwo,1);
    *d2llik -= 0.5*dn*linalg_ddot(n, gpsep->KiZ, 1, KiZtwo, 1) / gpsep->phi;
    free(KiZtwo);
  }

  /* now third part of the expression, re-using KiZtwo */
  /* now t(KiZ) %*% dK %*% KiZ */
  phirat = linalg_ddot(n, gpsep->KiZ, 1, gpsep->KiZ, 1) / gpsep->phi;
  if(d2llik) *d2llik += 0.5*dn*sq(phirat);
  *dllik += 0.5*dn*phirat;

  /* clean up */
  if(two) delete_matrix(two);
}


/*
 * dllikGPsep_R:
 *
 * R-interface to calculate the derivatives of the
 * likelihood of a GP - wrt lengthscale
 *
 * SIMILAR to dllikGP_R except returns vectorized dllik_out
 * and does not have d2llik_out
 */

void dllikGPsep_R(/* inputs */
               int *gpsepi_in,
               double *ab_in,

               /* outputs */
               double *dllik_out)
{
  GPsep *gpsep;
  unsigned int gpsepi;

  /* get the cloud */
  gpsepi = *gpsepi_in;
  if(gpseps == NULL || gpsepi >= NGPsep || gpseps[gpsepi] == NULL)
    error("gpsep %d is not allocated\n", gpsepi);
  gpsep = gpseps[gpsepi];

  /* double check that derivatives have been calculated */
  if(! gpsep->dK)
    error("derivative info not in gpsep; use newGPsep with dK=TRUE");

  /* calculate log likelihood */
  dllikGPsep(gpsep, ab_in, dllik_out);
}


/*
 * dllikGPsep_nug_R:
 *
 * R-interface to calculate the derivatives of the
 * likelihood of a GP - wrt the NUGGET
 *
 * SIMILAR to dllikGP_nug_R except without d2llik_out
 */

void dllikGPsep_nug_R(/* inputs */
               int *gpsepi_in,
               double *ab_in,

               /* outputs */
               double *dllik_out,
               double *d2llik_out)
{
  GPsep *gpsep;
  double *d2llik;
  unsigned int gpsepi;

  /* get the cloud */
  gpsepi = *gpsepi_in;
  if(gpseps == NULL || gpsepi >= NGPsep || gpseps[gpsepi] == NULL)
    error("gpsep %d is not allocated\n", gpsepi);
  gpsep = gpseps[gpsepi];


  /* check to see if we want 2nd derivative or not */
  if(d2llik_out[0] == 1) d2llik = d2llik_out;
  else d2llik = NULL;

  /* calculate log likelihood */
  dllikGPsep_nug(gpsep, ab_in, dllik_out, d2llik);
}


/*
 * getmGPsep_R:
 *
 * R-interface accessing the input dimension m
 * of a GPsep
 */

void getmGPsep_R(/* inputs */
              int *gpsepi_in,
               /* outputs */
              int *m_out)
{
  GPsep *gpsep;
  unsigned int gpsepi;

  /* get the cloud */
  gpsepi = *gpsepi_in;
  if(gpseps == NULL || gpsepi >= NGPsep || gpseps[gpsepi] == NULL)
    error("gpsep %d is not allocated\n", gpsepi);
  gpsep = gpseps[gpsepi];

  *m_out = gpsep->m;
}


/*
 * getgGPsep_R:
 *
 * R-interface accessing the snugget of a separable
 * of a GP
 */


void getgGPsep_R(/* inputs */
              int *gpsepi_in,
               /* outputs */
              double *g_out)
{
  GPsep *gpsep;
  unsigned int gpsepi;

  /* get the cloud */
  gpsepi = *gpsepi_in;
  if(gpseps == NULL || gpsepi >= NGPsep || gpseps[gpsepi] == NULL)
    error("gpsep %d is not allocated\n", gpsepi);
  gpsep = gpseps[gpsepi];

  *g_out = gpsep->g;
}

/*
 * getdGPsep_R:
 *
 * R-interface accessing the separable lengthscale parameter
 * of a GP
 */

void getdGPsep_R(/* inputs */
              int *gpsepi_in,
               /* outputs */
              double *d_out)
{
  GPsep *gpsep;
  unsigned int gpsepi;

  /* get the cloud */
  gpsepi = *gpsepi_in;
  if(gpseps == NULL || gpsepi >= NGPsep || gpseps[gpsepi] == NULL)
    error("gpsep %d is not allocated\n", gpsepi);
  gpsep = gpseps[gpsepi];

  /* double check that derivatives have been calculated */
  dupv(d_out, gpsep->d, gpsep->m);
}


/*
 * newparamsGPsep:
 *
 * change the lengthscale and nugget parameters to the gp
 *
 * SIMIAR to newparamsGP except vectorized d and always does
 * gradient
 */

void newparamsGPsep(GPsep* gpsep, double *d, const double g)
{
  int info, m, n;
  double **Kchol;

  /* sanity check */
  assert(g >= 0);

  /* build covariance matrix */
  m = gpsep->m; n = gpsep->n;
  dupv(gpsep->d, d, m);
  gpsep->g = g;
  covar_sep_symm(m, gpsep->X, n, gpsep->d, gpsep->g, gpsep->K);

  /* invert covariance matrix */
  id(gpsep->Ki, n);
  Kchol = new_dup_matrix(gpsep->K, n, n);
  info = linalg_dposv(n, Kchol, gpsep->Ki);
  if(info) {
#ifdef UNDEBUG
    printMatrix(gpsep->K, n, n, stdout);
#endif
    MYprintf(MYstdout, "d =");
    printVector(gpsep->d, m, MYstdout, HUMAN);
    error("bad Cholesky decomp (info=%d), g=%g", info, g);
  }
  gpsep->ldetK = log_determinant_chol(Kchol, n);
  delete_matrix(Kchol);

  /* phi <- t(Z) %*% Ki %*% Z */
  calc_ZtKiZ_sep(gpsep);

  /* calculate derivatives ? */
  if(gpsep->dK)
    diff_covar_sep_symm(gpsep->m, gpsep->X, gpsep->n, gpsep->d,
      gpsep->K, gpsep->dK);
}


/*
 * newparamsGPsep_R:
 *
 * R-interface allowing the internal/global separable GP representation
 * to change its parameterization without destroying the
 * memory and then re-allocating it
 */

void newparamsGPsep_R(/* inputs */
    int *gpsepi_in,
    double *d_in,
    double *g_in)
{
  GPsep *gpsep;
  unsigned int gpsepi, k;
  int dsame;

  /* get the cloud */
  gpsepi = *gpsepi_in;
  if(gpseps == NULL || gpsepi >= NGPsep || gpseps[gpsepi] == NULL)
    error("gpsep %d is not allocated\n", gpsepi);
  gpsep = gpseps[gpsepi];

  /* check if any are old */
  dsame = 1;
  for(k=0; k<gpsep->m; k++) {
    if(d_in[k] <= 0) d_in[k] = gpsep->d[k];
    else if(d_in[k] != gpsep->d[k]) dsame = 0;
  }
  if(*g_in < 0) *g_in = gpsep->g;

  /* check if there is nothing to do bc the params are the same */
  if(dsame && *g_in == gpsep->g) return;

  /* call real C routine */
  newparamsGPsep(gpsep, d_in, *g_in);
}


/*
 * utility structure for fcn_nllik_sep and fcn_ndllik_sep defined below
 * for use with lbfgsb (R's optim with "L-BFGS-B" method)
 * for optimization over the lengthscale parameter only
 */

struct callinfo_sep {
  GPsep *gpsep;
  double *dab;
  double *gab;
  int its;  /* updated but not used since lbfgsb counts fmin and gr evals */
  int verb;
};



/*
 * fcn_nllik_sep:
 *
 * a utility function for lbfgsb (R's optim with "L-BFGS-B" method) to
 * evaluating the separable GP log likelihood after changes to the
 * lengthscale parameter
 */

static double fcn_nllik_sep(int n, double *p, struct callinfo_sep *info)
{
  double llik;
  int psame, k, m;

  /* sanity check */
  m = info->gpsep->m;
  assert(n == m || n == m + 1);
  if(n == m+1) assert(info->gpsep->gab != NULL);
  else assert(info->gpsep->gab == NULL);

  /* check if parameters in p are new */
  psame = 1;
  for(k=0; k<n; k++) {
    if(k < m && p[k] != info->gpsep->d[k]) { psame = 0; break; }
    else if(k >= m && p[k] != info->gpsep->g) { psame = 0; break; }
  }

  /* update GP with new parameters */ 
  if(!psame) {
    (info->its)++;
    if(n == m) newparamsGPsep(info->gpsep, p, info->gpsep->g);
    else newparamsGPsep(info->gpsep, p, p[m]);
  }

  /* evaluate likelihood with potentially new paramterization */
  llik = llikGPsep(info->gpsep, info->dab, info->gab);

  /* progress meter */
  if(info->verb > 0) {
    MYprintf(MYstdout, "fmin it=%d, d=(%g", info->its, info->gpsep->d[0]);
    for(k=1; k<m; k++) MYprintf(MYstdout, " %g", info->gpsep->d[k]);
    if(n == m) MYprintf(MYstdout, "), llik=%g\n", llik);
    else MYprintf(MYstdout, "), g=%g, llik=%g\n", info->gpsep->g, llik);
  }

  /* done */
  return 0.0-llik;
}


/*
 * fcn_ngradllik_sep:
 *
 * a utility function for lbfgsb (R's optim with "L-BFGS-B" method)
 * evaluating the derivative of separable GP log likelihood after
 * changes to the lengthscale and nugget parameter
 */

static void fcn_ngradllik_sep(int n, double *p, double *df, struct callinfo_sep *info)
{
  int psame, k, m;

  /* sanity check */
  m = info->gpsep->m;
  assert(n == m + 1);

  /* check if parameters in p are new */
  psame = 1;
  for(k=0; k<m; k++) if(p[k] != info->gpsep->d[k]) { psame = 0; break; }
  if(psame) if(p[m] != info->gpsep->g) { psame = 0; }

  /* update GP with new parameters */
  if(!psame) error("ngradllik_sep incorrectly assumed grad comes after obj");
  /*
  if(!psame) {
    (info->its)++;
    newparamsGPsep(info->gpsep, p, p[m]);
  } */

  /* evaluate likelihood with potentially new paramterization */
  dllikGPsep(info->gpsep, info->dab, df);
  dllikGPsep_nug(info->gpsep, info->gab, df+m, NULL);

  /* negate values */
  for(k=0; k<n; k++) df[k] = 0.0-df[k];

  /* progress meter */
  if(info->verb > 1) {
    MYprintf(MYstdout, "grad it=%d, d=(%g", info->its, info->gpsep->d[0]);
    for(k=1; k<m; k++) MYprintf(MYstdout, " %g", info->gpsep->d[k]);
    MYprintf(MYstdout, "), g=%g, dd=(%g", info->gpsep->g, df[0]);
    for(k=1; k<m; k++) MYprintf(MYstdout, " %g", df[k]);
    MYprintf(MYstdout, "), dg=%g\n", df[m]);
  }
}


/*
 * fcn_ndllik_sep:
 *
 * a utility function for lbfgsb (R's optim with "L-BFGS-B" method)
 * evaluating the derivative of separable GP log likelihood after
 * changes to the lengthscale parameter
 */

static void fcn_ndllik_sep(int n, double *p, double *df, struct callinfo_sep *info)
{
  int dsame, k;

  /* sanity check */
  assert(n == info->gpsep->m);

  /* check if parameters in p are new */
  dsame = 1;
  for(k=0; k<n; k++) if(p[k] != info->gpsep->d[k]) { dsame = 0; break; }

  /* update GP with new parameters */
  if(!dsame) {
    (info->its)++;
    newparamsGPsep(info->gpsep, p, info->gpsep->g);
  }

  /* evaluate likelihood with potentially new paramterization */
  dllikGPsep(info->gpsep, info->dab, df);

  /* negate values */
  for(k=0; k<n; k++) df[k] = 0.0-df[k];

  /* progress meter */
  if(info->verb > 1) {
    MYprintf(MYstdout, "grad it=%d, d=(%g", info->its, info->gpsep->d[0]);
    for(k=1; k<n; k++) MYprintf(MYstdout, " %g", info->gpsep->d[k]);
    MYprintf(MYstdout, "), dd=(%g", df[0]);
    for(k=1; k<n; k++) MYprintf(MYstdout, " %g", df[k]);
    MYprintf(MYstdout, ")\n");
  }
}


/*
 * mleGPsep_both:
 *
 * update the separable GP to use its MLE separable
 * lengthscale and multiple nugget parameterization using the current data,
 * via the lbfgsb function
 */

void mleGPsep_both(GPsep* gpsep, double* tmin, double *tmax, double *ab,
  const unsigned int maxit, int verb, double *p, int *its, char *msg,
  int *conv, int fromR)
{
  double rmse;
  int k, lbfgs_verb;
  double *told;

  /* create structure for lbfgsb */
  struct callinfo_sep info;
  info.gpsep = gpsep;
  info.dab = ab;
  info.gab = ab+2;
  info.its = 0;
  info.verb = verb-6;

  /* copy the starting value */
  dupv(p, gpsep->d, gpsep->m);
  p[gpsep->m] = gpsep->g;
  told = new_dup_vector(p, gpsep->m + 1);

  if(verb > 0) {
    MYprintf(MYstdout, "(theta=[%g", p[0]);
    for(k=1; k<gpsep->m+1; k++) MYprintf(MYstdout, ",%g", p[k]);
    MYprintf(MYstdout, "], llik=%g) ", llikGPsep(gpsep, ab, ab+2));
  }

  /* set ifail argument and verb/trace arguments */
  *conv = 0;
  if(verb <= 1) lbfgs_verb = 0;
  else lbfgs_verb = verb - 1;

  /* call the C-routine behind R's optim function with method = "L-BFGS-B" */
  MYlbfgsb(gpsep->m+1, p, tmin, tmax,
         (double (*)(int, double*, void*)) fcn_nllik_sep,
         (void (*)(int, double *, double *, void *)) fcn_ngradllik_sep,
         conv, &info, 0, its, maxit, msg, lbfgs_verb, fromR);

  /* check if parameters in p are new */
  rmse = 0.0;
  for(k=0; k<gpsep->m; k++) rmse += sq(p[k] - gpsep->d[k]);
  if(sqrt(rmse/((double) gpsep->m)) > SDEPS) warning("stored d not same as d-hat");
  rmse = fabs(p[gpsep->m] - gpsep->g);
  if(rmse > SDEPS) warning("stored g not same as g-hat");
  rmse = 0.0;
  for(k=0; k<gpsep->m+1; k++) rmse += sq(p[k] - told[k]);
  if(sqrt(rmse/((double) (gpsep->m + 1))) < SDEPS) {
    snprintf(msg, 28, "lbfgs initialized at minima");
    *conv = 0;
    its[0] = its[1] = 0;
  }

  /* print progress */
  if(verb > 0) {
    MYprintf(MYstdout, "-> %d lbfgsb its -> (theta=[%g", its[1], p[0]);
    for(k=1; k<gpsep->m+1; k++) MYprintf(MYstdout, ",%g", p[k]);
    MYprintf(MYstdout, "], llik=%g)\n", llikGPsep(gpsep, ab, ab+2));
  }

  /* clean up */
  free(told);
}


/*
 * mleGPsep_both_R:
 *
 * R-interface to update the separable GP to use its MLE
 * separable lengthscale and multiple nugget
 * parameterization using the current data
 */

void mleGPsep_both_R(/* inputs */
       int *gpsepi_in,
       int *maxit_in,
       int *verb_in,
       double *tmin_in,
       double *tmax_in,
       double *ab_in,

       /* outputs */
       double *mle_out,
       int *its_out,
       char **msg_out,
       int *conv_out)
{
  GPsep *gpsep;
  unsigned int gpsepi, j;

  /* get the cloud */
  gpsepi = *gpsepi_in;
  if(gpseps == NULL || gpsepi >= NGPsep || gpseps[gpsepi] == NULL)
    error("gpsep %d is not allocated\n", gpsepi);
  gpsep = gpseps[gpsepi];

  /* check d against tmax and tmin */
  for(j=0; j<gpsep->m; j++) {
    if(tmin_in[j] <= 0) tmin_in[j] = SDEPS;
    if(tmax_in[j] <= 0) tmax_in[j] = sq((double) gpsep->m);
    if(gpsep->d[j] > tmax_in[j])
      error("d[%d]=%g > tmax[%d]=%g\n", j, gpsep->d[j], j, tmax_in[j]);
    else if(gpsep->d[j] < tmin_in[j])
      error("d[%d]=%g < tmin[%d]=%g\n", j, gpsep->d[j], j, tmin_in[j]);
  }

  /* check g and tmax */
  if(tmin_in[gpsep->m] <= 0) tmin_in[gpsep->m] = SDEPS;
  if(gpsep->g >= tmax_in[gpsep->m]) error("g=%g >= tmax=%g\n", gpsep->g, tmax_in[gpsep->m]);
  else if(gpsep->g <= tmin_in[gpsep->m]) error("g=%g <= tmin=%g\n", gpsep->g, tmin_in[gpsep->m]);

  /* check a & b */
  if(ab_in[0] < 0 || ab_in[1] < 0 || ab_in[2] < 0 || ab_in[3] < 0)
    error("ab_in must be a positive 4-vector");

  /* double check that derivatives have been calculated */
  if(!gpsep->dK)
    error("derivative info not in gpsep; use newGPsep with dK=TRUE");

  /* call C-side MLE */
  mleGPsep_both(gpsep, tmin_in, tmax_in, ab_in, *maxit_in, *verb_in, mle_out,
            its_out, *msg_out, conv_out, 1);
}


/*
 * mleGPsep:
 *
 * update the separable GP to use its MLE separable
 * lengthscale parameterization using the current data,
 * via the lbfgsb function
 *
 */

void mleGPsep(GPsep* gpsep, double* dmin, double *dmax, double *ab,
  const unsigned int maxit, int verb, double *p, int *its, char *msg,
  int *conv, int fromR)
{
  double rmse;
  int k, lbfgs_verb;
  double *dold;

  /* create structure for Brent_fmin */
  struct callinfo_sep info;
  info.gpsep = gpsep;
  info.dab = ab;
  info.gab = NULL;
  info.its = 0;
  info.verb = verb-6;

  /* copy the starting value */
  dupv(p, gpsep->d, gpsep->m);
  dold = new_dup_vector(gpsep->d, gpsep->m);

  if(verb > 0) {
    MYprintf(MYstdout, "(d=[%g", gpsep->d[0]);
    for(k=1; k<gpsep->m; k++) MYprintf(MYstdout, ",%g", gpsep->d[k]);
    MYprintf(MYstdout, "], llik=%g) ", llikGPsep(gpsep, ab, NULL));
  }

  /* set ifail argument and verb/trace arguments */
  *conv = 0;
  if(verb <= 1) lbfgs_verb = 0;
  else lbfgs_verb = verb - 1;

  /* call the C-routine behind R's optim function with method = "L-BFGS-B" */
  MYlbfgsb(gpsep->m, p, dmin, dmax,
         (double (*)(int, double*, void*)) fcn_nllik_sep,
         (void (*)(int, double *, double *, void *)) fcn_ndllik_sep,
         conv, &info, 0, its, maxit, msg, lbfgs_verb, fromR);

  /* check if parameters in p are new */
  rmse = 0.0;
  for(k=0; k<gpsep->m; k++) rmse += sq(p[k] - gpsep->d[k]);
  if(sqrt(rmse/k) > SDEPS) warning("stored d not same as d-hat");
  rmse = 0.0;
  for(k=0; k<gpsep->m; k++) rmse += sq(p[k] - dold[k]);
  if(sqrt(rmse/k) < SDEPS) {
    snprintf(msg, 28, "lbfgs initialized at minima");
    *conv = 0;
    its[0] = its[1] = 0;
  }

  /* print progress */
  if(verb > 0) {
    MYprintf(MYstdout, "-> %d lbfgsb its -> (d=[%g", its[1], gpsep->d[0]);
    for(k=1; k<gpsep->m; k++) MYprintf(MYstdout, ",%g", gpsep->d[k]);
    MYprintf(MYstdout, "], llik=%g)\n", llikGPsep(gpsep, ab, NULL));
  }

  /* clean up */
  free(dold);
}


/*
 * mleGPsep_R:
 *
 * R-interface to update the separable GP to use its MLE
 * separable lengthscale parameterization using the current data
 *
 * SIMPLIFIED compared to mleGPsep_R since only the (separable)
 * lengthscale is supported; for the nugget see mleGPsep_nug_R
 */

void mleGPsep_R(/* inputs */
       int *gpsepi_in,
       int *maxit_in,
       int *verb_in,
       double *dmin_in,
       double *dmax_in,
       double *ab_in,

       /* outputs */
       double *mle_out,
       int *its_out,
       char **msg_out,
       int *conv_out)
{
  GPsep *gpsep;
  unsigned int gpsepi, j;

  /* get the cloud */
  gpsepi = *gpsepi_in;
  if(gpseps == NULL || gpsepi >= NGPsep || gpseps[gpsepi] == NULL)
    error("gpsep %d is not allocated\n", gpsepi);
  gpsep = gpseps[gpsepi];

  /* check d against dmax and dmin */
  for(j=0; j<gpsep->m; j++) {
    if(dmin_in[j] <= 0) dmin_in[j] = SDEPS;
    if(dmax_in[j] <= 0) dmax_in[j] = sq((double) gpsep->m);
    if(gpsep->d[j] > dmax_in[j])
      error("d[%d]=%g > dmax[%d]=%g\n", j, gpsep->d[j], j, dmax_in[j]);
    else if(gpsep->d[j] < dmin_in[j])
      error("d[%d]=%g < dmin[%d]=%g\n", j, gpsep->d[j], j, dmin_in[j]);
  }

  /* check a & b */
  if(ab_in[0] < 0 || ab_in[1] < 0) error("ab_in must be a positive 2-vector");

  /* double check that derivatives have been calculated */
  if(!gpsep->dK)
    error("derivative info not in gpsep; use newGPsep with dK=TRUE");

  /* call C-side MLE */
  /* dupv(mle_out, gpsep->d, gpsep->m); */ /* duplicated first thing in mleGPsep */
  mleGPsep(gpsep, dmin_in, dmax_in, ab_in, *maxit_in, *verb_in, mle_out,
           its_out, *msg_out, conv_out, 1);
}


/*
 * utility structure for fcn_nllik_sep_nug defined below
 * for use with Brent_fmin (R's optimize) or uniroot
 *
 * SIMPLIFIED compared to callinfo in gp.c because it only does the nugget
 */

struct callinfo_sep_nug {
  GPsep *gpsep;
  double *ab;
  int its;
  int verb;
};


/*
 * fcn_nllik_sep_nug:
 *
 * a utility function for Brent_fmin (R's optimize) to apply to the separable
 * GP log likelihood after changes to the nugget parameter
 *
 * SIMPLIFIED compared to fcn_nllik in gp.c since it only does the nugget
 */

static double fcn_nllik_sep_nug(double x, struct callinfo_sep_nug *info)
{
  double llik;
  (info->its)++;
  newparamsGPsep(info->gpsep, info->gpsep->d, x);
  llik = llikGPsep(info->gpsep, NULL, info->ab);
  if(info->verb > 1)
    MYprintf(MYstdout, "fmin it=%d, g=%g, llik=%g\n", info->its, info->gpsep->g, llik);
  return 0.0-llik;
}


/*
 * Ropt_sep_nug:
 *
 * use R's Brent Fmin routine (from optimize) to optimize
 *
 * SIMPLIFIED compared to Ropt in GP because it only does the nugget
 */

double Ropt_sep_nug(GPsep* gpsep, double tmin, double tmax,
                   double *ab, char *msg, int *its, int verb)
{
  double tnew, th;
  double Tol = SDEPS;

  /* sanity check */
  assert(tmin < tmax);

  /* get parameter */
  th = gpsep->g;

  /* create structure for Brent_fmin */
  struct callinfo_sep_nug info;
  info.gpsep = gpsep;
  info.ab = ab;
  info.its = 0;
  info.verb = verb;

  /* call the C-routine behind R's optimize function */
  while(1) { /* check to make sure solution is not on boundary */
   tnew = Brent_fmin(tmin, tmax, (double (*)(double, void*)) fcn_nllik_sep_nug, &info, Tol);
   if(tnew > tmin && tnew < tmax) break;
   if(tnew == tmin) { /* left boundary found */
    tmin *= 2;
    if(verb > 0) MYprintf(MYstdout, "Ropt: tnew=tmin, increasing tmin=%g\n", tmin);
   } else { /* right boundary found */
    tmax /= 2.0;
    if(verb > 0) MYprintf(MYstdout, "Ropt: tnew=tmax, decreasing tmax=%g\n", tmax);
  }
  /* check that boundaries still valid */
  if(tmin >= tmax) error("unable to opimize in fmin()");
  }

  /* check that last value agrees with GP parameterization */
  if(gpsep->g != tnew) newparamsGPsep(gpsep, gpsep->d, tnew);

  /* possible print message and return */
  if(verb > 0) MYprintf(MYstdout, "Ropt %s: told=%g -[%d]-> tnew=%g\n",
      msg, th, info.its, tnew);

  *its += info.its;
  return(tnew);
}


/*
 * mleGPsep_nug:
 *
 * calculate the MLE with respect to the lengthscale parameter;
 * derivatives for the Newton method are calculated on the fly
 */

double mleGPsep_nug(GPsep* gpsep, double tmin, double tmax, double *ab,
             int verb, int *its)
{
  double tnew, dllik, d2llik, llik_init, llik_new, adj, rat;
  double th;
  double *gab, *dab;
  int restoredKGP;

  /* set priors based on Theta */
  dab = NULL;
  gab = ab;

  /* initialization */
  *its = 0;
  restoredKGP = 0;
  th = gpsep->g;

  /* check how close we are to tmin */
  if(fabs(th - tmin) < SDEPS) {
    if(verb > 0) MYprintf(MYstdout, "(g=%g) -- starting too close to min (%g)\n", th, tmin);
    goto alldone;
  }

  /* initial likelihood calculation */
  llik_init = llikGPsep(gpsep, dab, gab);

  /* initial printing */
  if(verb > 0)
      MYprintf(MYstdout, "(g=%g, llik=%g) ", gpsep->g, llik_init);
  if(verb > 1) MYprintf(MYstdout, "\n");

  while(1) { /* checking for improved llik */
    while(1) {  /* Newton step(s) */
      llik_new = R_NegInf;
      while(1) {  /* Newton proposal */

        /* calculate first and second derivatives */
        dllikGPsep_nug(gpsep, gab, &dllik, &d2llik);

        /* check for convergence by root */
        if(fabs(dllik) < SDEPS) {
          if(*its == 0) {
            if(verb > 0) MYprintf(MYstdout, "-- Newton not needed\n");
            goto alldone;
          } else goto newtondone;
        }

        /* Newton update */
        rat = dllik/d2llik; adj = 1.0; (*its)++;

        /* check if we're going the right way */
        if((dllik < 0 && rat < 0) || (dllik > 0 && rat > 0)) {
          if(!gpsep->dK && restoredKGP == 0) {
            deletedKGPsep(gpsep); restoredKGP = 1;
          }
          th = Ropt_sep_nug(gpsep, tmin, tmax, ab, "[slip]", its, verb); goto mledone;
        } else tnew = th - adj*rat;  /* right way: Newton: */

        /* check that we haven't proposed a tnew out of range */
        while((tnew <= tmin || tnew >= tmax) && adj > SDEPS) {
          adj /= 2.0; tnew = th - adj*rat;
        }

        /* if still out of range, restart? */
        if(tnew <= tmin || tnew >= tmax) {
          if(!gpsep->dK && restoredKGP == 0) {
            deletedKGPsep(gpsep); restoredKGP = 1;
          }
          th = Ropt_sep_nug(gpsep, tmin, tmax, ab, "[range]", its, verb);
          goto mledone;
        } else break;
      } /* end inner while -- Newton proposal */

      /* else, resets gpsep->g = tnew */
      if(!gpsep->dK && restoredKGP == 0) {
        deletedKGPsep(gpsep); restoredKGP = 1;
      }
      newparamsGPsep(gpsep, gpsep->d, tnew);

      /* print progress */
      if(verb > 1) MYprintf(MYstdout, "\ti=%d g=%g, c(a,b)=(%g,%g)\n",
                            *its, tnew, ab[0], ab[1]);

      /* check for convergence, and break or update */
      if(fabs(tnew - th) < SDEPS) break;
      else th = tnew;

      /* check for max its */
      if(*its >= 100) {
        if(verb > 0) warning("Newton 100/max iterations");
        /* could also call Ropt here as last resort */
       goto alldone;
      }
    } /* end middle while -- Newton step */

    /* sanity check check that we went in the right direction */
newtondone:
    llik_new = llikGPsep(gpsep, dab, gab);
    if(llik_new < llik_init-SDEPS) {
      if(verb > 0) MYprintf(MYstdout, "llik_new = %g\n", llik_new);
      llik_new = R_NegInf;
      if(!gpsep->dK && restoredKGP == 0) {
        deletedKGPsep(gpsep); restoredKGP = 1;
      }
      th = Ropt_sep_nug(gpsep, tmin, tmax, ab, "[dir]", its, verb);
      goto mledone;
    } else break;
  } /* outer improved llik check while(1) loop */

  /* capstone progress indicator */
mledone:
  if(!R_FINITE(llik_new)) llik_new = llikGPsep(gpsep, dab, gab);
  if(verb > 0) {
    MYprintf(MYstdout, "-> %d Newtons -> (g=%g, llik=%g)\n",
            *its, gpsep->g, llik_new);
  }

  /* return theta-value found */
alldone:
  if(restoredKGP) newdKGPsep(gpsep);
  return th;
}


/*
 * mleGPsep_nug_R:
 *
 * R-interface to update the separable GP to use its MLE
 * nugget parameterization using the current data
 *
 * SIMPLIFIED compared to mleGPsep_R since there is
 * no lengthscale option
 */

void mleGPsep_nug_R(/* inputs */
       int *gpsepi_in,
       int *verb_in,
       double *tmin_in,
       double *tmax_in,
       double *ab_in,

       /* outputs */
       double *mle_out,
       int *its_out)
{
  GPsep *gpsep;
  unsigned int gpsepi;

  /* get the cloud */
  gpsepi = *gpsepi_in;
  if(gpseps == NULL || gpsepi >= NGPsep || gpseps[gpsepi] == NULL)
    error("gpsep %d is not allocated\n", gpsepi);
  gpsep = gpseps[gpsepi];

  /* check theta and tmax */
  if(*tmin_in <= 0) *tmin_in = SDEPS;
  if(*tmax_in <= 0) *tmax_in = R_PosInf;
  if(gpsep->g >= *tmax_in) error("g=%g >= tmax=%g\n", gpsep->g, *tmax_in);
  else if(gpsep->g <= *tmin_in) error("g=%g <= tmin=%g\n", gpsep->g, *tmin_in);

  /* check a & b */
  if(ab_in[0] < 0 || ab_in[1] < 0) error("ab_in must be a positive 2-vector");

  /* call C-side MLE */
  *mle_out = mleGPsep_nug(gpsep, *tmin_in, *tmax_in, ab_in, *verb_in, its_out);
}


/*
 * jmleGPsep:
 *
 * calculate joint mle for separable lengthscale (d) and nugget (g)
 * by a coordinite-wise search, iterating over d and g searches via mleGPsep
 * and mleGPsep_nug
 */

void jmleGPsep(GPsep *gpsep, int maxit, double *dmin, double *dmax,
               double *grange, double *dab, double *gab, int verb,
               int *dits, int *gits, int *dconv, int fromR)
  {
    unsigned int i;
    int dit[2], git;
    char msg[60];
    double *d;

    /* sanity checks */
    assert(gab && dab);
    assert(dmin && dmax && grange);

    /* auxillary space for d-parameter values(s) */
    d = new_vector(gpsep->m);

    /* loop over coordinate-wise iterations */
    *dits = *gits = 0;
    for(i=0; i<100; i++) {
      mleGPsep(gpsep, dmin, dmax, dab, maxit, verb, d, dit, msg, dconv, fromR);
      if(dit[1] > dit[0]) dit[0] = dit[1];
      *dits += dit[0];
      mleGPsep_nug(gpsep, grange[0], grange[1], gab, verb, &git);
      *gits += git;
      if((git <= 2 && (dit[0] <= gpsep->m+1 && *dconv == 0)) || *dconv > 1) break;
    }
    if(i == 100 && verb > 0) warning("max outer its (N=100) reached");

    /* clean up */
    free(d);
  }


/*
 * jmleGP_R:
 *
 * R-interface to update the GP to use its joint MLE (lengthscale
 * and nugget) parameterization using the current data
 */

void jmleGPsep_R(/* inputs */
       int *gpsepi_in,
       int *maxit_in,
       int *verb_in,
       double *dmin_in,
       double *dmax_in,
       double *grange_in,
       double *dab_in,
       double *gab_in,

       /* outputs */
       double *d_out,
       double *g_out,
       int *dits_out,
       int *gits_out,
       int *dconv_out)
{
  GPsep *gpsep;
  unsigned int gpsepi, k;

  /* get the cloud */
  gpsepi = *gpsepi_in;
  if(gpseps == NULL || gpsepi >= NGPsep || gpseps[gpsepi] == NULL)
    error("gpsep %d is not allocated\n", gpsepi);
  gpsep = gpseps[gpsepi];

  /* check theta and tmax */
  assert(grange_in[0] >= 0 && grange_in[0] < grange_in[1]);
  for(k=0; k<gpsep->m; k++) {
    assert(dmin_in[k] >= 0 && dmin_in[k] < dmax_in[k]);
    if(gpsep->d[k] < dmin_in[k] || gpsep->d[k] > dmax_in[k])
      error("gpsep->d[%d]=%g outside drange[%d]=[%g,%g]",
        k, gpsep->d[k], k, dmin_in[k], dmax_in[k]);
  }
  if(gpsep->g < grange_in[0] || gpsep->g > grange_in[1])
    error("gpsep->g=%g outside grange=[%g,%g]", gpsep->g, grange_in[0], grange_in[1]);

  /* double check that derivatives have been calculated */
  if(! gpsep->dK)
    error("derivative info not in gpsep; use newGPsep with dK=TRUE");

  /* call C-side MLE */
  jmleGPsep(gpsep, *maxit_in, dmin_in, dmax_in, grange_in, dab_in, gab_in, *verb_in,
    dits_out, gits_out, dconv_out, 1);

  /* write back d and g */
  dupv(d_out, gpsep->d, gpsep->m);
  *g_out = gpsep->g;
}

/*
 * updateGPsep:
 *
 * quickly augment (O(n^2)) a gp based on new X-Z pairs.
 * Uses the Bartlet partition inverse equations
 */

void updateGPsep(GPsep* gpsep, unsigned int nn, double **XX, double *ZZ,
                 int verb)
{
  unsigned int i, j, l, n, m;
  double *kx, *x, *gvec;
  double mui, Ztg;
  double **Gmui, **temp;

  /* allocate space */
  n = gpsep->n; m = gpsep->m;
  kx = new_vector(n);
  gvec = new_vector(n);
  Gmui = new_matrix(n, n);
  temp = new_matrix(1, 1);

  /* for each new location */
  for(j=0; j<nn; j++) {

    /* shorthand for x being updated */
    x = XX[j];

    /* calculate the Bartlet quantities */
    calc_g_mui_kxy_sep(m, x, gpsep->X, n, gpsep->Ki, NULL, 0, gpsep->d,
                       gpsep->g, gvec, &mui, kx, NULL);

    /* Gmui = g %*% t(g)/mu */
    linalg_dgemm(CblasNoTrans,CblasTrans,n,n,1,
     mui,&gvec,n,&gvec,n,0.0,Gmui,n);

    /* Ki = Ki + Gmui */
    linalg_daxpy(n*n, 1.0, *Gmui, 1, *(gpsep->Ki), 1);

    /* now augment covariance matrices */
    /* for nn > 1 might be better to make bigger matricies once
       outside the for-loop */
    gpsep->Ki = new_bigger_matrix(gpsep->Ki, n, n, n+1, n+1);
    for(i=0; i<n; i++) gpsep->Ki[n][i] = gpsep->Ki[i][n] = gvec[i];
    gpsep->Ki[n][n] = 1.0/mui;
    gpsep->K = new_bigger_matrix(gpsep->K, n, n, n+1, n+1);
    for(i=0; i<n; i++) gpsep->K[n][i] = gpsep->K[i][n] = kx[i];
    covar_sep_symm(m, &x, 1, gpsep->d, gpsep->g, temp);
    gpsep->K[n][n] = **temp;

    /* update the determinant calculation */
    gpsep->ldetK += log(**temp + mui * linalg_ddot(n, gvec, 1, kx, 1));

    /* update KiZ and phi */
    /* Ztg = t(Z) %*% gvec */
    Ztg = linalg_ddot(n, gvec, 1, gpsep->Z, 1);
    gpsep->KiZ = realloc(gpsep->KiZ, sizeof(double)*(n+1));
    /* KiZ[1:n] += (Ztg/mu + Z*g) * gvec */
    linalg_daxpy(n, Ztg*mui + ZZ[j], gvec, 1, gpsep->KiZ, 1);
    /* KiZ[n+1] = Ztg + z*mu */
    gpsep->KiZ[n] = Ztg + ZZ[j]/mui;
    /* phi += Ztg^2/mu + 2*z*Ztg + z^2*mu */
    gpsep->phi += sq(Ztg)*mui + 2.0*ZZ[j]*Ztg + sq(ZZ[j])/mui;

    /* now augment X and Z */
    gpsep->X = new_bigger_matrix(gpsep->X, n, m, n+1, m);
    dupv(gpsep->X[n], x, m);
    gpsep->Z = (double*) realloc(gpsep->Z, sizeof(double)*(n+1));
    gpsep->Z[n] = ZZ[j];
    (gpsep->n)++;

    /* augment derivative covariance matrices */
    if(gpsep->dK) {
      for(l=0; l<m; l++)
        gpsep->dK[l] = new_bigger_matrix(gpsep->dK[l], n, n, n+1, n+1);
      double ***dKn = (double***) malloc(sizeof(double **) * m);
      for(l=0; l<m; l++) dKn[l] = new_matrix(1, n);
      diff_covar_sep(m, &x, 1, gpsep->X, n, gpsep->d, &(gpsep->K[n]), dKn);
      for(l=0; l<m; l++) {
        for(i=0; i<n; i++)
          gpsep->dK[l][i][n] = gpsep->dK[l][n][i] = dKn[l][0][i];
        delete_matrix(dKn[l]);
      }
      free(dKn);
      for(l=0; l<m; l++) gpsep->dK[l][n][n] = 0.0;
    }

    /* if more then re-allocate */
    if(j < nn-1) {
      kx = (double*) realloc(kx, sizeof(double)*(n+1));
      gvec = (double*) realloc(gvec, sizeof(double)*(n+1));
      Gmui = new_bigger_matrix(Gmui, n, n, n+1, n+1);
    }

    /* progress meter? */
    if(verb > 0)
      MYprintf(MYstdout, "update_sep j=%d, n=%d, ldetK=%g\n", j+1, gpsep->n, gpsep->ldetK);
    n = gpsep->n; /* increment for next interation */
  }

  /* clean up */
  delete_matrix(Gmui);
  free(kx);
  free(gvec);
  delete_matrix(temp);
}


/*
 * updateGPsep_R:
 *
 * R-interface allowing the internal/global GP representation
 * to be quickly augment (O(n^2)) based on new X-Z pairs.
 * Uses the Bartlet partition inverse equations
 */

void updateGPsep_R(/* inputs */
    int *gpsepi_in,
    int *m_in,
    int *nn_in,
    double *XX_in,
    double *ZZ_in,
    int *verb_in)
{
  GPsep *gpsep;
  unsigned int gpsepi;
  double **XX;

  /* get the cloud */
  gpsepi = *gpsepi_in;
  if(gpseps == NULL || gpsepi >= NGPsep || gpseps[gpsepi] == NULL)
    error("gpsep %d is not allocated\n", gpsepi);
  gpsep = gpseps[gpsepi];
  if((unsigned) *m_in != gpsep->m)
    error("ncol(X)=%d does not match GPsep/C-side (%d)", *m_in, gpsep->m);

  /* check that this is not a degenerate GP: not implemented (yet) */
  if(gpsep->d[0] <= 0) error("updating degenerate GP (d=0) not supported");

  /* sanity check and XX representation */
  XX = new_matrix_bones(XX_in, *nn_in, gpsep->m);

  /* call real C routine */
  updateGPsep(gpsep, *nn_in, XX, ZZ_in, *verb_in);

  /* clean up */
  free(XX);
}


/*
 * predGPsep:
 *
 * return the student-t predictive equations,
 * i.e., parameters to a multivatiate t-distribution
 * for XX predictive locations of dimension (n*m)
 */

void predGPsep(GPsep* gpsep, unsigned int nn, double **XX, const int nonug, 
  double *mean, double **Sigma, double *df, double *llik)
{
  unsigned int m, n;
  double **k;
  double phidf, g;

  /* easier referencing for dims */
  n = gpsep->n;  m = gpsep->m;

  /* are we using a nugget in the final calculation */
  if(nonug) g = SDEPS;
  else g = gpsep->g;

  /* variance (s2) components */
  *df = (double) n;
  phidf = gpsep->phi/(*df);

  /* calculate marginal likelihood (since we have the bits) */
  *llik = 0.0 - 0.5*((*df) * log(0.5 * gpsep->phi) + gpsep->ldetK);
  /* continuing: - ((double) n)*M_LN_SQRT_2PI;*/

  /* k <- covar(X1=X, X2=XX, d=Zt$d, g=0) */
  k = new_matrix(n, nn);
  covar_sep(m, gpsep->X, n, XX, nn, gpsep->d, k);
  /* Sigma <- covar(X1=XX, d=Zt$d, g=Zt$g) */
  covar_sep_symm(m, XX, nn, gpsep->d, g, Sigma);

  /* call generic function that would work for all GP covariance specs */
  pred_generic(n, phidf, gpsep->Z, gpsep->Ki, nn, k, mean, Sigma);

  /* clean up */
  delete_matrix(k);
}


/*
 * new_predutilGPsep_lite:
 *
 * utility function that allocates and calculate useful vectors
 * and matrices for prediction; used by predGPsep_lite and dmus2GP
 */

void new_predutilGPsep_lite(GPsep *gpsep, unsigned int nn, double **XX,
  double ***k, double ***ktKi, double **ktKik)
{
  /* k <- covar(X1=X, X2=XX, d=Zt$d, g=0) */
  *k = new_matrix(gpsep->n, nn);
  covar_sep(gpsep->m, gpsep->X, gpsep->n, XX, nn, gpsep->d, *k);

  /* call generic function that would work for all GP covariance specs */
  new_predutil_generic_lite(gpsep->n, gpsep->Ki, nn, *k, ktKi, ktKik);
}


/*
 * predGPsep_lite:
 *
 * return the student-t predictive equations,
 * i.e., parameters to a multivatiate t-distribution
 * for XX predictive locations of dimension (n*m);
 * lite because sigma2 not Sigma is calculated
 */

void predGPsep_lite(GPsep* gpsep, unsigned int nn, double **XX, const int nonug,
  double *mean, double *sigma2, double *df, double *llik)
{
  unsigned int i;
  double **k, **ktKi;
  double *ktKik;
  double phidf, g;

  /* sanity checks */
  assert(df);
  *df = gpsep->n;

  /* are we using a nugget in the final calculation */
  if(nonug) g = SDEPS;
  else g = gpsep->g;

  /* utility calculations */
  new_predutilGPsep_lite(gpsep, nn, XX, &k, &ktKi, &ktKik);

  /* mean <- ktKi %*% Z */
  if(mean) linalg_dgemv(CblasNoTrans,nn,gpsep->n,1.0,ktKi,nn,gpsep->Z,
                        1,0.0,mean,1);

  /* Sigma <- phi*(Sigma - ktKik)/df */
  /* *df = n - m - 1.0; */  /* only if estimating beta */
  if(sigma2) {
    phidf = gpsep->phi/(*df);
    // printVector(ktKik, nn, MYstdout, MACHINE);
    // MYprintf(MYstdout, "phi=%g, df=%g, phidf=%g, g=%g\n", gpsep->phi, *df, phidf, gpsep->g);
    for(i=0; i<nn; i++) sigma2[i] = phidf * (1.0 + g - ktKik[i]);
  }

  /* calculate marginal likelihood (since we have the bits) */
  /* might move to updateGP if we decide to move phi to updateGP */
  if(llik) *llik = 0.0 - 0.5*(((double) gpsep->n) * log(0.5* gpsep->phi) +
    gpsep->ldetK);
  /* continuing: - ((double) n)*M_LN_SQRT_2PI;*/

  /* clean up */
  delete_matrix(k);
  delete_matrix(ktKi);
  free(ktKik);
}


/*
 * predGPsep_R:
 *
 * R-interface to C-side function that
 * returns the student-t predictive equations,
 * i.e., parameters to a multivatiate t-distribution
 * for XX predictive locations of dimension (n*m)
 * using the stored GP parameterization
 */

void predGPsep_R(/* inputs */
        int *gpsepi_in,
        int *m_in,
        int *nn_in,
        double *XX_in,
        int *lite_in,
        int *nonug_in,

        /* outputs */
        double *mean_out,
        double *Sigma_out,
        double *df_out,
        double *llik_out)
{
  GPsep* gpsep;
  unsigned int gpsepi;
  double **Sigma, **XX;

  /* get the gp */
  gpsepi = *gpsepi_in;
  if(gpseps == NULL || gpsepi >= NGPsep || gpseps[gpsepi] == NULL)
    error("gpsep %d is not allocated\n", gpsepi);
  gpsep = gpseps[gpsepi];
  if((unsigned) *m_in != gpsep->m)
    error("ncol(X)=%d does not match GPsep/C-side (%d)", *m_in, gpsep->m);

  /* sanity check and XX representation */
  XX = new_matrix_bones(XX_in, *nn_in, *m_in);
  if(! *lite_in) Sigma = new_matrix_bones(Sigma_out, *nn_in, *nn_in);
  else Sigma = NULL;

  /* call the C-only Predict function */
  if(*lite_in) predGPsep_lite(gpsep, *nn_in, XX, *nonug_in, mean_out, 
    Sigma_out, df_out, llik_out);
  else predGPsep(gpsep, *nn_in, XX, *nonug_in, mean_out, Sigma, df_out, llik_out);

  /* clean up */
  free(XX);
  if(Sigma) free(Sigma);
}



/*
 * alGPsep_R:
 *
 * R interface to C-side function that returns the a Monte Carlo approximation
 * to the expected improvement (EI) and expected y-value (EY) under an
 * augmented Lagrangian with constraint separable GPs (cgpseps) assuming a
 * known linear objective function with scale bscale.  The constraints can
 * be scaled with the cnorms
 */

void alGPsep_R(/* inputs */
       int *m_in,
       double *XX_in,
       int *nn_in,
       int *fgpsepi_in,
       double *ff_in,
       double *fnorm_in,
       int *ncgpseps_in,
       int *cgpsepis_in,
       double *CC_in,
       double *cnorms_in,
       double *lambda_in,
       double *alpha_in,
       double *ymin_in,
       int *slack_in,
       double *equal_in,
       int *N_in,

       /* outputs */
       double *eys_out,
       double *eis_out)
{
  GPsep **cgpseps, *fgpsep;
  unsigned int gpsepi, ncgpseps, i, j, k, known, nknown;
  double **cmu, **cs, **XX, **CC;
  double *mu, *s;
  double df;

  /* get the gps */
  ncgpseps = *ncgpseps_in;
  nknown = 0;
  cgpseps = (GPsep**) malloc(sizeof(GPsep*) * ncgpseps);
  for(i=0; i<ncgpseps; i++) {
    if(cgpsepis_in[i] < 0) { cgpseps[i] = NULL; nknown++; continue; }
    gpsepi = cgpsepis_in[i];
    if(gpseps == NULL || gpsepi >= NGPsep || gpseps[gpsepi] == NULL)
      error("gpsep %d is not allocated\n", gpsepi);
    cgpseps[i] = gpseps[gpsepi];
    if((unsigned) *m_in != cgpseps[i]->m)
      error("ncol(X)=%d does not match GPsep/C-side (%d)",
        *m_in, cgpseps[i]->m);
  }

  /* make matrix bones */
  XX = new_matrix_bones(XX_in, *nn_in, *m_in);
  if(nknown > 0) CC = new_matrix_bones(CC_in, nknown, *nn_in);
  else CC = NULL;

  /* allocate storage for the (possibly null) distribution of f */
  if(*fgpsepi_in >= 0) { /* if modeling f */
    gpsepi = *fgpsepi_in;
    if(gpseps == NULL || gpsepi >= NGPsep || gpseps[gpsepi] == NULL)
      error("gpsep %d is not allocated\n", gpsepi);
    fgpsep = gpseps[gpsepi];
    mu = new_vector(*nn_in);
    s = new_vector(*nn_in);
    predGPsep_lite(fgpsep, *nn_in, XX, 0, mu, s, &df, NULL);
    for(k=0; k<*nn_in; k++) s[k] = sqrt(s[k]);
  } else { /* not modeling f; using known mean */
    mu = ff_in;
    s = NULL;
  }

  /* allocate storage for means and variances under normal approx */
  cmu = (double**) malloc(sizeof(double*) * ncgpseps);
  cs = (double**) malloc(sizeof(double*) * ncgpseps);
  known = 0;
  for(j=0; j<ncgpseps; j++) {
    if(cgpseps[j]) {
      cmu[j] = new_vector(*nn_in);
      cs[j] = new_vector(*nn_in);
      predGPsep_lite(cgpseps[j], *nn_in, XX, 0, cmu[j], cs[j], &df, NULL);
      for(k=0; k<*nn_in; k++) cs[j][k] = sqrt(cs[j][k]);
    } else { cmu[j] = CC[known]; cs[j] = NULL; known++; }
  }

  /* clean up */
  free(XX);
  free(cgpseps);

  GetRNGstate();

  /* use mu and s to calculate EI and EY */
  if(!(*slack_in)) {
    MC_al_eiey(ncgpseps, *nn_in, mu, s, *fnorm_in, cmu, cs, cnorms_in,
      lambda_in, *alpha_in, *ymin_in, equal_in, *N_in, eys_out, eis_out);
  } else {
    /* MC_alslack_eiey(ncgpseps, *nn_in, mu, s, *fnorm_in, cmu, cs, cnorms_in,
      lambda_in, alpha_in, *ymin_in, equal_in, *N_in, eys_out, eis_out); */
    if(nknown > 0) error("slack not implemented for nknown > 0");
    calc_alslack_eiey(ncgpseps, *nn_in, mu, s, *fnorm_in, cmu, cs, cnorms_in,
      lambda_in, *alpha_in, *ymin_in, equal_in, eys_out, eis_out);
  }

  PutRNGstate();

  /* clean up */
  for(i=0; i<ncgpseps; i++) if(cgpsepis_in[i] >= 0) {
    free(cmu[i]); free(cs[i]);
  }
  free(cmu);
  free(cs);
  if(*fgpsepi_in >= 0) free(mu);
  if(s) free(s);
}


/*
 * ieciGPsep:
 *
 * calculate the integrated expected conditional improvement
 * at locations Xcand averaging over reference locations Xref:
 * similar structure to ALC, except with expected improvement
 * rather than reduction invariance
 */

void ieciGPsep(GPsep *gpsep, unsigned int ncand, double **Xcand,
  double fmin, unsigned int nref, double **Xref, double *w, 
  int nonug, int verb, double *ieci)
{
  unsigned int m, n;
  int i;
  double **k;
  double *kx, *kxy, *gvec, *ktKik, *ktKikx, *pmref;
  double mui, df, g;
  double s2p[2] = {0, 0};

  /* degrees of freedom */
  m = gpsep->m;
  n = gpsep->n;

  /* are we using a nugget in the final calculation */
  if(nonug) g = SDEPS;
  else g = gpsep->g;

  /* allocate g, kxy, and ktKikx vectors */
  gvec = new_vector(n);
  kxy = new_vector(nref);
  kx = new_vector(n);
  ktKikx = new_vector(nref);

  /* calculate Xref predictive quantities */
  pmref = new_vector(nref);
  ktKik = new_vector(nref);
  /* this is a little inefficit since both ktKik and k (used below) are
   * calculated inside predGPsep_lite as intermediate steps */
  predGPsep_lite(gpsep, nref, Xref, 0, pmref, ktKik, &df, NULL);
  for(i=0; i<nref; i++) ktKik[i] = 1.0 + gpsep->g - (df/gpsep->phi)*ktKik[i];

  /* k <- covar(X1=X, X2=Xref, d=Zt$d, g=0) */
  k = new_matrix(nref, n);
  covar_sep(m, Xref, nref, gpsep->X, n, gpsep->d, k);

  /* calculate the ALC for each candidate */
  for(i=0; i<ncand; i++) {

    /* progress meter */
    if(verb > 0) 
      MYprintf(MYstdout, "ieciGPsep: calculating IECI for point %d of %d\n",
        i+1, ncand);

    /* calculate the g vector, mui, and kxy */
    calc_g_mui_kxy_sep(m, Xcand[i], gpsep->X, n, gpsep->Ki, Xref, nref,
      gpsep->d, gpsep->g, gvec, &mui, kx, kxy);

    /* skip if numerical problems */
    if(mui <= SDEPS) {
      ieci[i] = R_PosInf;
      continue;
    }

    /* use g, mu, and kxy to calculate ktKik.x */
    calc_ktKikx(ktKik, nref, k, n, gvec, mui, kxy, NULL, NULL, ktKikx);

    /* calculate the IECI */
    ieci[i] = calc_ieci(nref, ktKikx, s2p, gpsep->phi, g, NULL, pmref, df, fmin, w);
  }

  /* clean up */
  free(ktKikx);
  free(gvec);
  free(kx);
  free(kxy);
  free(pmref);
  free(ktKik);
  delete_matrix(k);
}


/*
 * ieciGPsep_R:
 *
 * R interface to C-side function that calculates the
 * integrated expected conditional improvement
 * at locations Xcand averaging over reference locations Xref:
 * similar structure to ALC, except with expected improvement
 * rather than reduction invariance
 */

void ieciGPsep_R(
      /* inputs */
      int *gpsepi_in,
      int *m_in,
      double *Xcand_in,
      int *ncand_in,
      double *fmin_in,
      double *Xref_in,
      int *nref_in,
      double *w_in,
      int *wb_in,
      int *nonug_in,
      int *verb_in,

       /* outputs */
      double *ieci_out)
{
  GPsep *gpsep;
  unsigned int gpsepi;
  double **Xcand, **Xref;

  /* get the gp */
  gpsepi = *gpsepi_in;
  if(gpseps == NULL || gpsepi >= NGPsep || gpseps[gpsepi] == NULL)
    error("gpsep %d is not allocated\n", gpsepi);
  gpsep = gpseps[gpsepi];
  if((unsigned) *m_in != gpsep->m)
    error("ncol(X)=%d does not match GPsep/C-side (%d)", *m_in, gpsep->m);

  /* check for null w */
  if(! *wb_in) w_in = NULL;

  /* make matrix bones */
  Xcand = new_matrix_bones(Xcand_in, *ncand_in, *m_in);
  Xref = new_matrix_bones(Xref_in, *nref_in, *m_in);

  /* call the C-only function */
  ieciGPsep(gpsep, *ncand_in, Xcand, *fmin_in, *nref_in, Xref,
    w_in, *nonug_in, *verb_in, ieci_out);

  /* clean up */
  free(Xcand);
  free(Xref);
}



/*
 * alcGPsep:
 *
 * return s2' component of the ALC calculation of the
 * expected reduction in variance calculation at locations
 * Xcand averaging over reference locations Xref:
 * ds2 = s2 - s2', where the s2s are at Xref and the
 * s2' incorporates Xcand, and everything is averaged
 * over Xref.
 */

void alcGPsep(GPsep *gpsep, unsigned int ncand, double **Xcand,
  unsigned int nref, double **Xref,  int verb, double *alc)
{
  unsigned int m, n;
  int i;
  double **k; //, **Gmui;
  double *kx, *kxy, *gvec, *ktKikx; //, *ktGmui;
  double mui, df;
  double s2p[2] = {0, 0};

  /* degrees of freedom */
  m = gpsep->m;
  n = gpsep->n;
  df = (double) n;

  /* allocate g, kxy, and ktKikx vectors */
  gvec = new_vector(n);
  kxy = new_vector(nref);
  kx = new_vector(n);
  ktKikx = new_vector(nref);

  /* k <- covar(X1=X, X2=Xref, d=Zt$d, g=0) */
  k = new_matrix(nref, n);
  covar_sep(m, Xref, nref, gpsep->X, n, gpsep->d, k);

  /* utility allocations */
  // Gmui = new_matrix(n, n);
  // ktGmui = new_vector(n);

  /* calculate the ALC for each candidate */
  for(i=0; i<ncand; i++) {

    /* progress meter */
    if(verb > 0) 
      MYprintf(MYstdout, "alcGPsep: calculating ALC for point %d of %d\n",
        i+1, ncand);

    /* calculate the g vector, mui, and kxy */
    calc_g_mui_kxy_sep(m, Xcand[i], gpsep->X, n, gpsep->Ki, Xref, nref,
      gpsep->d, gpsep->g, gvec, &mui, kx, kxy);

    /* skip if numerical problems */
    if(mui <= SDEPS) {
      alc[i] = R_NegInf;
      continue;
    }

    /* use g, mu, and kxy to calculate ktKik.x */
    // calc_ktKikx(NULL, nref, k, n, gvec, mui, kxy, Gmui, ktGmui, ktKikx);
    calc_ktKikx(NULL, nref, k, n, gvec, mui, kxy, NULL, NULL, ktKikx);

    /* calculate the ALC */
    alc[i] = calc_alc(nref, ktKikx, s2p, gpsep->phi, NULL, df, NULL);
  }

  /* clean up */
  // delete_matrix(Gmui);
  // free(ktGmui);
  free(ktKikx);
  free(gvec);
  free(kx);
  free(kxy);
  delete_matrix(k);
}


/*
 * alcGPsep_R:
 *
 * R interface to C-side function that returns the
 * s2' component of the ALC calculation of the
 * expected reduction in variance calculation given
 * the stored GP parameterization at locations Xcand
 * averaging over reference locations Xref:
 * ds2 = s2 - s2', where the s2s are at Xref and the
 * s2' incorporates Xcand, and everything is averaged
 * over Xref.
 */

void alcGPsep_R(
      /* inputs */
      int *gpsepi_in,
      int *m_in,
      double *Xcand_in,
      int *ncand_in,
      double *Xref_in,
      int *nref_in,
      int *verb_in,

       /* outputs */
       double *alc_out)
{
  GPsep *gpsep;
  unsigned int gpsepi;
  double **Xcand, **Xref;

  /* get the gp */
  gpsepi = *gpsepi_in;
  if(gpseps == NULL || gpsepi >= NGPsep || gpseps[gpsepi] == NULL)
    error("gpsep %d is not allocated\n", gpsepi);
  gpsep = gpseps[gpsepi];
  if((unsigned) *m_in != gpsep->m)
    error("ncol(X)=%d does not match GPsep/C-side (%d)", *m_in, gpsep->m);

  /* make matrix bones */
  Xcand = new_matrix_bones(Xcand_in, *ncand_in, *m_in);
  Xref = new_matrix_bones(Xref_in, *nref_in, *m_in);

  /* call the C-only function */
  alcGPsep(gpsep, *ncand_in, Xcand, *nref_in, Xref, *verb_in, alc_out);

  /* clean up */
  free(Xcand);
  free(Xref);
}



#ifdef _OPENMP
/*
 * alcGPsep_omp:
 *
 * OpenMP version of alcGPsep, above
 */

void alcGPsep_omp(GPsep *gpsep, unsigned int ncand, double **Xcand, unsigned int nref,
     double **Xref,  int verb, double *alc)
{
  unsigned int m, n;
  double df;
  double **k;
  double s2p[2] = {0, 0};

  /* degrees of freedom */
  m = gpsep->m;
  n = gpsep->n;
  df = (double) n;

  /* k <- covar(X1=X, X2=Xref, d=Zt$d, g=0) */
  k = new_matrix(nref, n);
  covar_sep(m, Xref, nref, gpsep->X, n, gpsep->d, k);

  #pragma omp parallel
  {
    int i, me, nth;
    // double **Gmui;
    double *kx, *kxy, *gvec, *ktKikx; //, *ktGmui;
    double mui;

    /* allocate g, kxy, and ktKikx vectors */
    gvec = new_vector(n);
    kxy = new_vector(nref);
    kx = new_vector(n);
    ktKikx = new_vector(nref);

    /* utility allocations */
    // Gmui = new_matrix(n, n);
    // ktGmui = new_vector(n);

    /* get thread information */
    me = omp_get_thread_num();
    nth = omp_get_num_threads();

    /* calculate the ALC for each candidate */
    for(i=me; i<ncand; i+=nth) {

      /* progress meter */
      #pragma omp master
      if(verb > 0) 
        MYprintf(MYstdout, "alcGPsep_omp: calculating ALC for point %d of %d\n",
          i+1, ncand);

      /* calculate the g vector, mui, and kxy */
      calc_g_mui_kxy_sep(m, Xcand[i], gpsep->X, n, gpsep->Ki, Xref, nref,
        gpsep->d, gpsep->g, gvec, &mui, kx, kxy);

      /* skip if numerical problems */
      if(mui <= SDEPS) {
        alc[i] = R_NegInf;
        continue;
      }

      /* use g, mu, and kxy to calculate ktKik.x */
      // calc_ktKikx(NULL, nref, k, n, gvec, mui, kxy, Gmui, ktGmui, ktKikx);
      calc_ktKikx(NULL, nref, k, n, gvec, mui, kxy, NULL, NULL, ktKikx);

      /* calculate the ALC */
      alc[i] = calc_alc(nref, ktKikx, s2p, gpsep->phi, NULL, df, NULL);
    }

    /* clean up */
    // delete_matrix(Gmui);
    // free(ktGmui);
    free(ktKikx);
    free(gvec);
    free(kx);
    free(kxy);
  }

  /* clean up noncd ..-parallel stuff */
  delete_matrix(k);
}


/*
 * alcGPsep_omp_R:
 *
 * OpenMP version of alcGPsep_R interface
 */

void alcGPsep_omp_R(/* inputs */
       int *gpsepi_in,
       int *m_in,
       double *Xcand_in,
       int *ncand_in,
       double *Xref_in,
       int *nref_in,
       int *verb_in,

       /* outputs */
       double *alc_out)
{
  GPsep *gpsep;
  unsigned int gpsepi;
  double **Xcand, **Xref;

  /* get the gp */
  gpsepi = *gpsepi_in;
  if(gpseps == NULL || gpsepi >= NGPsep || gpseps[gpsepi] == NULL)
    error("gpsep %d is not allocated\n", gpsepi);
  gpsep = gpseps[gpsepi];
  if((unsigned) *m_in != gpsep->m)
    error("ncol(X)=%d does not match GPsep/C-side (%d)", *m_in, gpsep->m);

  /* make matrix bones */
  Xcand = new_matrix_bones(Xcand_in, *ncand_in, *m_in);
  Xref = new_matrix_bones(Xref_in, *nref_in, *m_in);

  /* call the C-only function */
  alcGPsep_omp(gpsep, *ncand_in, Xcand, *nref_in, Xref, *verb_in, alc_out);

  /* clean up */
  free(Xcand);
  free(Xref);
}
#endif



/*
 * utility structure for fcn_nalcray_sep and defined below
 * for use with Brent_fmin (R's optimize) or uniroot;
 * very similar to alcinfo from gp.c, except with GPsep
 */

struct callinfo_alcray_sep {
  double **Xstart;
  double **Xend;
  double **Xref;
  GPsep *gpsep;
  double **k;
  double *gvec;
  double *kxy;
  double *kx;
  double *ktKikx;
  // double **Gmui;
  //  double *ktGmui;
  double *Xcand;
  double s2p[2];
  double df;
  double mui;
  int its;
  int verb;
};


/*
 * fcn_nalcray_sep:
 *
 * utility function lassed to Brent_Fmin (R's optimize) or
 * uniroot in order to optimize along a RAY with the ALC
 * statistic; ported from gp.c with GPsep
 */

static double fcn_nalcray_sep(double x, struct callinfo_alcray_sep *info)
{
  int m, n, j;
  double alc;

  m = info->gpsep->m;
  n = info->gpsep->n;
  (info->its)++;

  /* calculate Xcand along the ray */
  for(j=0; j<m; j++) info->Xcand[j] = (1.0 - x)*(info->Xstart[0][j]) + x*(info->Xend[0][j]);

  /* calculate the g vector, mui, and kxy */
  calc_g_mui_kxy_sep(m, info->Xcand, info->gpsep->X, n, info->gpsep->Ki, info->Xref,
    1, info->gpsep->d, info->gpsep->g, info->gvec, &(info->mui), info->kx, info->kxy);

  /* skip if numerical problems */
  if(info->mui <= SDEPS) alc = R_NegInf;
  else {
    /* use g, mu, and kxy to calculate ktKik.x */
    /* calc_ktKikx(NULL, 1, info->k, n, info->gvec, info->mui, info->kxy, info->Gmui,
      info->ktGmui, info->ktKikx); */
    calc_ktKikx(NULL, 1, info->k, n, info->gvec, info->mui, info->kxy, NULL,
      NULL, info->ktKikx);

    /* calculate the ALC */
    alc = calc_alc(1, info->ktKikx, info->s2p, info->gpsep->phi, NULL, info->df, NULL);
  }

  /* progress meter */
  if(info->verb > 0) {
    MYprintf(MYstdout, "alcray eval i=%d, Xcand=", info->its);
    for(j=0; j<m; j++) MYprintf(MYstdout, "%g ", info->Xcand[j]);
    MYprintf(MYstdout, "(s=%g), alc=%g\n", x, alc);
  }

  return 0.0-alc;
}


/* alcrayGPsep:
 *
 * optimize AIC via a ray search using the pre-stored separable GP
 * representation.  Return the convex combination s in (0,1) between
 * Xstart and Xend; copied from gp.c for GPsep
 */

double* alcrayGPsep(GPsep *gpsep, double **Xref, const unsigned int nump,
  double **Xstart, double **Xend, double *negalc, const unsigned int verb)
{
  unsigned int m, n, r;
  struct callinfo_alcray_sep info;
  double Tol = SDEPS;
  double obj0, na;
  double *snew;

  /* degrees of freedom */
  m = gpsep->m;
  n = gpsep->n;
  info.df = (double) n;

  /* other copying/default parameters */
  info.verb = verb;
  info.its = 0;
  info.s2p[0] = info.s2p[1] = 0;

  /* copy input pointers */
  info.Xref = Xref;
  info.Xcand = new_vector(m);
  info.gpsep = gpsep;

  /* allocate g, kxy, and ktKikx vectors */
  info.gvec = new_vector(n);
  info.kxy = new_vector(1);
  info.kx = new_vector(n);
  info.ktKikx = new_vector(1);

  /* k <- covar(X1=X, X2=Xref, d=Zt$d, g=0) */
  info.k = new_matrix(1, n);
  covar_sep(m, Xref, 1, gpsep->X, n, gpsep->d, info.k);

  /* utility allocations */
  // info.Gmui = new_matrix(n, n);
  // info.ktGmui = new_vector(n);

  /* allocate snew */
  snew = new_vector(nump);

  /* loop ovewr all pairs Xstart and Xend */
  assert(nump > 0);
  for(r=0; r<nump; r++) {

    /* select the rth start and end pair */
    info.Xstart = Xstart + r;
    info.Xend = Xend + r;

    /* use the C-backend of R's optimize function */
    snew[r] = Brent_fmin(0.0, 1.0, (double (*)(double, void*)) fcn_nalcray_sep, &info, Tol);
    if(snew[r] < Tol) snew[r] = 0.0;

    /* check s=0, as multi-modal ALC may result in larger domain of attraction
       for larger s-values but with lower mode */
    if(snew[r] > 0.0) {
      obj0 = fcn_nalcray_sep(0.0, &info);
      na = fcn_nalcray_sep(snew[r], &info);
      if(obj0 < na) { snew[r] = 0.0; na = obj0; }
      if(negalc) negalc[r] = na;
    } else if(negalc) negalc[r] = fcn_nalcray_sep(snew[r], &info);
  }

  /* clean up */
  // delete_matrix(info.Gmui);
  // free(info.ktGmui);
  free(info.ktKikx);
  free(info.gvec);
  free(info.kx);
  free(info.kxy);
  delete_matrix(info.k);
  free(info.Xcand);

  return(snew);
}


/* alcrayGPsep_R:
 *
 * R interface to C-side function that optimizes AIC via a ray search
 * using the pre-stored separable GP representation.  Return the convex
 * combination s in (0,1) between Xstart and Xend
 */

void alcrayGPsep_R(
      /* inputs */
       int *gpsepi_in,
       int *m_in,
       double *Xref_in,
       int *numrays_in,
       double *Xstart_in,
       double *Xend_in,
       int *verb_in,

       /* outputs */
       double *s_out,
       int *r_out)
{
  GPsep *gpsep;
  unsigned int gpsepi, rui;
  double **Xref, **Xstart, **Xend;
  double *s, *negalc;

  /* get the gp */
  gpsepi = *gpsepi_in;
  if(gpseps == NULL || gpsepi >= NGPsep || gpseps[gpsepi] == NULL)
    error("gpsep %d is not allocated\n", gpsepi);
  gpsep = gpseps[gpsepi];
  if((unsigned) *m_in != gpsep->m)
    error("ncol(X)=%d does not match GPsep/C-side (%d)", *m_in, gpsep->m);

  /* check numrays */
  if(*numrays_in < 1)
    error("numrays should be an integer scalar >= 1");

  /* make matrix bones */
  Xref = new_matrix_bones(Xref_in, 1, *m_in);
  Xstart = new_matrix_bones(Xstart_in, *numrays_in, *m_in);
  Xend = new_matrix_bones(Xend_in, *numrays_in, *m_in);

  /* call the C-only function */
  negalc = new_vector(*numrays_in);
  s = alcrayGPsep(gpsep, Xref, *numrays_in, Xstart, Xend, negalc, *verb_in);

  /* get the best combination */
  min(negalc, *numrays_in, &rui);
  *s_out = s[rui];
  *r_out = rui;

  /* clean up */
  free(s);
  free(negalc);
  free(Xref);
  free(Xstart);
  free(Xend);
}


/* lalcrayGPsep:
 *
 * local search of via ALC on rays (see alcrayGP) which finds the element
 * of Xcand that is closest to the max ALC value along a random ray emanating
 * from the (one of the) closest Xcands to Xref.  The offset determines which
 * candidate the ray emanates from (0 being the NN).  On input this function
 * assumes that the rows od Xcand are ordered by distance to Xref; this is
 * a straightforward adaptation of lalcrayGP to GPsep objects
 */

int lalcrayGPsep(GPsep *gpsep, double **Xcand, const unsigned int ncand,
  double **Xref, const unsigned int offset, unsigned int nr, double **rect,
  int verb)
{
  unsigned int m, mini, rmin;
  double **Xstart, **Xend;
  double *s, *negalc;

  /* gp dimension */
  m = gpsep->m;

  /* check numrays argument */
  assert(nr > 0);
  if(nr > ncand) nr = ncand;

  /* allocation and initialization */
  Xend = new_matrix(nr, m);
  Xstart = new_matrix(nr, m);
  negalc = new_vector(nr);

  /* set up starting and ending pairs */
  ray_bounds(offset, nr, m, rect, Xref, ncand, Xcand, Xstart, Xend);

  /* calculate ALC along ray */
  s = alcrayGPsep(gpsep, Xref, nr, Xstart, Xend, negalc, verb);

  /* find the best amongst the pairs */
  min(negalc, nr, &rmin);

  /* find the index into Xcand that is closest to Xstart + s*Xend */
  mini = convex_index(s, rmin, offset, nr, m, ncand, Xcand, Xstart, Xend);
  /* careful, storage from Xend re-used above */

  /* clean up */
  delete_matrix(Xstart);
  delete_matrix(Xend);
  free(s);
  free(negalc);

  return(mini);
}


/* lalcrayGPsep_R:
 *
 * R interface to C-side function that implements a local search of via ALC
 * on rays (see alcrayGPsep) which finds the element of Xcand that is closest
 * to the max ALC value along a random ray emanating from the (one of the)
 * closest Xcands to Xref.  The offset determines which candidate the ray
 * eminates from (0 being the NN).  On input this function assumes that the
 * rows od Xcand are ordered by distance to Xref
 */

void lalcrayGPsep_R(/* inputs */
       int *gpsepi_in,
       int *m_in,
       double *Xcand_in,
       int *ncand_in,
       double *Xref_in,
       int *offset_in,
       int *numrays_in,
       double *rect_in,
       int *verb_in,

       /* outputs */
       int *w_out)
{
  GPsep *gpsep;
  unsigned int gpsepi;
  double **Xref, **Xcand, **rect;

  /* get the gp */
  gpsepi = *gpsepi_in;
  if(gpseps == NULL || gpsepi >= NGPsep || gpseps[gpsepi] == NULL)
    error("gpsep %d is not allocated\n", gpsepi);
  gpsep = gpseps[gpsepi];
  if((unsigned) *m_in != gpsep->m)
    error("ncol(X)=%d does not match GPsep/C-side (%d)", *m_in, gpsep->m);

  /* check num rays */
  if(*numrays_in <= 0) error("numrays must be an integer scalar >= 1");

  /* make matrix bones */
  Xref = new_matrix_bones(Xref_in, 1, *m_in);
  Xcand = new_matrix_bones(Xcand_in, *ncand_in, *m_in);
  rect = new_matrix_bones(rect_in, 2, *m_in);

  /* call the C-only function */
  *w_out = lalcrayGPsep(gpsep, Xcand, *ncand_in, Xref, *offset_in,
    *numrays_in, rect, *verb_in);

  /* clean up */
  free(Xref);
  free(Xcand);
  free(rect);
}


/*
 * utility structure for fcn_nalcsep and fcn_ndalcsep defined below
 * for use with lbfgsb (R's optim with "L-BFGS-B" method)
 * for continuously optimizing ALC
 */

struct callinfo_alcsep {
  GPsep *gpsep;
  double alc;
  double *dalc;
  double *p;
  double **Xref;
  int nref;
  int its;  /* updated but not used since lbfgsb counts fmin and gr evals */
int verb;

double *gvec;
double *kxy; 
double *kx; 
double *ktKikx; 
double *Kidks; 
double **k;
double *dk;
};



/*
 * dalcGPsep:
 *
 * calculate the derivative, with respect to Xcand, of the
 * s2' component of the ALC calculation of the
 * expected reduction in variance calculation at locations
 * Xcand averaging over reference locations Xref:
 * ds2 = s2 - s2', where the s2s are at Xref and the
 * s2' incorporates Xcand, and everything is averaged
 * over Xref.
 */

void dalcGPsep(GPsep *gpsep, unsigned int ncand, double **Xcand, unsigned int nref,
            double **Xref,  int verb, double *alc, double **dalc, void *info)
{
  unsigned int m, n;
  int i, j, ell, iref;
  double **k; //, **Gmui;
  double *kx, *kxy, *gvec, *ktKikx, /* *ktGmui,*/ *dk, *Kidks;
  double mui, df, dkKikx, kgvec, dkxy, kKidks;
  double s2p[2] = {0, 0};
  
  /* degrees of freedom */
  m = gpsep->m;
  n = gpsep->n;
  df = (double) n;
  
  /* allocate g, kxy, and ktKikx vectors */
  gvec = new_vector(n);
  kxy = new_vector(nref);
  kx = new_vector(n);
  ktKikx = new_vector(nref);
  Kidks = new_vector(n);
  
  /* k <- covar.sep(X1=X, X2=Xref, d=Zt$d, g=0) */
  k = new_matrix(nref, n);
  covar_sep(m, Xref, nref, gpsep->X, n, gpsep->d, k);
  
  /* jth component of the derivative of the covariance between Xcand[i] and each X */
  dk = new_vector(n);
  
  /* utility allocations */
  // Gmui = new_matrix(n, n);
  // ktGmui = new_vector(n);
  
  /* initialize */
  zerov(*dalc, ncand*m);
  
  /* calculate the ALC for each candidate */
  for(i=0; i<ncand; i++) {
    
    /* progress meter */
    if(verb > 0) MYprintf(MYstdout, "calculating DALC for point %d of %d\n", verb, i, ncand);
    
    /* calculate the g vector, mui, and kxy */
    calc_g_mui_kxy_sep(m, Xcand[i], gpsep->X, n, gpsep->Ki, Xref, nref, gpsep->d,
                   gpsep->g, gvec, &mui, kx, kxy);
    
    /* skip if numerical problems */
    if(mui <= SDEPS) {
      alc[i] = R_NegInf;
      zerov(dalc[i], m);
      continue;
    }
    
    /* use g, mu, and kxy to calculate ktKik.x */
    // calc_ktKikx(NULL, nref, k, n, gvec, mui, kxy, Gmui, ktGmui, ktKikx);
    calc_ktKikx(NULL, nref, k, n, gvec, mui, kxy, NULL, NULL, ktKikx);
    
    /* calculate the ALC */
    alc[i] = calc_alc(nref, ktKikx, s2p, gpsep->phi, NULL, df, NULL);
    
    /* for derivative, loop over input coordinates */
    for(j=0; j<m; j++) {
      
      /* d_k.xprime <- (-2 * (xj[,j] - xc[,j]) / theta[j]) * k.xprime */
      for(ell=0; ell<n; ell++) dk[ell] = 0.0 - 2.0*(Xcand[i][j] - gpsep->X[ell][j])/gpsep->d[j] * kx[ell];
      
      /* B <- drop(t(d_k.xprime) %*% K.Js %*% k.xprime + t(k.xprime) %*% K.Js %*% d_k.xprime) */
      /* dkKikx <- 2.0 * dk %*% (Ki %*% kx) with the latter stored in gvec */
      dkKikx = 0.0 - 2.0 * linalg_ddot(n, dk, 1, gvec, 1) * mui;
      
      /* no longer need dk; re-use for ds = dk_shift = dk + kx * dkKikx * mui */
      linalg_daxpy(n, dkKikx / mui, kx, 1, dk, 1);
      
      /* Kidks <- Ki %*% ds with ds = dk calculated above */
      linalg_dsymv(n,1.0,gpsep->Ki,n,dk,1,0.0,Kidks,1);
      
      /* loop over reference locations */
      dalc[i][j] = 0.0;
      for(iref=0; iref<nref; iref++) {
        
        /* Furong's ds1 <- 2.0*(-t(k.x) %*% Kidks %*% t(g.xprime) %*% k.x) */
        /* kgvec is calculated in calc_ktKikx above, so this is partly duplicated */
        kgvec = linalg_ddot(n, k[iref], 1, gvec, 1);
        kKidks = linalg_ddot(n, k[iref], 1, Kidks, 1);
        dalc[i][j] -= 2.0 * kgvec * kKidks;
        
        /* Furong's ds3 <- -t(k.x) %*% g.xprime %*% t(g.xprime) %*% k.x * B */
        dalc[i][j] -= sq(kgvec) * dkKikx;
        
        /* Furong's d_K.xprime.x <- (-2 * (xj[,j] - xp[i,j]) / theta[j]) * K.xprime.x */
        dkxy = (-2.0 * (Xcand[i][j] - Xref[iref][j]) / gpsep->d[j]) * kxy[iref];
        
        /* Furong's ds4 <- 2 * ( - t(k.x) %*% Kidks * K.xprime.x / v.xprime +
        t(k.x) %*% g.xprime * d_K.xprime.x) */
        dalc[i][j] += 2.0 *(0.0 - kKidks * kxy[iref] / mui + kgvec * dkxy);
        
        /* Furong's ds5 <- K.xprime.x * (2 * d_K.xprime.x + K.xprime.x * B / v.xprime) / v.xprime */
        dalc[i][j] += kxy[iref] * (2.0 * dkxy + kxy[iref] * dkKikx / mui) / mui;
      }
      
      /* after this line, result matches Furong's code */
      dalc[i][j] /= ((double) nref);
      
      /* scaling required to match alcGP output */
      dalc[i][j] *= (df/(df-2.0))*(s2p[1] + gpsep->phi)/(s2p[0] + df);
      
    }
  }
  
  /* clean up */
  // delete_matrix(Gmui);
  // free(ktGmui);
  free(ktKikx);
  free(gvec);
  free(kx);
  free(kxy);
  free(dk);
  free(Kidks);
  delete_matrix(k);
}

/*
 * dalcGPsep_R:
 *
 * R interface to C-side function that returns the
 * derivative, with respect to Xcand_in values,
 * of the s2' component of the ALC calculation of the
 * expected reduction in variance calculation given
 * the stored separable GP parameterization at locations Xcand
 * averaging over reference locations Xref:
 * ds2 = s2 - s2', where the s2s are at Xref and the
 * s2' incorporates Xcand, and everything is averaged
 * over Xref.
 */

void dalcGPsep_R(/* inputs */
       int *gpsepi_in,
       int *m_in,
       double *Xcand_in,
       int *ncand_in,
       double *Xref_in,
       int *nref_in,
       int *verb_in,

       /* outputs */
       double *alc_out,
       double *dalc_out)
{
  GPsep *gpsep;
  unsigned int gpsepi;
  double **Xcand, **Xref, **dalc;

  /* get the gp */
  gpsepi = *gpsepi_in;
  if(gpseps == NULL || gpsepi >= NGPsep || gpseps[gpsepi] == NULL)
    error("gpsep %d is not allocated\n", gpsepi);
  gpsep = gpseps[gpsepi];
  if((unsigned) *m_in != gpsep->m)
    error("ncol(X)=%d does not match GPsep/C-side (%d)", *m_in, gpsep->m);

  /* make matrix bones */
  Xcand = new_matrix_bones(Xcand_in, *ncand_in, *m_in);
  Xref = new_matrix_bones(Xref_in, *nref_in, *m_in);
  dalc = new_matrix_bones(dalc_out, *ncand_in, *m_in);

  /* call the C-only function */
  dalcGPsep(gpsep, *ncand_in, Xcand, *nref_in, Xref, *verb_in, alc_out, dalc, NULL);

  /* clean up */
  free(Xcand);
  free(Xref);
  free(dalc);
}


/*
 * fcn_nalcsep:
 *
 * a utility function for lbfgsb (R's optim with "L-BFGS-B" method) to
 * evaluate the separable ALC calculation at the reference locations
 * specified in info; calculates derivative at the same time and stores
 * it in info to be read later by the derivative function
 */

static double fcn_nalcsep(int n, double *p, struct callinfo_alcsep *info)
{
  double alc;
  int k, m;
  
  /* sanity check */
  m = info->gpsep->m;
  
  /* evaluate alc and derivative with potentially new paramterization */
  dalcGPsep(info->gpsep, 1, &p, info->nref, info->Xref, info->verb, &alc, 
         &(info->dalc), info);
  
  /* copy p and alc into info */
  (info->its)++;
  dupv(info->p, p, m);
  info->alc = alc;
  
  /* progress meter */
  if(info->verb > 0) {
    MYprintf(MYstdout, "fmin it=%d, par=(%g", info->its, p[0]);
    for(k=1; k<m; k++) MYprintf(MYstdout, " %g", p[k]);
    MYprintf(MYstdout, "), log(alc)=%g\n", log(alc));
  }
  
  /* done */
  return 0.0-log(alc); 
}


/*
 * fcn_ndalcsep:
 *
 * a utility function for lbfgsb (R's optim with "L-BFGS-B" method) to
 * evaluate the separable ALC calculation at the reference locations
 * specified in info; read the stored derivative from fcn_nalcsep
 * 
 */

static void fcn_ndalcsep(int n, double *p, double *df, struct callinfo_alcsep *info)
{
  int psame, k, m;
  
  /* sanity check */
  m = info->gpsep->m;
  assert(n == m + 1);
  
  /* check if parameters in p are new */
  psame = 1;
  for(k=0; k<m; k++) if(p[k] != info->p[k]) { psame = 0; break; }
  if(!psame) error("ndalc incorrectly assumed grad comes after obj");
  
  /* calculate (and negate) derivative from saved values */
  for(k=0; k<n; k++) df[k] = 0.0-info->dalc[k]/info->alc;
  
  /* progress meter */
  if(info->verb > 1) {
    MYprintf(MYstdout, "grad it=%d, par=(%g", info->its, p[0]);
    for(k=1; k<m; k++) MYprintf(MYstdout, " %g", p[k]);
    MYprintf(MYstdout, "), dd=(%g", df[0]);
    for(k=1; k<m; k++) MYprintf(MYstdout, " %g", df[k]);
    MYprintf(MYstdout, "\n", df[m]);
  }
}



/*
 * alcoptGPsep:
 *
 * continuously optimizes ALC via derivatives
 * to get an approximate new location for multiple design.
 */

double alcoptGPsep(GPsep* gpsep, double *start, double* lower, double *upper, 
              double **Xref, const int nref, const unsigned int maxit, int verb, double *p, 
              int *its, char *msg, int *conv, int fromR)
{
  double obj;
  int k, lbfgs_verb;
  
  /* create structure for lbfgsb */
  struct callinfo_alcsep info;
  info.gpsep = gpsep;
  info.p = new_vector(gpsep->m);
  info.dalc = new_vector(gpsep->m);
  info.Xref = Xref;
  info.nref = nref;
  info.its = 0;
  info.verb = verb-6;
  
  /* allocate memory shared across all dalc evals */
  info.gvec = new_vector(gpsep->n);
  info.kxy = new_vector(nref);
  info.kx = new_vector(gpsep->n);
  info.ktKikx = new_vector(nref);
  info.Kidks = new_vector(gpsep->n);
  info.k = new_matrix(nref, gpsep->n);
  info.dk = new_vector(gpsep->n);
  
  /* copy the starting value */
  dupv(p, start, gpsep->m);
  
  /* potentially start progress meter */
  if(verb > 0) {
    MYprintf(MYstdout, "(par=[%g", p[0]);
    for(k=1; k<gpsep->m; k++) MYprintf(MYstdout, ",%g", p[k]);
    alcGPsep(gpsep, 1, &p, nref, Xref, 0, &obj);
    MYprintf(MYstdout, "], log(alc)=%g) ", log(obj));
  }
  
  /* set ifail argument and verb/trace arguments */
  *conv = 0;
  if(verb <= 1) lbfgs_verb = 0;
  else lbfgs_verb = verb - 1;
  
  /* call the C-routine behind R's optim function with method = "L-BFGS-B" */
  obj = MYlbfgsb(gpsep->m, p, lower, upper,
           (double (*)(int, double*, void*)) fcn_nalcsep,
           (void (*)(int, double *, double *, void *)) fcn_ndalcsep,
           conv, &info, 0.1, its, maxit, msg, lbfgs_verb, fromR);
  
  /* print progress */
  if(verb > 0) {
    MYprintf(MYstdout, "-> %d lbfgsb its -> (par=[%g", its[1], p[0]);
    for(k=1; k<gpsep->m; k++) MYprintf(MYstdout, ",%g", p[k]);
    MYprintf(MYstdout, "], obj=%g", obj);
    alcGPsep(gpsep, 1, &p, nref, Xref, 0, &obj);
    MYprintf(MYstdout, ", log(alc)=%g)\n", log(obj));
  }
  
  /* clean up */
  free(info.dalc);
  free(info.p);
  free(info.gvec); 
  free(info.kxy); 
  free(info.kx); 
  free(info.ktKikx); 
  free(info.Kidks); 
  delete_matrix(info.k);
  free(info.dk); 

  /* return optimize objective value */
  return(obj);
}


/*
* alcoptGPsep_R:
*
* R-interface to continuously optimize ALC via derivatives
* to get an approximate new location for multiple design.
*/

void alcoptGPsep_R(
/* inputs */
  int *gpsepi_in,
  int *maxit_in,
  int *verb_in,
  double *start_in,
  double *lower_in,
  double *upper_in,
  int *m_in,
  double *Xref_in,
  int *nref_in,

  /* outputs */
  double *par_out,
  double *val_out,
  int *its_out,
  char **msg_out,
  int *conv_out)
{
  GPsep *gpsep;
  unsigned int gpsepi;
  double ** Xref;
  
  /* get the cloud */
  gpsepi = *gpsepi_in;
  if(gpseps == NULL || gpsepi >= NGPsep || gpseps[gpsepi] == NULL)
    error("gpsep %d is not allocated\n", gpsepi);
  gpsep = gpseps[gpsepi];
  if((unsigned) *m_in != gpsep->m)
    error("ncol(X)=%d does not match GPsep/C-side (%d)", *m_in, gpsep->m);
  
  /* make matrix bones */
  Xref = new_matrix_bones(Xref_in, *nref_in, *m_in);
  
  /* call the ordinary C function */
  *val_out = alcoptGPsep(gpsep, start_in, lower_in, upper_in, Xref, *nref_in, *maxit_in, 
           *verb_in, par_out, its_out, *msg_out, conv_out, 1);
}


/* lalcoptGPsep:
 *
 * local search of via ALC via optim (see alcoptGPsep) which finds the 
 * element of Xcand that is closest to the max ALC via derivatives.  
 * The offset determines which candidate is the starting value of the 
 * search (0 being the NN).  On input this function assumes 
 * that the rows of Xcand are ordered by distance to Xref
 */

int lalcoptGPsep(GPsep *gpsep, double **Xcand, const unsigned int ncand, double **Xref,
              const unsigned int nref, const unsigned int offset, unsigned int numstart, 
              double **rect, int maxit, int verb, int fromR)
{
  unsigned int m, mini, mini_best, eoff;
  int conv, s, its[2];
  char msg[60];
  double *start, *p;
  double alc, alc_best;
  
  /* gpsep dimension */
  m = gpsep->m;
  
  /* check numrays argument */
  assert(numstart > 0);
  if(numstart > ncand) numstart = ncand;
  
  /* allocation and initialization */
  start = new_vector(m);
  p = new_vector(m);
  
  /* initialize search over numstart starting locations */
  mini_best = 0; 
  alc_best = R_PosInf;
  
  /* set up starting and ending pairs */
  for(s=0; s<numstart; s++) {
    
    /* starting point for derivative based search */
    eoff = (offset + s) % ncand;
    dupv(start, Xcand[eoff], m);
    
    /* optimize ALC via derivatives */
    alcoptGPsep(gpsep, start, rect[0], rect[1], Xref, nref, maxit, 
             verb, p, its, msg, &conv, fromR);
    
    /* find the index into Xcand that is closest to Xstart + s*Xend */
    mini = closest_index(m, ncand, Xcand, p);
    
    /* calculate ALC at Xcand[mini,] and see if its the best */
    alcGPsep(gpsep, 1, &(Xcand[mini]), nref, Xref, verb, &alc);
    if(alc < alc_best) {
      mini_best = mini;
      alc_best = alc;
    }
  }
  
  /* clean up */
  free(start);
  free(p);
  
  return(mini_best);
}


/* lalcoptGPsep_R:
*
* R interface to C-side function that implements a local search of via ALC
* via optim (see alcoptGPsep) which finds the element of Xcand that is closest
* to the max ALC via derivatives.  The offset determines which candidate is the
* starting value of the search (0 being the NN).  On input this function assumes 
* that the rows of Xcand are ordered by distance to Xref
*/

void lalcoptGPsep_R(/* inputs */
int *gpsepi_in,
int *m_in,
double *Xcand_in,
int *ncand_in,
double *Xref_in,
int *nref_in,
int *offset_in,
int *numstart_in,
double *rect_in,
int *maxit_in,
int *verb_in,

/* outputs */
int *w_out)
{
  GPsep *gpsep;
  unsigned int gpsepi;
  double **Xref, **Xcand, **rect;
  
  /* get the gpsep */
  gpsepi = *gpsepi_in;
  if(gpseps == NULL || gpsepi >= NGPsep || gpseps[gpsepi] == NULL)
    error("gpsep %d is not allocated\n", gpsepi);
  gpsep = gpseps[gpsepi];
  if((unsigned) *m_in != gpsep->m)
    error("ncol(X)=%d does not match GPsep/C-side (%d)", *m_in, gpsep->m);
  
  /* check num rays */
  if(*numstart_in <= 0) error("numstart must be an integer scalar >= 1");
  
  /* make matrix bones */
  Xref = new_matrix_bones(Xref_in, *nref_in, *m_in);
  Xcand = new_matrix_bones(Xcand_in, *ncand_in, *m_in);
  rect = new_matrix_bones(rect_in, 2, *m_in);
  
  /* call the C-only function */
  *w_out = lalcoptGPsep(gpsep, Xcand, *ncand_in, Xref, *nref_in, *offset_in, 
                     *numstart_in, rect, *maxit_in, *verb_in, 1);
                     
                     /* clean up */
                     free(Xref);
                     free(Xcand);
                     free(rect);
}
