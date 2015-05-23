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


#include "matrix.h"
#include "gp.h"
#include "util.h"
#include "linalg.h"
#include "rhelp.h"
#include "covar.h"
#include "ieci.h"
#include <assert.h>
#include <stdlib.h>
#include <math.h>
#include <Rmath.h>
#ifdef _GPU
  #include "alc_gpu.h"
#endif
#ifdef _OPENMP
  #include <omp.h>
#endif


/*
 * Global variables used to accumulate data on the C-side
 * as passed from initialization and updtating R calls 
 */

unsigned int NGP = 0;
GP **gps = NULL;


/*
 * get_gp:
 *
 * returns an integer reference to a free gp 
 * 
 */

unsigned int get_gp(void)
{
  unsigned int i;
  if(NGP == 0) {
    assert(gps == NULL);
    gps = (GP**) malloc(sizeof(GP*));
    gps[0] = NULL;
    NGP = 1;
    return 0;
  } else {
    for(i=0; i<NGP; i++) {
      if(gps[i] == NULL) return i;
    }
    gps = (GP**) realloc(gps, sizeof(GP*) * (2*NGP));
    for(i=NGP; i<2*NGP; i++) gps[i] = NULL;
    NGP *= 2;
    return NGP/2;
  }
}


/*
 * deletedKGP:
 *
 * delete the dK components of the gp 
 */

void deletedKGP(GP *gp)
{
  if(gp->dK) { delete_matrix(gp->dK); gp->dK = NULL; }
  if(gp->d2K) { delete_matrix(gp->d2K); gp->d2K = NULL; }
  gp->F = 0.0;
}


/*
 * deletedKGP_R:
 *
 * R-interface to code destroying dK information 
 * so that they are not updated in future calculations
 */

void deletedKGP_R(/* inputs */
		  int *gpi_in)
{
  GP *gp;
  unsigned int gpi;

  /* get the cloud */
  gpi = *gpi_in;
  if(gps == NULL || gpi >= NGP || gps[gpi] == NULL) 
    error("gp %d is not allocated\n", gpi);
  gp = gps[gpi];

  /* check if needed */
  if(! gp->dK) error("derivative info not in gp");
  
  /* call real C routine */
  deletedKGP(gp);
}


/* 
 * deleteGP:
 *
 * free the memory allocated to a gp structure 
 */

void deleteGP(GP* gp)
{
  assert(gp);
  assert(gp->X); delete_matrix(gp->X);
  assert(gp->Z); free(gp->Z);
  assert(gp->K); delete_matrix(gp->K);
  assert(gp->Ki); delete_matrix(gp->Ki);  
  assert(gp->KiZ); free(gp->KiZ);
  deletedKGP(gp);
  free(gp);
}


/* 
 * deleteGP_index:
 *
 * delete the i-th gp
 */

void deleteGP_index(unsigned int i)
{
  if(!(gps == NULL || i >= NGP || gps[i] == NULL)) { 
    deleteGP(gps[i]);
    gps[i] = NULL;
  } else error("gp %d is not allocated\n", i);
}


/*
 * deleteGP_R:
 *
 * R-interface to deleteGP
 */

void deleteGP_R(int *gp)
{
  deleteGP_index(*gp);
}


/*
 * deleteGPs:
 *
 * delete all of the gps stored in
 * the array, and destroy the array
 */

void deleteGPs(void)
{
  int i;
  for(i=0; i<NGP; i++) {
    if(gps[i]) {
      myprintf(mystdout, "removing gp %d\n", i);
      deleteGP(gps[i]);
    }
  }
  if(gps) free(gps);
  gps = NULL;
  NGP = 0;
}


/*
 * deleteGPs_R:
 *
 * R interface to deleteGPs
 */

void deleteGPs_R(void)
{ 
  if(gps) deleteGPs();
}


/*
 * dllikGP:
 *
 * batch calculation of the first and second derivatives
 * of the log likelihood of a gp, with respect to the 
 * lengthscale parameter, d; requires that derivatives
 * be pre-calculated
 */

void dllikGP(GP *gp, double *ab, double *dllik, double *d2llik)
{
  double *KiZtwo;
  double **dKKidK, **dKKi, **two;
  unsigned int i, j, n;
  double dn, phirat, dlp, d2lp;

  /* sanity check */
  assert(gp->dK && gp->d2K);
  assert(dllik || d2llik);  /* one can be NULL if not needed */

  /* deal with possible prior */
  if(ab && ab[0] > 0 && ab[1] > 0) {
    dlp = (ab[0] - 1.0)/gp->d - ab[1];
    d2lp = 0.0 - (ab[0] - 1.0)/sq(gp->d);
  } else dlp = d2lp = 0.0;

  /* copy dims for fast access */
  n = gp->n;
  dn = (double) n;
  
  if(d2llik) {
    /* dKKi = dK %*% Ki */
    dKKi = new_matrix(n, n);
    linalg_dsymm(CblasRight,n,n,1.0,gp->Ki,n,gp->dK,n,0.0,dKKi,n);
    /* dkKidK = dK %*% Ki %*% dK */
    dKKidK = new_matrix(n, n);
    linalg_dsymm(CblasRight,n,n,1.0,gp->dK,n,dKKi,n,0.0,dKKidK,n);
  } else dKKi = dKKidK = NULL;

  /* d2llik = - 0.5 * tr(Ki %*% [d2K - dKKidK]); the first expression */
  /* dllik = - 0.5 * tr(Ki %*% dK) */
  if(d2llik) *d2llik = d2lp;
  if(dllik) *dllik = dlp; 
  /* two = second expresion: 2*dKKidK - dKKidK, re-using dKKidK */
  two = dKKi; /* this memory is no longer needed */
  for(i=0; i<n; i++) {
    for(j=0; j<i; j++) { /* off diagonal */
      if(dllik) *dllik -= gp->Ki[i][j] * gp->dK[i][j];
      if(d2llik) {
        *d2llik -= gp->Ki[i][j] * (gp->d2K[i][j] - dKKidK[i][j]);
        two[i][j] = two[j][i] = 2.0*dKKidK[i][j] - gp->d2K[i][j];
      }
    }
    /* on-diagonal */
    if(dllik) *dllik -= 0.5 * gp->Ki[i][i] * gp->dK[i][i];
    if(d2llik) {
      *d2llik -= 0.5 * gp->Ki[i][i] * (gp->d2K[i][i] - dKKidK[i][i]);
      two[i][i] = 2.0*dKKidK[i][i] - gp->d2K[i][i];
    }
  }

  /* now the second part of the expression: */
  /* d2llik -= 0.5 * KiZ %*% two %*% KiZ */
  KiZtwo = new_vector(n);
  if(d2llik) {
    linalg_dsymv(n,1.0,two,n,gp->KiZ,1,0.0,KiZtwo,1);
    *d2llik -= 0.5*dn*linalg_ddot(n, gp->KiZ, 1, KiZtwo, 1) / gp->phi;
  }

  /* now third part of the expression, re-using KiZtwo */
  /* KiZtwo = dK %*% KiZ */
  linalg_dsymv(n,1.0,gp->dK,n,gp->KiZ,1,0.0,KiZtwo,1);
  /* now t(KiZ) %*% dK %*% KiZ */
  phirat = linalg_ddot(n, gp->KiZ, 1, KiZtwo, 1) / gp->phi;
  if(d2llik) *d2llik += 0.5*dn*sq(phirat);
  if(dllik) *dllik += 0.5*dn*phirat;

  /* clean up */
  free(KiZtwo);
  if(dKKi) delete_matrix(dKKi);
  if(dKKidK) delete_matrix(dKKidK);
}


/*
 * dllikGP_nug:
 *
 * batch calculation of the first and second derivatives
 * of the log likelihood of a gp, with respect to the 
 * NUGGET parameter, d
 */

void dllikGP_nug(GP *gp, double *ab, double *dllik, double *d2llik)
{
  double *KiZtwo;
  double **two, **dKKidK;
  unsigned int i, j, n;
  double dn, phirat, dlp, d2lp;

  /* sanity check */
  assert(dllik || d2llik);  /* one can be NULL if not needed */

  /* deal with possible prior */
  if(ab && ab[0] > 0 && ab[1] > 0) {
    dlp = (ab[0] - 1.0)/gp->g - ab[1];
    d2lp = 0.0 - (ab[0] - 1.0)/sq(gp->g);
  } else dlp = d2lp = 0.0;

  /* copy dims for fast access */
  n = gp->n;
  dn = (double) n;
  
  if(d2llik) {
    two = new_matrix(n, n);
    dKKidK = gp->Ki; 
  } else two = dKKidK = NULL;

  /* d2llik = - 0.5 * tr(Ki %*% [0.0 - Ki]); the first expression */
  /* dllik = - 0.5 * tr(Ki) */
  if(d2llik) *d2llik = d2lp;
  if(dllik) *dllik = dlp; 
  /* two = second expresion: 2*dKKidK - dKKidK, re-using dKKidK */
  for(i=0; i<n; i++) {
    if(d2llik) {
      for(j=0; j<i; j++) { /* off diagonal */
        *d2llik += gp->Ki[i][j] * dKKidK[i][j];
        two[i][j] = two[j][i] = 2.0*dKKidK[i][j];
      }
    }
    /* on-diagonal */
    if(dllik) *dllik -= 0.5 * gp->Ki[i][i];
    if(d2llik) {
      *d2llik += 0.5 * gp->Ki[i][i] * dKKidK[i][i];
      two[i][i] = 2.0*dKKidK[i][i];
    }
  }

  /* now the second part of the expression: */
  /* d2llik -= 0.5 * KiZ %*% two %*% KiZ */
  if(d2llik) {
    KiZtwo = new_vector(n);
    linalg_dsymv(n,1.0,two,n,gp->KiZ,1,0.0,KiZtwo,1);
    *d2llik -= 0.5*dn*linalg_ddot(n, gp->KiZ, 1, KiZtwo, 1) / gp->phi;
    free(KiZtwo);
  }

  /* now third part of the expression */
  /* now t(KiZ) %*% dK %*% KiZ */
  phirat = linalg_ddot(n, gp->KiZ, 1, gp->KiZ, 1) / gp->phi;
  if(d2llik) *d2llik += 0.5*dn*sq(phirat);
  if(dllik) *dllik += 0.5*dn*phirat;

  /* clean up */
  if(two) delete_matrix(two);
}



/*
 * dllikGP_R:
 *
 * R-interface to calculate the derivatives of the
 * likelihood of a GP - wrt lengthscale
 */

void dllikGP_R(/* inputs */
               int *gpi_in,
               double *ab_in,

               /* outputs */
               double *dllik_out,
               double *d2llik_out)
{
  GP *gp;
  unsigned int gpi;

  /* get the cloud */
  gpi = *gpi_in;
  if(gps == NULL || gpi >= NGP || gps[gpi] == NULL)
    error("gp %d is not allocated\n", gpi);
  gp = gps[gpi];

  /* double check that derivatives have been calculated */
  if(! gp->dK) error("derivative info not in gp; use newGP with dK=TRUE");

  /* calculate log likelihood */
  dllikGP(gp, ab_in, dllik_out, d2llik_out);
}




/*
 * dllikGP_nug_R:
 *
 * R-interface to calculate the derivatives of the
 * likelihood of a GP - wrt the NUGGET
 */

void dllikGP_nug_R(/* inputs */
               int *gpi_in,
               double *ab_in,

               /* outputs */
               double *dllik_out,
               double *d2llik_out)
{
  GP *gp;
  unsigned int gpi;

  /* get the cloud */
  gpi = *gpi_in;
  if(gps == NULL || gpi >= NGP || gps[gpi] == NULL)
    error("gp %d is not allocated\n", gpi);
  gp = gps[gpi];

  /* calculate log likelihood */
  dllikGP_nug(gp, ab_in, dllik_out, d2llik_out);
}



/*
 * fishinfoGP:
 *
 * batch calculation of the  Fisher Information matrix of
 * a gp; requires that derivatives be pre-calculated
 */

double fishinfoGP(GP *gp)
{
  double d2llik;
  dllikGP(gp, NULL, NULL, &d2llik);
  return -d2llik;
}


/*
 * calc_ZtKiZ:
 *
 * re-calculates phi = ZtKiZ from Ki and Z stored in
 * the GP object; also update KiZ on which it depends
 */

void calc_ZtKiZ(GP *gp) 
{
  assert(gp);
  /* phi <- t(Z) %*% Ki %*% Z */
  if(gp->KiZ == NULL) gp->KiZ = new_vector(gp->n);
  linalg_dsymv(gp->n,1.0,gp->Ki,gp->n,gp->Z,1,0.0,gp->KiZ,1);
  gp->phi = linalg_ddot(gp->n, gp->Z, 1, gp->KiZ, 1);
}


/*
 * newdKGP:
 *
 * allocate new space for dK and d2K calculations, and 
 * cancluate derivatives and Fisher informations
 */

void newdKGP(GP *gp)
{
  assert(gp->dK == NULL && gp->d2K == NULL);
  gp->dK = new_matrix(gp->n, gp->n);
  gp->d2K = new_matrix(gp->n, gp->n);
  diff_covar_symm(gp->m, gp->X, gp->n, gp->d, gp->dK, gp->d2K);
  gp->F = fishinfoGP(gp);
}


/*
 * buildKGP_R:
 *
 * R-interface to code allocating dK information 
 * for future calculations
 */

void buildKGP_R(/* inputs */
		int *gpi_in)
{
  GP *gp;
  unsigned int gpi;

  /* get the cloud */
  gpi = *gpi_in;
  if(gps == NULL || gpi >= NGP || gps[gpi] == NULL) 
    error("gp %d is not allocated\n", gpi);
  gp = gps[gpi];

  /* check if needed */
  if(gp->dK) error("derivative info already in gp");
  
  /* call real C routine */
  newdKGP(gp);
}


/*
 * buildGP:
 *
 * intended for newly created GPs, e.g., via newGP or newGP_sub;
 * does all of the correlation calculations, etc., after data and
 * parameters are defined */


GP* buildGP(GP *gp, int dK)
{ 
  double **Kchol, **X;
  unsigned int n, m;
  int info;

  assert(gp && gp->K == NULL);
  if(gp->d == 0) assert(!dK);
  n = gp->n;
  m = gp->m;
  X = gp->X;

  /* build covariance matrix */
  gp->K = new_matrix(n, n);
  if(gp->d > 0) covar_symm(m, X, n, gp->d, gp->g, gp->K);
  else id(gp->K, n);
  
  /* invert covariance matrix */
  gp->Ki = new_id_matrix(n);
  if(gp->d > 0) {
    Kchol = new_dup_matrix(gp->K, n, n);
    info = linalg_dposv(n, Kchol, gp->Ki);
    if(info) {
#ifdef UNDEBUG
      printMatrix(gp->K, n, n, stdout);
#endif
      error("bad Cholesky decomp (info=%d), d=%g, g=%g", 
            info, gp->d, gp->g);
    }
    gp->ldetK = log_determinant_chol(Kchol, n);
    delete_matrix(Kchol);
  } else gp->ldetK = 0.0;

  /* phi <- t(Z) %*% Ki %*% Z */
  gp->KiZ = NULL;
  calc_ZtKiZ(gp);

  /* calculate derivatives and Fisher info based on them ? */
  gp->dK = gp->d2K = NULL; 
  gp->F = 0;
  /* first with NULL values then for if dk=TRUE */
  if(dK) newdKGP(gp);

  /* return new structure */
  return(gp);
}


/*
 * newGP:
 *
 * allocate a new GP structure using the data and parameters
 * provided
 */ 

GP* newGP(const unsigned int m, const unsigned int n, double **X,
	  double *Z, const double d, const double g, const int dK)
{
  GP* gp;

  /* new gp structure */
  gp = (GP*) malloc(sizeof(GP));
  gp->m = m;
  gp->n = n;
  gp->X = new_dup_matrix(X, n, m);
  gp->Z = new_dup_vector(Z, n);
  gp->d = d;
  gp->g = g;
  gp->K = NULL;

  return buildGP(gp, dK);
}


/*
 * newGP_sub:
 *
 * allocate a new GP structure using the parameters
 * provided, and the subset (rows) of the data specified by p
 */ 

GP* newGP_sub(const unsigned int m, const unsigned int n, int *p, 
	      double **X, double *Z, const double d, const double g, const int dK)
{
  unsigned int i;
  GP* gp;

  /* new gp structure */
  gp = (GP*) malloc(sizeof(GP));
  gp->m = m;
  gp->n = n;
  gp-> X = new_p_submatrix_rows(p, X, n, gp->m, 0);
  gp->Z = new_vector(n);
  for(i=0; i<n; i++) gp->Z[i] = Z[p[i]];
  gp->d = d;
  gp->g = g;
  gp->K = NULL;

  return buildGP(gp, dK);
}


/*
 * newGP_R:
 *
 * R-interface initializing a new GP, allocating and
 * assigning values to the global variables, which are
 * written over if already in use
 */

void newGP_R(/* inputs */
	     int *m_in,
	     int *n_in,
	     double *X_in,
	     double *Z_in,
	     double *d_in,
	     double *g_in,
	     int *dK,
	     
	     /* outputs */
	     int *gp_index)
{
  double **X;

  /* assign a new gp index */
  *gp_index = get_gp();

  /* create a new GP; */
  X = new_matrix_bones(X_in, *n_in, *m_in);
  gps[*gp_index] = newGP(*m_in, *n_in, X, Z_in, *d_in, *g_in, *dK);
  free(X);
}



/*
 * newparamsGP:
 *
 * change the lengthscale and nugget parameters to the gp
 */ 

void newparamsGP(GP* gp, const double d, const double g)
{
  int info, m, n;
  double **Kchol;

  /* sanity check */
  assert(d >= 0 && g >= 0);
  if(d == 0) assert(gp->dK == 0);

  /* build covariance matrix */
  m = gp->m; n = gp->n;
  gp->d = d;
  gp->g = g;
  if(d > 0) covar_symm(m, gp->X, n, d, g, gp->K);
  else id(gp->K, n);
  
  /* invert covariance matrix */
  id(gp->Ki, n);
  if(d > 0) {
    Kchol = new_dup_matrix(gp->K, n, n);
    info = linalg_dposv(n, Kchol, gp->Ki);
    if(info) {
#ifdef UNDEBUG
      printMatrix(gp->K, n, n, stdout);
#endif
      error("bad Cholesky decomp (info=%d), d=%g, g=%g", info, d, g);
    }
    gp->ldetK = log_determinant_chol(Kchol, n);
    delete_matrix(Kchol);
  } else gp->ldetK = 0.0;

  /* phi <- t(Z) %*% Ki %*% Z */
  calc_ZtKiZ(gp);

  /* calculate derivatives and Fisher info based on them ? */
  if(gp->dK) {
    diff_covar_symm(m, gp->X, n, gp->d, gp->dK, gp->d2K);
    gp->F = fishinfoGP(gp);
  } else { 
    gp->dK = gp->d2K = NULL; 
    gp->F = 0; /* null value of fisher information */
  }
}


/*
 * newparamsGP_R:
 *
 * R-interface allowing the internal/global GP representation
 * to change its parameterization without destroying the
 * memory and then re-allocating it
 */

void newparamsGP_R(/* inputs */
		int *gpi_in,
		double *d_in,
		double *g_in)
{
  GP *gp;
  unsigned int gpi;

  /* get the cloud */
  gpi = *gpi_in;
  if(gps == NULL || gpi >= NGP || gps[gpi] == NULL) 
    error("gp %d is not allocated\n", gpi);
  gp = gps[gpi];

  /* check if any are old */
  if(*d_in <= 0) *d_in = gp->d;
  if(*g_in < 0) *g_in = gp->g;

  /* call real C routine */
  newparamsGP(gp, *d_in, *g_in);
}


/*
 * llikGP:
 *
 * calculate and return the log marginal likelihood
 *
 */

double llikGP(GP *gp, double *dab, double *gab)
{
  double llik;

  /* proportional to likelihood calculation */
  llik = 0.0 - 0.5*(((double) gp->n) * log(0.5 * gp->phi) + gp->ldetK);
  // myprintf(mystdout, "d=%g, g=%g, phi=%g, llik=%g\n", gp->d, gp->g, gp->phi, llik); 
  /* llik += lgamma(0.5*((double) gp->n)) - ((double) gp->n)*M_LN_SQRT_2PI; */

  /* if priors are being used; for lengthscale */
  if(gp->d > 0 && dab && dab[0] > 0 && dab[1] > 0) 
    llik += dgamma(gp->d, dab[0], 1.0/dab[1], 1);

  /* if priors are being used; for nugget */
  if(gp->g > 0 && gab && gab[0] > 0 && gab[1] > 0) 
    llik += dgamma(gp->g, gab[0], 1.0/gab[1], 1);

  return(llik);
}


/*
 * llikGP_R:
 *
 * R-interface to calculate the marginal likelihood of a GP
 */

void llikGP_R(/* inputs */
	      int *gpi_in,
        double *dab_in,
        double *gab_in,

	      /* outputs */
	      double *llik_out)
{
  GP *gp;
  unsigned int gpi;

  /* get the cloud */
  gpi = *gpi_in;
  if(gps == NULL || gpi >= NGP || gps[gpi] == NULL) 
    error("gp %d is not allocated\n", gpi);
  gp = gps[gpi];

  /* calculate log likelihood */
  *llik_out = llikGP(gp, dab_in, gab_in);
}


/*
 * utility structure for fcnllik and fcndllik defined below
 * for use with Brent_fmin (R's optimize) or uniroot
 */

struct callinfo {
  Theta theta;
  GP *gp;
  double *ab;
  int its;
  int verb;
};

/*
 * fcnllik:
 * 
 * a utility function for Brent_fmin (R's optimize) to apply to the GP
 * log likelihood after changes to the lengthscale or nugget parameter 
 */

static double fcnnllik(double x, struct callinfo *info)
{
  double llik;
  (info->its)++;
  if(info->theta == LENGTHSCALE) {
    newparamsGP(info->gp, x, info->gp->g);
    llik = llikGP(info->gp, info->ab, NULL);
     if(info->verb > 1) 
      myprintf(mystdout, "fmin it=%d, d=%g, llik=%g\n", info->its, info->gp->d, llik);
  } else {
    newparamsGP(info->gp, info->gp->d, x);
    llik = llikGP(info->gp, NULL, info->ab);
    if(info->verb > 1)
      myprintf(mystdout, "fmin it=%d, g=%g, llik=%g\n", info->its, info->gp->g, llik);
  }
  return 0.0-llik;
} 


/*
 * Ropt:
 *
 * use R's Brent Fmin routine (from optimize) to optimize
 */

double Ropt(GP* gp, Theta theta, double tmin, double tmax, 
                   double *ab, char *msg, int *its, int verb)
{
  double tnew, th;
  // double ax, bx, fa, fb;
  double Tol = SDEPS;
  // int Maxit = 100;

  /* sanity check */
  assert(tmin < tmax);

  /* get parameter */
  if(theta == LENGTHSCALE) th = gp->d;
  else th = gp->g;

  /* create structure for Brent_fmin */
  struct callinfo info;
  info.gp = gp;
  info.theta = theta;
  info.ab = ab;
  info.its = 0;
  info.verb = verb;

  /* call the C-routine behind R's optimize function */
  while(1) { /* check to make sure solution is not on boundary */
   tnew = Brent_fmin(tmin, tmax, (double (*)(double, void*)) fcnnllik, &info, Tol);  
   if(tnew > tmin && tnew < tmax) break;
   if(tnew == tmin) { /* left boundary found */
    tmin *= 2;
    if(verb > 0) myprintf(mystdout, "Ropt: tnew=tmin, increasing tmin=%g\n", tmin);
   } else { /* right boundary found */
    tmax /= 2.0;
    if(verb > 0) myprintf(mystdout, "Ropt: tnew=tmax, decreasing tmax=%g\n", tmax);
  }
  /* check that boundaries still valid */
  if(tmin >= tmax) error("unable to opimize in fmin()");
  } 

  /* check that last value agrees with GP parameterization */
  if(theta == LENGTHSCALE) {
    if(gp->d != tnew) newparamsGP(gp, tnew, gp->g);
  } else {
    if(gp->g != tnew) newparamsGP(gp, gp->d, tnew);
  }

  /* possible print message and return */
  if(verb > 0) myprintf(mystdout, "Ropt %s: told=%g -[%d]-> tnew=%g\n",
			msg, th, info.its, tnew);

  *its += info.its;
  return(tnew);
}


/*
 * mleGP:
 *
 * calculate the MLE with respect to the lengthscale parameter;
 * requires that derivatives be pre-calculated; uses Newton's
 * method initialized at the current gp->d value
 */

double mleGP(GP* gp, Theta theta, double tmin, double tmax, double *ab, 
             int *its, int verb)
{
  double tnew, dllik, d2llik, llik_init, llik_new, adj, rat;
  double th;
  double *gab, *dab;
  int restoredKGP;

  /* set priors based on Theta */
  dab = gab = NULL;
  if(theta == LENGTHSCALE) dab = ab;
  else gab = ab;
  
  /* initialization */
  *its = 0;
  restoredKGP = 0;
  /* theta parameter is d or g */
  if(theta == LENGTHSCALE) th = gp->d;
  else th = gp->g;

  /* check how close we are to tmin */
  if(theta == NUGGET && fabs(th - tmin) < SDEPS) {
    if(verb > 0) myprintf(mystdout, "(g=%g) -- starting too close to min (%g)\n", th, tmin);
    goto alldone;
  }

  /* initial likelihood calculation */
  llik_init = llikGP(gp, dab, gab);

  /* initial printing */
  if(verb > 0) {
    if(theta == LENGTHSCALE)
      myprintf(mystdout, "(d=%g, llik=%g) ", gp->d, llik_init);
    else 
      myprintf(mystdout, "(g=%g, llik=%g) ", gp->g, llik_init);
  } if(verb > 1) myprintf(mystdout, "\n");

  while(1) { /* checking for improved llik */
    while(1) {  /* Newton step(s) */
      llik_new = 0.0-1e300*1e300;
      while(1) {  /* Newton proposal */

	      /* calculate first and second derivatives */
	      if(theta == LENGTHSCALE) dllikGP(gp, dab, &dllik, &d2llik);
        else dllikGP_nug(gp, gab, &dllik, &d2llik);

        /* check for convergence by root */
        if(fabs(dllik) < SDEPS) {
          if(*its == 0) {
            if(verb > 0) myprintf(mystdout, "-- Newton not needed\n");
            goto alldone;
          } else goto newtondone;
        }

	      /* Newton update */
	      rat = dllik/d2llik; adj = 1.0; (*its)++;

        /* check if we're going the right way */
        if((dllik < 0 && rat < 0) || (dllik > 0 && rat > 0)) {
          if(!gp->dK && restoredKGP == 0) { deletedKGP(gp); restoredKGP = 1; }
          th = Ropt(gp, theta, tmin, tmax, ab, "[slip]", its, verb); goto mledone; 
        } else tnew = th - adj*rat;  /* right way: Newton: */

	      /* check that we haven't proposed a tnew out of range */
	      while((tnew <= tmin || tnew >= tmax) && adj > SDEPS) {
	        adj /= 2.0; tnew = th - adj*rat;
	      }

	      /* if still out of range, restart? */
	      if(tnew <= tmin || tnew >= tmax) { 
          if(!gp->dK && restoredKGP == 0) { deletedKGP(gp); restoredKGP = 1; }
	        th = Ropt(gp, theta, tmin, tmax, ab, "[range]", its, verb);
          goto mledone;
	      } else break;
      } /* end inner while -- Newton proposal */

      /* else, resets gp->d = tnew */
      if(theta == LENGTHSCALE) newparamsGP(gp, tnew, gp->g);
      else { /* NUGGET; possibly deleted derivatives */
        if(!gp->dK && restoredKGP == 0) { deletedKGP(gp); restoredKGP = 1; }
        newparamsGP(gp, gp->d, tnew);
      }

      /* print progress */
      if(verb > 1) myprintf(mystdout, "\ti=%d theta=%g, c(a,b)=(%g,%g)\n", 
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
    llik_new = llikGP(gp, dab, gab);
    if(llik_new < llik_init-SDEPS) { 
      if(verb > 0) myprintf(mystdout, "llik_new = %g\n", llik_new);
      llik_new = 0.0-1e300*1e300;
      if(!gp->dK && restoredKGP == 0) { deletedKGP(gp); restoredKGP = 1; }
      th = Ropt(gp, theta, tmin, tmax, ab, "[dir]", its, verb); 
      goto mledone;
    } else break;
  } /* outer improved llik check while(1) loop */

  /* capstone progress indicator */
mledone:
  if(!R_FINITE(llik_new)) llik_new = llikGP(gp, dab, gab);
  if(verb > 0) {
    if(theta == LENGTHSCALE)
      myprintf(mystdout, "-> %d Newtons -> (d=%g, llik=%g)\n", 
               *its, gp->d, llik_new);
    else myprintf(mystdout, "-> %d Newtons -> (g=%g, llik=%g)\n", 
               *its, gp->g, llik_new);
  }

  /* return theta-value found */
alldone:
  if(restoredKGP) newdKGP(gp);
  return th;
}


/*
 * mleGP_R:
 *
 * R-interface to update the GP to use its MLE lengthscale
 * parameterization using the current data
 */

void mleGP_R(/* inputs */
	     int *gpi_in,
       int *param_in,
	     int *verb_in,
	     double *tmin_in,
	     double *tmax_in,
       double *ab_in,

	     /* outputs */
	     double *mle_out,
	     int * its_out)
{
  GP *gp;
  unsigned int gpi;

  /* get the cloud */
  gpi = *gpi_in;
  if(gps == NULL || gpi >= NGP || gps[gpi] == NULL) 
    error("gp %d is not allocated\n", gpi);
  gp = gps[gpi];

  /* check param */
  Theta theta = LENGTHSCALE;
  if(*param_in == 2) theta = NUGGET;
  else if(*param_in != 1) error("param must be 1 (d) or 2 (g)");

  /* check theta and tmax */
  if(*tmin_in <= 0) *tmin_in = SDEPS; 
  if(theta == LENGTHSCALE) {
    if(*tmax_in <= 0) *tmax_in = sq((double) gp->m);
    if(gp->d >= *tmax_in) error("d=%g >= tmax=%g\n", gp->d, *tmax_in);
    else if(gp->d <= *tmin_in) error("d=%g <= tmin=%g\n", gp->d, *tmin_in);
  } else {
    if(gp->g >= *tmax_in) error("g=%g >= tmax=%g\n", gp->g, *tmax_in);
    else if(gp->g <= *tmin_in) error("g=%g <= tmin=%g\n", gp->g, *tmin_in);
  }

  /* check a & b */
  if(ab_in[0] < 0 || ab_in[1] < 0) error("ab_in must be a positive 2-vector");

  /* double check that derivatives have been calculated */
  if(theta == LENGTHSCALE && !gp->dK) 
    error("derivative info not in gp; use newGP with dK=TRUE");

  /* call C-side MLE */
  *mle_out = mleGP(gp, theta, *tmin_in, *tmax_in, ab_in, 
                 its_out, *verb_in);
}


/*
 * jmleGP:
 *
 * calculate joint mle for lengthscale (d) and nugget (g) by a coordinite-
 * wise search, iterating over d and g searches via mleGP
 */

void jmleGP(GP *gp, double *drange, double *grange, double *dab, double *gab,
            int *dits, int *gits, int verb) 
  {
    unsigned int i;
    int dit, git;

    /* sanity checks */
    assert(gab && dab);
    assert(grange && drange);

    /* loop over coordinate-wise iterations */
    *dits = *gits = 0;
    for(i=0; i<100; i++) {
      mleGP(gp, LENGTHSCALE, drange[0], drange[1], dab, &dit, verb);
      *dits += dit;
      mleGP(gp, NUGGET, grange[0], grange[1], gab, &git, verb);
      *gits += git;
      if(dit <= 1 && git <= 1) break;
    }
    if(i == 100 && verb > 0) warning("max outer its (N=100) reached");
  }


/*
 * jmleGP_R:
 *
 * R-interface to update the GP to use its joint MLE (lengthscale
* and nugget) parameterization using the current data
 */

void jmleGP_R(/* inputs */
       int *gpi_in,
       int *verb_in,
       double *drange_in,
       double *grange_in,
       double *dab_in,
       double *gab_in, 

       /* outputs */
       double *d_out,
       double *g_out,
       int *dits_out,
       int *gits_out)
{
  GP *gp;
  unsigned int gpi;

  /* get the cloud */
  gpi = *gpi_in;
  if(gps == NULL || gpi >= NGP || gps[gpi] == NULL) 
    error("gp %d is not allocated\n", gpi);
  gp = gps[gpi];

  /* check theta and tmax */
  assert(drange_in[0] >= 0 && drange_in[0] < drange_in[1]);
  assert(grange_in[0] >= 0 && grange_in[0] < grange_in[1]);
  if(gp->d < drange_in[0] || gp->d > drange_in[1])
    error("gp->d=%g outside drange=[%g,%g]", gp->d, drange_in[0], drange_in[1]);
  if(gp->g < grange_in[0] || gp->g > grange_in[1])
    error("gp->g=%g outside grange=[%g,%g]", gp->g, grange_in[0], grange_in[1]);

  /* double check that derivatives have been calculated */
  if(! gp->dK) 
    error("derivative info not in gp; use newGP with dK=TRUE");

  /* call C-side MLE */
  jmleGP(gp, drange_in, grange_in, dab_in, gab_in, dits_out, gits_out, *verb_in);

  /* write back d and g */
  *d_out = gp->d;
  *g_out = gp->g;
}


/*
 * copyGP:
 * 
 * allocate space for a new GP structure filled with the 
 * contents of an old one
 */

GP* copyGP(GP* gp)
{
  GP *new_gp;

  new_gp = (GP*) malloc(sizeof(GP));
  new_gp->m = gp->m;
  new_gp->n = gp->n;
  new_gp->X = new_dup_matrix(gp->X, new_gp->n, new_gp->m);
  new_gp->Z = new_dup_vector(gp->Z, new_gp->n);
  new_gp->K = new_dup_matrix(gp->K, new_gp->n, new_gp->n);
  new_gp->Ki = new_dup_matrix(gp->Ki, new_gp->n, new_gp->n);
  if(gp->dK) new_gp->dK = new_dup_matrix(gp->dK, new_gp->n, new_gp->n);
  else new_gp->dK = NULL;
  if(gp->d2K) new_gp->d2K = new_dup_matrix(gp->d2K, new_gp->n, new_gp->n);
  else new_gp->d2K = NULL;
  new_gp->F = gp->F;
  new_gp->ldetK = gp->ldetK;
  new_gp->d = gp->d;
  new_gp->g = gp->g;
  new_gp->phi = gp->phi;
  new_gp->KiZ = new_dup_vector(gp->KiZ, new_gp->n);

  return(new_gp);
}


/*
 * copyGP_R:
 *
 * R-interface to copy a GP
 */

void copyGP_R(/* inputs */
	      int *gpi_in,

	      /* outputs */
	      int *newgpi_out)
{
  GP *gp;
  unsigned int gpi;

  /* get the cloud */
  gpi = *gpi_in;
  if(gps == NULL || gpi >= NGP || gps[gpi] == NULL) 
    error("gp %d is not allocated\n", gpi);
  gp = gps[gpi];

  /* assign a new gp index */
  *newgpi_out = get_gp();

  /* allocate a new gp */
  gps[*newgpi_out] = copyGP(gp);
}


/*
 * updateGP:
 *
 * quickly augment (O(n^2)) a gp based on new X-Z pairs.  
 * Uses the Bartlet partition inverse equations
 */

void updateGP(GP* gp, unsigned int nn, double **XX, double *ZZ, 
	           int verb)
{
  unsigned int i, j, n, m;
  double *kx, *x, *gvec;
  double mui, Ztg;
  double **Gmui, **temp, **d2temp;  
  
  /* allocate space */
  n = gp->n; m = gp->m;
  kx = new_vector(n);
  gvec = new_vector(n);
  Gmui = new_matrix(n, n);
  temp = new_matrix(1,1);
  if(gp->dK) d2temp = new_matrix(1,1);
  else d2temp = NULL;

  /* for each new location */
  for(j=0; j<nn; j++) {

    /* shorthand for x being updated */
    x = XX[j];

    /* update approx fisher information, must call before
       actually upating the GP sufficient information */
    if(gp->dK) update_fishinfoGP(gp, x, ZZ[j]);

    /* calculate the Bartlet quantities */
    calc_g_mui_kxy(m, x, gp->X, n, gp->Ki, NULL, 0, gp->d, 
		               gp->g, gvec, &mui, kx, NULL);

    /* Gmui = g %*% t(g)/mu */
    linalg_dgemm(CblasNoTrans,CblasTrans,n,n,1,
		 mui,&gvec,n,&gvec,n,0.0,Gmui,n);
    
    /* Ki = Ki + Gmui */
    linalg_daxpy(n*n, 1.0, *Gmui, 1, *(gp->Ki), 1);
    
    /* now augment covariance matrices */
    /* for nn > 1 might be better to make bigger matricies once
       outside the for-loop */
    gp->Ki = new_bigger_matrix(gp->Ki, n, n, n+1, n+1);
    for(i=0; i<n; i++) gp->Ki[n][i] = gp->Ki[i][n] = gvec[i];
    gp->Ki[n][n] = 1.0/mui;
    gp->K = new_bigger_matrix(gp->K, n, n, n+1, n+1);
    for(i=0; i<n; i++) gp->K[n][i] = gp->K[i][n] = kx[i];
    covar_symm(m, &x, 1, gp->d, gp->g, temp);
    gp->K[n][n] = **temp;

    /* update the determinant calculation */
    gp->ldetK += log(**temp + mui * linalg_ddot(n, gvec, 1, kx, 1));

    /* update KiZ and phi */
    /* Ztg = t(Z) %*% gvec */
    Ztg = linalg_ddot(n, gvec, 1, gp->Z, 1);
    gp->KiZ = realloc(gp->KiZ, sizeof(double)*(n+1));
    /* KiZ[1:n] += (Ztg/mu + Z*g) * gvec */
    linalg_daxpy(n, Ztg*mui + ZZ[j], gvec, 1, gp->KiZ, 1);
    /* KiZ[n+1] = Ztg + z*mu */
    gp->KiZ[n] = Ztg + ZZ[j]/mui;
    /* phi += Ztg^2/mu + 2*z*Ztg + z^2*mu */
    gp->phi += sq(Ztg)*mui + 2.0*ZZ[j]*Ztg + sq(ZZ[j])/mui;    

    /* now augment X and Z */
    gp->X = new_bigger_matrix(gp->X, n, m, n+1, m);
    dupv(gp->X[n], x, m);
    gp->Z = (double*) realloc(gp->Z, sizeof(double)*(n+1));
    gp->Z[n] = ZZ[j];
    (gp->n)++;

    /* augment derivative covariance matrices, re-using kx memory */
    /* maybe move to new updateDGP function */
    if(gp->dK) {
      assert(gp->d2K); assert(d2temp);
      gp->dK = new_bigger_matrix(gp->dK, n, n, n+1, n+1);
      gp->d2K = new_bigger_matrix(gp->d2K, n, n, n+1, n+1);
      diff_covar(m, &x, 1, gp->X, n, gp->d, &(gp->dK[n]), &(gp->d2K[n]));
      for(i=0; i<n; i++) { gp->dK[i][n] = gp->dK[n][i]; gp->d2K[i][n] = gp->d2K[n][i]; }
      diff_covar_symm(m, &x, 1, gp->d, temp, d2temp);
      gp->dK[n][n] = **temp;  gp->d2K[n][n] = **d2temp;
    } else { assert(!gp->d2K); assert(!d2temp); }

    /* if more then re-allocate */
    if(j < nn-1) {
      kx = (double*) realloc(kx, sizeof(double)*(n+1));
      gvec = (double*) realloc(gvec, sizeof(double)*(n+1));
      Gmui = new_bigger_matrix(Gmui, n, n, n+1, n+1);
    }

    /* progress meter? */
    if(verb > 0)
      myprintf(mystdout, "update j=%d, n=%d, ldetK=%g\n", j+1, gp->n, gp->ldetK);
    n = gp->n; /* increment for next interation */
  }

  /* clean up */
  delete_matrix(Gmui);
  free(kx);
  free(gvec);
  delete_matrix(temp);
  if(d2temp) delete_matrix(d2temp);
}


/*
 * updateGP_R:
 *
 * R-interface allowing the internal/global GP representation
 * to be quickly augment (O(n^2)) based on new X-Z pairs.  
 * Uses the Bartlet partition inverse equations
 */

void updateGP_R(/* inputs */
		int *gpi_in,
		int *m_in,
		int *nn_in,
		double *XX_in,
		double *ZZ_in,
		int *verb_in)
{
  GP *gp;
  unsigned int gpi;
  double **XX;

  /* get the cloud */
  gpi = *gpi_in;
  if(gps == NULL || gpi >= NGP || gps[gpi] == NULL) 
    error("gp %d is not allocated\n", gpi);
  gp = gps[gpi];
  if((unsigned) *m_in != gp->m)  
    error("ncol(X)=%d does not match GP/C-side (%d)", *m_in, gp->m);

  /* check that this is not a degenerate GP: not implemented (yet) */
  if(gp->d <= 0) error("updating degenerate GP (d=0) not supported");

  /* sanity check and XX representation */
  XX = new_matrix_bones(XX_in, *nn_in, gp->m);

  /* call real C routine */
  updateGP(gp, *nn_in, XX, ZZ_in, *verb_in);

  /* clean up */
  free(XX);
}


/*
 * predGP:
 *
 * return the student-t predictive equations,
 * i.e., parameters to a multivatiate t-distribution
 * for XX predictive locations of dimension (n*m)
 */

void predGP(GP* gp, unsigned int nn, double **XX, double *mean, 
	    double **Sigma, double *df, double *llik)
{
  unsigned int i, j, m, n;
  double **k, **ktKi, **ktKik;
  double phidf;

  /* easier referencing for dims */
  n = gp->n;  m = gp->m;

  /* variance (s2) components */
  *df = (double) n; 
  phidf = gp->phi/(*df);

  /* calculate marginal likelihood (since we have the bits) */
  *llik = 0.0 - 0.5*((*df) * log(0.5 * gp->phi) + gp->ldetK);
  /* continuing: - ((double) n)*M_LN_SQRT_2PI;*/

  /* degenerate GP case; ignores nugget */
  if(gp->d == 0) {
    zerov(mean, nn);
    zerov(*Sigma, nn*nn);
    for(i=0; i<nn; i++) Sigma[i][i] = phidf;
    return;
  }

  /* k <- covar(X1=X, X2=XX, d=Zt$d, g=0) */
  k = new_matrix(n, nn);
  covar(m, gp->X, n, XX, nn, gp->d, 0.0, k);
  /* Sigma <- covar(X1=XX, d=Zt$d, g=Zt$g) */
  covar_symm(m, XX, nn, gp->d, gp->g, Sigma);
  
  /* ktKi <- t(k) %*% util$Ki */
  ktKi = new_matrix(n, nn);
  linalg_dsymm(CblasRight,nn,n,1.0,gp->Ki,n,k,nn,0.0,ktKi,nn);
  /* ktKik <- ktKi %*% k */
  ktKik = new_matrix(nn, nn);
  linalg_dgemm(CblasNoTrans,CblasTrans,nn,nn,n,
               1.0,k,nn,ktKi,nn,0.0,ktKik,nn);

  /* mean <- ktKi %*% Z */
  linalg_dgemv(CblasNoTrans,nn,n,1.0,ktKi,nn,gp->Z,1,0.0,mean,1);

  /* Sigma <- phi*(Sigma - ktKik)/df */
  for(i=0; i<nn; i++) {
     Sigma[i][i] = phidf * (Sigma[i][i] - ktKik[i][i]);
    for(j=0; j<i; j++)
      Sigma[j][i] = Sigma[i][j] = phidf * (Sigma[i][j] - ktKik[i][j]);
  }

  /* clean up */
  delete_matrix(k);
  delete_matrix(ktKi);
  delete_matrix(ktKik);
}


/*
 * new_predutilGP_lite:
 *
 * utility function that allocates and calculate useful vectors 
 * and matrices for prediction; used by predGP_lite and dmus2GP
 */

void new_predutilGP_lite(GP *gp, unsigned int nn, double **XX, double ***k, 
			 double ***ktKi, double **ktKik)
{
  unsigned int i, j, m, n;

  /* k <- covar(X1=X, X2=XX, d=Zt$d, g=0) */
  n = gp->n;  m = gp->m;
  *k = new_matrix(n, nn);
  covar(m, gp->X, n, XX, nn, gp->d, 0.0, *k);
  
  /* ktKi <- t(k) %*% util$Ki */
  *ktKi = new_matrix(n, nn);
  linalg_dsymm(CblasRight,nn,n,1.0,gp->Ki,n,*k,nn,0.0,*ktKi,nn);
  /* ktKik <- diag(ktKi %*% k) */
  *ktKik = new_zero_vector(nn); 
  for(i=0; i<nn; i++) for(j=0; j<n; j++) (*ktKik)[i] += (*ktKi)[j][i]*(*k)[j][i];
}


/*
 * predGP_lite:
 *
 * return the student-t predictive equations,
 * i.e., parameters to a multivatiate t-distribution
 * for XX predictive locations of dimension (n*m);
 * lite because sigma2 not Sigma is calculated
 */

void predGP_lite(GP* gp, unsigned int nn, double **XX, double *mean, 
		 double *sigma2, double *df, double *llik)
{
  unsigned int i;
  double **k, **ktKi;
  double *ktKik;
  double phidf;
  
  /* sanity checks */
  assert(df);
  *df = gp->n; 

   /* degenerate GP case; ignores nugget */
  if(gp->d == 0) {
    if(mean) zerov(mean, nn);
    if(sigma2) { 
      phidf = gp->phi/(*df);
      for(i=0; i<nn; i++) sigma2[i] = phidf;
    }
    return;
  }

  /* utility calculations */
  new_predutilGP_lite(gp, nn, XX, &k, &ktKi, &ktKik);

  /* mean <- ktKi %*% Z */
  if(mean) linalg_dgemv(CblasNoTrans,nn,gp->n,1.0,ktKi,nn,gp->Z,1,0.0,mean,1);

  /* Sigma <- phi*(Sigma - ktKik)/df */
  /* *df = n - m - 1.0; */  /* only if estimating beta */
  if(sigma2) {
    phidf = gp->phi/(*df);
    for(i=0; i<nn; i++) sigma2[i] = phidf * (1.0 + gp->g - ktKik[i]);
  }

  /* calculate marginal likelihood (since we have the bits) */
  /* might move to updateGP if we decide to move phi to updateGP */
  if(llik) *llik = 0.0 - 0.5*(((double) gp->n) * log(0.5* gp->phi) + gp->ldetK);
  /* continuing: - ((double) n)*M_LN_SQRT_2PI;*/

  /* clean up */
  delete_matrix(k);
  delete_matrix(ktKi);
  free(ktKik);
}


/*
 * predGP_R:
 *
 * R-interface to C-side function that 
 * returns the student-t predictive equations,
 * i.e., parameters to a multivatiate t-distribution
 * for XX predictive locations of dimension (n*m)
 * using the stored GP parameterization
 */

void predGP_R(/* inputs */
	      int *gpi_in,
	      int *m_in,
	      int *nn_in,
	      double *XX_in,
	      int *lite_in,
	      
	      /* outputs */
	      double *mean_out,
	      double *Sigma_out,
	      double *df_out,
	      double *llik_out)
{
  GP* gp;
  unsigned int gpi;
  double **Sigma, **XX;

  /* get the gp */
  gpi = *gpi_in;
  if(gps == NULL || gpi >= NGP || gps[gpi] == NULL) 
    error("gp %d is not allocated\n", gpi);
  gp = gps[gpi];
  if((unsigned) *m_in != gp->m) 
    error("ncol(X)=%d does not match GP/C-side (%d)", *m_in, gp->m);

  /* sanity check and XX representation */
  XX = new_matrix_bones(XX_in, *nn_in, *m_in);
  if(! *lite_in) Sigma = new_matrix_bones(Sigma_out, *nn_in, *nn_in);
  else Sigma = NULL;

  /* call the C-only Predict function */
  if(*lite_in) predGP_lite(gp, *nn_in, XX, mean_out, Sigma_out, df_out, llik_out);
  else predGP(gp, *nn_in, XX, mean_out, Sigma, df_out, llik_out);
  
  /* clean up */
  free(XX);
  if(Sigma) free(Sigma);
}


/*
 * utility structure for fcnnalc and defined below
 * for use with Brent_fmin (R's optimize) or uniroot
 */

struct alcinfo {
  double **Xstart;
  double **Xend;
  double **Xref;
  GP *gp;
  double **k;
  double *gvec;
  double *kxy;
  double *kx;
  double *ktKikx;
  double **Gmui;
  double *ktGmui;
  double *Xcand;
  double s2p[2];
  double df;
  double mui;
  int its;
  int verb;
};


static double fcnnalc(double x, struct alcinfo *info)
{
  int m, n, j;
  double alc;

  m = info->gp->m;
  n = info->gp->n;
  (info->its)++;

  /* calculate Xcand along the ray */
  for(j=0; j<m; j++) info->Xcand[j] = (1.0 - x)*(info->Xstart[0][j]) + x*(info->Xend[0][j]); 
    
  /* calculate the g vector, mui, and kxy */
  calc_g_mui_kxy(m, info->Xcand, info->gp->X, n, info->gp->Ki, info->Xref, 
    1, info->gp->d, info->gp->g, info->gvec, &(info->mui), info->kx, info->kxy);

  /* skip if numerical problems */
  if(info->mui <= SDEPS) alc = 0.0 - 1e300 * 1e300;
  else {
    /* use g, mu, and kxy to calculate ktKik.x */
    calc_ktKikx(NULL, 1, info->k, n, info->gvec, info->mui, info->kxy, info->Gmui, 
      info->ktGmui, info->ktKikx);
        
    /* calculate the ALC */
    alc = calc_alc(1, info->ktKikx, info->s2p, info->gp->phi, NULL, info->df, NULL);
  }

  /* progress meter */
  if(info->verb > 0) {
    myprintf(mystdout, "alcray eval i=%d, Xcand=", info->its);
    for(j=0; j<m; j++) myprintf(mystdout, "%g ", info->Xcand[j]);
    myprintf(mystdout, "(s=%g), alc=%g\n", x, alc);
  }

  return 0.0-alc;
} 


/* alcrayGP:
 *
 * optimize AIC via a ray search using the pre-stored GP representation.  
 * Return the convex combination s in (0,1) between Xstart and Xend
 */

double* alcrayGP(GP *gp, double **Xref, const unsigned int nump, 
  double **Xstart, double **Xend, double *negalc, const unsigned int verb)
{
  unsigned int m, n, r;
  struct alcinfo info;
  double Tol = SDEPS;
  double obj0, na;
  double *snew;

  /* degrees of freedom */
  m = gp->m;
  n = gp->n;
  info.df = (double) n;

  /* other copying/default parameters */
  info.verb = verb;
  info.its = 0;
  info.s2p[0] = info.s2p[1] = 0;

  /* copy input pointers */
  info.Xref = Xref;
  info.Xcand = new_vector(m);
  info.gp = gp;

  /* allocate g, kxy, and ktKikx vectors */
  info.gvec = new_vector(n);
  info.kxy = new_vector(1);
  info.kx = new_vector(n);
  info.ktKikx = new_vector(1);

  /* k <- covar(X1=X, X2=Xref, d=Zt$d, g=0) */
  info.k = new_matrix(1, n);
  covar(m, Xref, 1, gp->X, n, gp->d, 0.0, info.k);
  
  /* utility allocations */
  info.Gmui = new_matrix(n, n);
  info.ktGmui = new_vector(n);

  /* allocate snew */
  snew = new_vector(nump);

  /* loop ovewr all pairs Xstart and Xend */
  assert(nump > 0);
  for(r=0; r<nump; r++) {

    /* select the rth start and end pair */
    info.Xstart = Xstart + r;
    info.Xend = Xend + r;

    /* use the C-backend of R's optimize function */
    snew[r] = Brent_fmin(0.0, 1.0, (double (*)(double, void*)) fcnnalc, &info, Tol);  
    if(snew[r] < Tol) snew[r] = 0.0;

    /* check s=0, as multi-modal ALC may result in larger domain of attraction
       for larger s-values but with lower mode */
    if(snew[r] > 0.0) {
      obj0 = fcnnalc(0.0, &info);
      na = fcnnalc(snew[r], &info);
      if(obj0 < na) { snew[r] = 0.0; na = obj0; }
      if(negalc) negalc[r] = na;
    } else if(negalc) negalc[r] = fcnnalc(snew[r], &info);
  }

  /* clean up */
  delete_matrix(info.Gmui);
  free(info.ktGmui);
  free(info.ktKikx);
  free(info.gvec);
  free(info.kx);
  free(info.kxy);
  delete_matrix(info.k);
  free(info.Xcand);

  return(snew);
}

/* lalcrayGP:
 *
 * local search of via ALC on rays (see alcrayGP) which finds the element
 * of Xcand that is closest to the max ALC value along a random ray eminating
 * from the (one of the) closest Xcands to Xref.  The offset determines which
 * candidate the ray eminates from (0 being the NN).  On input this function
 * assumes that the rows od Xcand are ordered by distance to Xref
 */

int lalcrayGP(GP *gp, double **Xcand, const unsigned int ncand, double **Xref, 
  const unsigned int offset, unsigned int nr, double **rect, int verb)
{
  unsigned int m, j, k, i, mini, r, rmin, eoff; 
  double **Xstart, **Xend;
  double *s, *negalc;
  double sc, smin, mind, dist;

  /* gp dimension */
  m = gp->m; 

  /* check numrays argument */
  assert(nr > 0);
  if(nr > ncand) nr = ncand;

  /* allocation and initialization */
  Xend = new_matrix(nr, m);
  Xstart = new_matrix(nr, m);
  negalc = new_vector(nr);

  /* set up starting and ending pairs */
  for(r=0; r<nr; r++) {

    /* starting point of a ray */
    eoff = (offset + r) % ncand;
    dupv(Xstart[r], Xcand[eoff], m);

    /* ending point of ray */
    for(j=0; j<m; j++) Xend[r][j] = 10.0*(Xstart[r][j] - Xref[0][j]) + Xstart[r][j];

    /* adjusting Xend to fit in bounding box */
    for(j=0; j<m; j++) {
      if(Xend[r][j] < rect[0][j]) {
        sc = (rect[0][j] - Xstart[r][j])/(Xend[r][j] - Xstart[r][j]);
        for(k=0; k<m; k++) Xend[r][k] = (Xend[r][k] - Xstart[r][k])*sc + Xstart[r][k];
      } else if(Xend[r][j] > rect[1][j]) {
        sc = (rect[1][j] - Xstart[r][j])/(Xend[r][j] - Xstart[r][j]);
        for(k=0; k<m; k++) Xend[r][k] = (Xend[r][k] - Xstart[r][k])*sc + Xstart[r][k];
      }
    }
  }

  /* calculate ALC along ray */
  s = alcrayGP(gp, Xref, nr, Xstart, Xend, negalc, verb);
  
  /* find the best amongst the pairs */
  min(negalc, nr, &rmin);
  smin = s[rmin];

  /* find Xstar with smin */
  if(smin > 0.0) {

    /* re-using Xend */
    for(j=0; j<m; j++) Xend[rmin][j] = (1-smin)*Xstart[rmin][j] + smin*Xend[rmin][j];

    /* find the candidate closest to Xstar (Xend) */
    mini = -1;
    mind = 1e300*1e300;
    eoff = offset + nr; /* explicitly avoid searcing over Xstart locations */
    if(eoff >= ncand) eoff = 0;  /* unless there aren't any candidates left */
    for(i=eoff; i<ncand; i++) {
      dist = 0;
      for(j=0; j<m; j++) {
        dist += sq(Xend[rmin][j] - Xcand[i][j]);
        if(dist > mind) break;
      }
      if(dist > mind) continue;
      mind = dist;
      mini = i;
    }
  } else mini = (offset + rmin) % ncand;

  /* clean up */
  delete_matrix(Xstart);
  delete_matrix(Xend);
  free(s);
  free(negalc);

  return(mini);
}



/*
 * alcGP:
 *
 * return s2' component of the ALC calculation of the
 * expected reduction in variance calculation at locations 
 * Xcand averaging over reference locations Xref: 
 * ds2 = s2 - s2', where the s2s are at Xref and the
 * s2' incorporates Xcand, and everything is averaged
 * over Xref.
 */

void alcGP(GP *gp, unsigned int ncand, double **Xcand, unsigned int nref,
	   double **Xref,  int verb, double *alc)
{
  unsigned int m, n;
  int i;
  double **k, **Gmui;
  double *kx, *kxy, *gvec, *ktKikx, *ktGmui;
  double mui, df;
  double s2p[2] = {0, 0};

  /* degrees of freedom */
  m = gp->m;
  n = gp->n;
  df = (double) n;

  /* allocate g, kxy, and ktKikx vectors */
  gvec = new_vector(n);
  kxy = new_vector(nref);
  kx = new_vector(n);
  ktKikx = new_vector(nref);

  /* k <- covar(X1=X, X2=Xref, d=Zt$d, g=0) */
  k = new_matrix(nref, n);
  covar(m, Xref, nref, gp->X, n, gp->d, 0.0, k);
  
  /* utility allocations */
  Gmui = new_matrix(n, n);
  ktGmui = new_vector(n);

  /* calculate the ALC for each candidate */
  for(i=0; i<ncand; i++) {

    /* progress meter */
    if(verb > 0) myprintf(mystdout, "calculating ALC for point %d of %d\n", verb, i, ncand);
    
    /* calculate the g vector, mui, and kxy */
    calc_g_mui_kxy(m, Xcand[i], gp->X, n, gp->Ki, Xref, nref, gp->d, 
		               gp->g, gvec, &mui, kx, kxy);

    /* skip if numerical problems */
    if(mui <= SDEPS) {
      alc[i] = 0.0 - 1e300 * 1e300;
      continue;
    }

    /* use g, mu, and kxy to calculate ktKik.x */
    calc_ktKikx(NULL, nref, k, n, gvec, mui, kxy, Gmui, ktGmui, ktKikx);
        
    /* calculate the ALC */
    alc[i] = calc_alc(nref, ktKikx, s2p, gp->phi, NULL, df, NULL);
  }

  /* clean up */
  delete_matrix(Gmui);
  free(ktGmui);
  free(ktKikx);
  free(gvec);
  free(kx);
  free(kxy);
  delete_matrix(k);
}


#ifdef _OPENMP
/*
 * alcGP_omp:
 *
 * OpenMP version of alcGP, above
 */

void alcGP_omp(GP *gp, unsigned int ncand, double **Xcand, unsigned int nref,
     double **Xref,  int verb, double *alc)
{
  unsigned int m, n;
  double df;
  double **k;
  double s2p[2] = {0, 0};

  /* degrees of freedom */
  m = gp->m;
  n = gp->n;
  df = (double) n;

  /* k <- covar(X1=X, X2=Xref, d=Zt$d, g=0) */
  k = new_matrix(nref, n);
  covar(m, Xref, nref, gp->X, n, gp->d, 0.0, k);
  
  #pragma omp parallel
  {
    int i, me, nth;
    double **Gmui;
    double *kx, *kxy, *gvec, *ktKikx, *ktGmui;
    double mui;

    /* allocate g, kxy, and ktKikx vectors */
    gvec = new_vector(n);
    kxy = new_vector(nref);
    kx = new_vector(n);
    ktKikx = new_vector(nref);

    /* utility allocations */
    Gmui = new_matrix(n, n);
    ktGmui = new_vector(n);

    /* get thread information */
    me = omp_get_thread_num();
    nth = omp_get_num_threads();

    /* calculate the ALC for each candidate */
    for(i=me; i<ncand; i+=nth) {

      /* progress meter */
      #pragma omp master
      if(verb > 0) myprintf(mystdout, "calculating ALC for point %d of %d\n", verb, i, ncand);
    
      /* calculate the g vector, mui, and kxy */
      calc_g_mui_kxy(m, Xcand[i], gp->X, n, gp->Ki, Xref, nref, gp->d, 
                    gp->g, gvec, &mui, kx, kxy);

      /* skip if numerical problems */
      if(mui <= SDEPS) {
        alc[i] = 0.0 - 1e300 * 1e300;
        continue;
      }

      /* use g, mu, and kxy to calculate ktKik.x */
      calc_ktKikx(NULL, nref, k, n, gvec, mui, kxy, Gmui, ktGmui, ktKikx);
        
      /* calculate the ALC */
      alc[i] = calc_alc(nref, ktKikx, s2p, gp->phi, NULL, df, NULL);
    }

    /* clean up */
    delete_matrix(Gmui);
    free(ktGmui);
    free(ktKikx);
    free(gvec);
    free(kx);
    free(kxy);
  }

  /* clean up non-parallel stuff */
  delete_matrix(k);
}



/*
 * alcGP_omp_R:
 *
 * OpenMP version of alcGP_R interface
 */

void alcGP_omp_R(/* inputs */
       int *gpi_in,
       int *m_in,
       double *Xcand_in,
       int *ncand_in,
       double *Xref_in,
       int *nref_in,
       int *verb_in,
       
       /* outputs */
       double *alc_out)
{
  GP *gp;
  unsigned int gpi;
  double **Xcand, **Xref;

  /* get the gp */
  gpi = *gpi_in;
  if(gps == NULL || gpi >= NGP || gps[gpi] == NULL) 
    error("gp %d is not allocated\n", gpi);
  gp = gps[gpi];
  if((unsigned) *m_in != gp->m)  
    error("ncol(X)=%d does not match GP/C-side (%d)", *m_in, gp->m);

  /* make matrix bones */
  Xcand = new_matrix_bones(Xcand_in, *ncand_in, *m_in);
  Xref = new_matrix_bones(Xref_in, *nref_in, *m_in);

  /* call the C-only function */
  alcGP_omp(gp, *ncand_in, Xcand, *nref_in, Xref, *verb_in, alc_out);

  /* clean up */
  free(Xcand);
  free(Xref);
}
#endif



#ifdef _GPU
/*
 * alcGP_gpu:
 *
 * CUDA/NVIDIA GPU version of alcGP, above
 */

void alcGP_gpu(GP *gp, unsigned int ncand, double **Xcand, unsigned int nref,
     double **Xref,  int verb, double *alc, int omp_threadnum)
{
  double **k;

  /* k <- covar(X1=X, X2=Xref, d=Zt$d, g=0) */
  k = new_matrix(nref, gp->n);
  covar(gp->m, Xref, nref, gp->X, gp->n, gp->d, 0.0, k);

  alc_gpu(gp->d, gp->g, gp->phi, gp->m, gp->n, *(gp->X), *(gp->Ki), ncand, 
          *Xcand, nref, *Xref, *k, alc, omp_threadnum);
  
  /* clean up non-parallel stuff */
  delete_matrix(k);
}


/*
 * alcGP_gpu_R:
 *
 * GPU version of alcGP_R interface
 */

void alcGP_gpu_R(/* inputs */
       int *gpi_in,
       int *m_in,
       double *Xcand_in,
       int *ncand_in,
       double *Xref_in,
       int *nref_in,
       int *verb_in,
       
       /* outputs */
       double *alc_out)
{
  GP *gp;
  unsigned int gpi;
  double **Xcand, **Xref;

  /* get the gp */
  gpi = *gpi_in;
  if(gps == NULL || gpi >= NGP || gps[gpi] == NULL) 
    error("gp %d is not allocated\n", gpi);
  gp = gps[gpi];
  if((unsigned) *m_in != gp->m)  
    error("ncol(X)=%d does not match GP/C-side (%d)", *m_in, gp->m);

  /* make matrix bones */
  Xcand = new_matrix_bones(Xcand_in, *ncand_in, *m_in);
  Xref = new_matrix_bones(Xref_in, *nref_in, *m_in);

  /* call the C-only function */
  alcGP_gpu(gp, *ncand_in, Xcand, *nref_in, Xref, *verb_in, alc_out, 0);

  /* clean up */
  free(Xcand);
  free(Xref);
}
#endif


/* lalcrayGP_R:
 *
 * R interface to C-side function that implements a local search of via ALC 
 * on rays (see alcrayGP) which finds the element of Xcand that is closest 
 * to the max ALC value along a random ray eminating from the (one of the) 
 * closest Xcands to Xref.  The offset determines which candidate the ray 
 * eminates from (0 being the NN).  On input this function assumes that the 
 * rows od Xcand are ordered by distance to Xref
 */

void lalcrayGP_R(/* inputs */
       int *gpi_in,
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
  GP *gp;
  unsigned int gpi;
  double **Xref, **Xcand, **rect;

  /* get the gp */
  gpi = *gpi_in;
  if(gps == NULL || gpi >= NGP || gps[gpi] == NULL) 
    error("gp %d is not allocated\n", gpi);
  gp = gps[gpi];
  if((unsigned) *m_in != gp->m) 
    error("ncol(X)=%d does not match GP/C-side (%d)", *m_in, gp->m);

  /* check num rays */
  if(*numrays_in <= 0) error("numrays must me an interger scalar >= 1");

  /* make matrix bones */
  Xref = new_matrix_bones(Xref_in, 1, *m_in);
  Xcand = new_matrix_bones(Xcand_in, *ncand_in, *m_in);
  rect = new_matrix_bones(rect_in, 2, *m_in);

  /* call the C-only function */
  *w_out = lalcrayGP(gp, Xcand, *ncand_in, Xref, *offset_in, *numrays_in, rect, *verb_in);

  /* clean up */
  free(Xref);
  free(Xcand);
  free(rect);
}


/* alcrayGP_R:
 *
 * R interface to C-side function that optimizes AIC via a ray search 
 * using the pre-stored GP representation.  Return the convex 
 * combination s in (0,1) between Xstart and Xend
 */

void alcrayGP_R(
      /* inputs */
       int *gpi_in,
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
  GP *gp;
  unsigned int gpi, rui;
  double **Xref, **Xstart, **Xend;
  double *s, *negalc;

  /* get the gp */
  gpi = *gpi_in;
  if(gps == NULL || gpi >= NGP || gps[gpi] == NULL) 
    error("gp %d is not allocated\n", gpi);
  gp = gps[gpi];
  if((unsigned) *m_in != gp->m)  
    error("ncol(X)=%d does not match GP/C-side (%d)", *m_in, gp->m);

  /* check numrays */
  if(*numrays_in < 1)
    error("numrays should be a integer scalar >= 1");

  /* make matrix bones */
  Xref = new_matrix_bones(Xref_in, 1, *m_in);
  Xstart = new_matrix_bones(Xstart_in, *numrays_in, *m_in);
  Xend = new_matrix_bones(Xend_in, *numrays_in, *m_in);

  /* call the C-only function */
  negalc = new_vector(*numrays_in);
  s = alcrayGP(gp, Xref, *numrays_in, Xstart, Xend, negalc, *verb_in);

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


/*
 * alcGP_R:
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

void alcGP_R(/* inputs */
	     int *gpi_in,
	     int *m_in,
	     double *Xcand_in,
	     int *ncand_in,
	     double *Xref_in,
	     int *nref_in,
	     int *verb_in,
	     
	     /* outputs */
	     double *alc_out)
{
  GP *gp;
  unsigned int gpi;
  double **Xcand, **Xref;

  /* get the gp */
  gpi = *gpi_in;
  if(gps == NULL || gpi >= NGP || gps[gpi] == NULL) 
    error("gp %d is not allocated\n", gpi);
  gp = gps[gpi];
  if((unsigned) *m_in != gp->m) 
    error("ncol(X)=%d does not match GP/C-side (%d)", *m_in, gp->m);

  /* make matrix bones */
  Xcand = new_matrix_bones(Xcand_in, *ncand_in, *m_in);
  Xref = new_matrix_bones(Xref_in, *nref_in, *m_in);

  /* call the C-only function */
  alcGP(gp, *ncand_in, Xcand, *nref_in, Xref, *verb_in, alc_out);

  /* clean up */
  free(Xcand);
  free(Xref);
}


/*
 * alGP_R:
 *
 * R interface to C-side function that returns the a Monte Carlo approximation 
 * to the expected improvement (EI) and expected y-value (EY) under an augmented 
 * Lagrangian with constraint GPs (cgps) assuming a known linear objective 
 * function with scale bscale.  The constraints can be scaled with the cnorms
 */

void alGP_R(/* inputs */
       int *m_in,
       double *XX_in,
       int *nn_in,
       int *fgpi_in,
       double *fnorm_in,
       int *ncgps_in,
       int *cgpis_in,
       double *cnorms_in,
       double *lambda_in,
       double *alpha_in,
       double *ymin_in,
       int *nomax_in,
       int *N_in,
       
       /* outputs */
       double *eys_out,
       double *eis_out)
{
  GP **cgps, *fgp;
  unsigned int gpi, ncgps, i, j, k;
  double **cmu, **cs, **XX;
  double *mu, *s;
  double df;

  /* get the gps */
  ncgps = *ncgps_in;
  cgps = (GP**) malloc(sizeof(GP*) * ncgps);
  for(i=0; i<ncgps; i++) {
    gpi = cgpis_in[i];
    if(gps == NULL || gpi >= NGP || gps[gpi] == NULL) 
      error("gp %d is not allocated\n", gpi);
    cgps[i] = gps[gpi];
    if((unsigned) *m_in != cgps[i]->m)  
      error("ncol(X)=%d does not match GP/C-side (%d)", *m_in, cgps[i]->m);
  }

  /* make matrix bones */
  XX = new_matrix_bones(XX_in, *nn_in, *m_in);

  /* allocate storage for the (null) distribution of f */
  mu = new_vector(*nn_in);
  if(*fgpi_in >= 0) {
    gpi = *fgpi_in;
    if(gps == NULL || gpi >= NGP || gps[gpi] == NULL) 
      error("gp %d is not allocated\n", gpi);
    fgp = gps[gpi];
    s = new_vector(*nn_in);
    predGP_lite(fgp, *nn_in, XX, mu, s, &df, NULL);
    for(k=0; k<*nn_in; k++) s[k] = sqrt(s[k]);
  } else {
    for(k=0; k<*nn_in; k++) mu[k] = sumv(XX[k], cgps[0]->m);
    s = NULL;
  }

  /* allocate storage for means and variances under normal approx */
  cmu = new_matrix(ncgps, *nn_in);
  cs = new_matrix(ncgps, *nn_in);
  for(j=0; j<ncgps; j++) {
    predGP_lite(cgps[j], *nn_in, XX, cmu[j], cs[j], &df, NULL);
    for(k=0; k<*nn_in; k++) cs[j][k] = sqrt(cs[j][k]);
  }

  /* clean up */
  free(XX);
  free(cgps);

  GetRNGstate();

  /* use mu and s to calculate EI and EY */
  calc_al_eiey(ncgps, *nn_in, mu, s, *fnorm_in, cmu, cs, cnorms_in, 
    lambda_in, alpha_in, *ymin_in, *nomax_in, *N_in, eys_out, eis_out);

  PutRNGstate();

  /* clean up */
  delete_matrix(cmu);
  delete_matrix(cs);
  free(mu);
  if(s) free(s);
}


/*
 * d_Kik: 
 * 
 * calculates the derivative of Kik wrt d, the range parameter
 * using pre-calculated derivatives of k and K; uses dk as 
 * temporary space; each dk and ktKi and dktKi are wrt an input
 * location (x)
 */

void d_ktKi(const int n, double **Ki, const int nn, double **dk, 
	    double **dK, double **ktKi, double **dktKi)
{
  double **temp;

  /* allocate temporary vector */
  temp = new_dup_matrix(dk, n, nn);

  /* temp <- dk - dK %*% ktKi */
  linalg_dsymm(CblasRight,nn,n,0.0-1.0,dK,n,ktKi,nn,1.0,temp,nn);
  
  /* dktKi <- Ki %*% dk */
  linalg_dsymm(CblasRight,nn,n,1.0,Ki,n,temp,nn,0.0,dktKi,nn);

  /* clean up */
  delete_matrix(temp);
}


/*
 * d2_ktKi: 
 * 
 * calculates the second derivative of KiZ wrt d, the range parameter
 * using pre-calculated derivatives of k and K; each dk and ktKi and 
 * dktKi are wrt an input location (x)
 */

void d2_ktKi(const int n, double **Ki, const int nn, double **d2k, 
	       double **dK, double **d2K, double **ktKi, double **dktKi,
	       double **d2ktKi)
{
  double **temp;

  /* allocate temporary vector */
  temp = new_dup_matrix(d2k, n, nn);

  /* temp <- d2k - d2K %*% ktKi */
  linalg_dsymm(CblasRight,nn,n,0.0-1.0,d2K,n,ktKi,nn,1.0,temp,nn);

  /* temp <- temp - 2*dK %*% dktKi */
  linalg_dsymm(CblasRight,nn,n,0.0-2.0,dK,n,dktKi,nn,1.0,temp,nn);
  
  /* dktKi <- Ki %*% temp */
  linalg_dsymm(CblasRight,nn,n,1.0,Ki,n,temp,nn,0.0,d2ktKi,nn);

  /* clean up */
  delete_matrix(temp);
}


/*
 * d_KiZ: 
 * 
 * calculates the  derivative of Kik wrt d, the range parameter
 * using pre-calculated derivatives of k and K
 */

void d_KiZ(const int n, double **Ki, double **dK, double *KiZ, 
	   double *dKiZ)
{
  double *dKKiZ;

  /* allocate temporary vector */
  dKKiZ = new_vector(n);

  /* dKKiz <- dK %*% KiZ */
  linalg_dsymv(n,1.0,dK,n,KiZ,1,0.0,dKKiZ,1);

  /* dKiZ <- Ki %*% d2k */
  linalg_dsymv(n,0.0-1.0,Ki,n,dKKiZ,1,0.0,dKiZ,1);

  /* clean up */
  free(dKKiZ);
}


/*
 * d2_KiZ: 
 * 
 * calculates the second derivative of Kik wrt d, the range parameter
 * using pre-calculated derivatives of k and K
 */

void d2_KiZ(const int n, double **Ki, double **dK, double **d2K,
	    double *KiZ, double *dKiZ, double *d2KiZ)
{
  double *dKdKiZ2;

  /* allocate temporary vector */
  dKdKiZ2 = new_vector(n);

  /* dKdKiZ2 <- 2 * dK %*% dKiZ */
  linalg_dsymv(n,2.0,dK,n,dKiZ,1,0.0,dKdKiZ2,1);

  /* dKdKiZ <- dKdKiZ + d2K %*% KiZ */
  linalg_dsymv(n,1.0,d2K,n,KiZ,1,1.0,dKdKiZ2,1);

  /* dKiZ <- Ki %*% ddKdKiZ */
  linalg_dsymv(n,0.0-1.0,Ki,n,dKdKiZ2,1,0.0,d2KiZ,1);

  /* clean up */
  free(dKdKiZ2);
}


/*
 * dmus2GP:
 *
 * fills mu, dmu, d2mu, s2, ds2, d2s2 mean and scale and the derivatives of
 * the mean and scale of the predictive equation(s) at XX input locations of 
 * dimension (nn*m) using the stored GP parameterization
 */

void dmus2GP(GP* gp, unsigned int nn, double **XX, double *mu, double *dmu, 
	     double *d2mu, double *s2, double *ds2, double *d2s2)
{
  unsigned int i, j, n, m;
  double **k, **dk, **d2k, **ktKi, **dktKi, **d2ktKi, *ktKidk;
  double *dKiZ, *d2KiZ, *ktKik, *ktKid2k;
  double d2phi, dphi, dfi;

  /* pre-calculate useful vectors and matrices for prediction */
  n = gp->n;  m = gp->m;
  dfi = 1.0/(n-2.0);
  new_predutilGP_lite(gp, nn, XX, &k, &ktKi, &ktKik);

  /* maybe also calculate mean and variance */
  /* mean <- ktKi %*% Z */
  if(mu) linalg_dgemv(CblasNoTrans,nn,gp->n,1.0,ktKi,nn,gp->Z,1,0.0,mu,1);
  /* Sigma <- phi*(Sigma - ktKik)/df */
  if(s2) for(i=0; i<nn; i++) s2[i] = dfi*(gp->phi)*(1.0 + gp->g - ktKik[i]);

  /* now the derivatives */
  dk = new_matrix(n, nn);
  if(d2mu || d2s2) d2k = new_matrix(n, nn);
  else d2k = NULL;
  diff_covar(m, gp->X, n, XX, nn, gp->d, dk, d2k);
 
  /* derivatives of ktKi */
  dktKi = new_matrix(n, nn); /* first */
  d_ktKi(n, gp->Ki, nn, dk, gp->dK, ktKi, dktKi); 
  if(d2k) {
    d2ktKi = new_matrix(n, nn); /* second */
    d2_ktKi(n, gp->Ki, nn, d2k, gp->dK, gp->d2K, ktKi, dktKi, d2ktKi);
  } else d2ktKi = NULL;

  /* finishing mu derivative calculations */
  /* dmu <- dktKi %*% Z; d2mu <- d2ktKi %*% Zd */
  assert(dmu); 
  linalg_dgemv(CblasNoTrans,nn,n,1.0,dktKi,nn,gp->Z,1,0.0,dmu,1);
  if(d2ktKi) {
    assert(d2mu);
    linalg_dgemv(CblasNoTrans,nn,n,1.0,d2ktKi,nn,gp->Z,1,0.0,d2mu,1);
  }

  /* check if we need to do more calculating */
  if(ds2 || d2mu || d2s2) {

    /* calculate derivatives of KiZ */
    dKiZ = new_vector(n); /* first */
    d_KiZ(n, gp->Ki, gp->dK, gp->KiZ, dKiZ);
    dphi = linalg_ddot(n, gp->Z, 1, dKiZ, 1);
    if(d2mu || d2s2) {
      d2KiZ = new_vector(n); /* second */
      d2_KiZ(n, gp->Ki, gp->dK, gp->d2K, gp->KiZ, dKiZ, d2KiZ);
      d2phi = linalg_ddot(n, gp->Z, 1, d2KiZ, 1);
    } else { d2KiZ = NULL; d2phi = 0; }
    
    /* ktKidk <- diag(ktKi %*% dk + dktKi %*% k) */
    ktKidk = new_zero_vector(nn);
    for(i=0; i<nn; i++) 
      for(j=0; j<n; j++) 
	       ktKidk[i] += ktKi[j][i]*dk[j][i] + dktKi[j][i]*k[j][i];
    
    /* wrap up the ds2 calculation */
    assert(ds2);
    for(i=0; i<nn; i++) 
      ds2[i] = dfi*(dphi*(1.0+gp->g - ktKik[i]) - (gp->phi)*ktKidk[i]);
    
    /* d2ktKik <- d2k %*% Kik + 2.0 * dkt %*% dktKi + kt %*% d2ktKi */
    if(d2k) {
      ktKid2k = new_zero_vector(nn);
      for(i=0; i<nn; i++) for(j=0; j<n; j++) {
        ktKid2k[i] += ktKi[j][i]*d2k[j][i];
        ktKid2k[i] += 2.0*dktKi[j][i]*dk[j][i];
        ktKid2k[i] += d2ktKi[j][i]*k[j][i];
      }
    } else ktKid2k = NULL;
    
    /* wrap up the d2s2 calculation */
    if(ktKid2k) {
      assert(d2s2);
      for(i=0; i<nn; i++) 
	       d2s2[i] = dfi*(d2phi*(1 + gp->g - ktKik[i]) -
		       2.0*dphi*ktKidk[i] - (gp->phi)*ktKid2k[i]);
    }

    /* clean up */
    free(dKiZ); free(ktKidk); 
    if(ktKidk) free(d2KiZ); 
    if(ktKid2k) free(ktKid2k);
  } 
   
  /* clean up */
  free(ktKik);
  delete_matrix(dk); delete_matrix(dktKi); 
  delete_matrix(ktKi); delete_matrix(k);
  if(d2k) delete_matrix(d2k); 
  if(d2ktKi) delete_matrix(d2ktKi);
}



/*
 * dmus2GP_R:
 *
 * R-interface to C-side function that returns the 
 * derivative of the scale of the predictive equation(s) 
 * for XX predictive locations of dimension (nn*m)
 * using the stored GP parameterization
 */

void dmus2GP_R(/* inputs */
	      int *gpi_in,
	      int *m_in,
	      int *nn_in,
	      double *XX_in,
	      
	      /* outputs */
	      double *mu_out,
	      double *dmu_out,
	      double *d2mu_out,
	      double *s2_out,
	      double *ds2_out,
	      double *d2s2_out)
{
  GP* gp;
  unsigned int gpi;
  double **XX;

  /* get the gp */
  gpi = *gpi_in;
  if(gps == NULL || gpi >= NGP || gps[gpi] == NULL) 
    error("gp %d is not allocated\n", gpi);
  gp = gps[gpi];
  if((unsigned) *m_in != gp->m)  
    error("ncol(X)=%d does not match GP/C-side (%d)", *m_in, gp->m);

  /* double check that derivatives have been calculated */
  if(! gp->dK) error("derivative info not in gp; use buildKGP or newGP with dK=TRUE");

  /* sanity check and XX representation */
  XX = new_matrix_bones(XX_in, *nn_in, *m_in);

  /* call the C-only predict function */
  dmus2GP(gp, *nn_in, XX, mu_out, dmu_out, d2mu_out, s2_out, ds2_out, d2s2_out);
  
  /* clean up */
  free(XX);
}


/*
 * efiGP:
 *
 * fills efi with the expected (approx) Fisher information at the new
 * predictive locations XX of dimension (nn*m), using the stored 
 * GP parameterization.  Returns the absolute value (i.e., the 
 * determinant)
 */

void efiGP(GP* gp, unsigned int nn, double **XX, double *efi)
{
  double *dmu, *ds2, *s2;
  unsigned int i;

  /* allocate memory */
  dmu = new_vector(nn);
  ds2 = new_vector(nn);
  s2 = new_vector(nn);

  /* calculate first derivatives */
  dmus2GP(gp, nn, XX, NULL, dmu, NULL, s2, ds2, NULL);

  /* caluclate efi */
  for(i=0; i<nn; i++) 
    //efi[i] = fabs(gp->F + 0.5*sq(ds2[i]/s2[i]) + sq(dmu[i])/s2[i]);
    efi[i] = gp->F + 0.5*sq(ds2[i]/s2[i]) + sq(dmu[i])/s2[i];

  /* clean up */
  free(dmu);
  free(ds2);
  free(s2);
}


/*
 * efiGP_R:
 *
 * R-interface to C-side function that returns the 
 * expected (approx) Fisher information for new predictive 
 * locations XX of dimension (n*m) using the stored GP 
 * parameterization.  Returns the absolute value (i.e., the 
 * determinant)
 */

void efiGP_R(/* inputs */
	      int *gpi_in,
	      int *m_in,
	      int *nn_in,
	      double *XX_in,
	      
	      /* outputs */
	      double *efi_out)
{
  GP* gp;
  unsigned int gpi;
  double **XX;

  /* get the gp */
  gpi = *gpi_in;
  if(gps == NULL || gpi >= NGP || gps[gpi] == NULL) 
    error("gp %d is not allocated\n", gpi);
  gp = gps[gpi];
  if((unsigned) *m_in != gp->m) 
    error("ncol(X)=%d does not match GP/C-side (%d)", *m_in, gp->m);

  /* double check that derivatives have been calculated */
  if(! gp->dK) error("derivative info not in gp; use newGP with dK=TRUE");

  /* sanity check and XX representation */
  XX = new_matrix_bones(XX_in, *nn_in, *m_in);

  /* call the C-only predict function */
  efiGP(gp, *nn_in, XX, efi_out);
  
  /* clean up */
  free(XX);
}


/*
 * update_fishinfoGP:
 *
 * upaate the approximate Fisher information in light of a new
 * observation x
 */

void update_fishinfoGP(GP *gp, double *x, double z)
{
  double mu, dmu, d2mu, s2, ds2, d2s2, d2llik, zmmu, zmmu2, s4, tmp, dn, denom;
  
  /* calculate derivatives of predictive equations */
  dmus2GP(gp, 1, &x, &mu, &dmu, &d2mu, &s2, &ds2, &d2s2);

  /* utility calculations */
  dn = (double) gp->n;
  zmmu = z - mu;
  zmmu2 = sq(zmmu);
  s4 = sq(s2);
  denom = dn-2.0 + zmmu2/s2;

  /* approximate 2nd derivative of the log likelihood calculation */
  d2llik = 0.0 - 0.5*d2s2/s2;
  d2llik += 0.5*sq(ds2/s2);
  d2llik += 0.5*(dn+1.0) * sq(2.0*zmmu*dmu/s2 + zmmu2*ds2/s4) / sq(denom);

  /* second half of the calculation */
  tmp = 0.0 - 2.0*sq(dmu)/s2;
  tmp -= 4.0*zmmu*dmu*ds2/s4;
  tmp += 2.0*zmmu*d2mu/s2;
  tmp -= 2.0*zmmu2*sq(ds2)/(s4*s2);
  tmp += zmmu2*d2s2/s4;
  d2llik += 0.5*(dn+1.0) * tmp / denom;

  /* update approx fishinfo */
  gp->F -= d2llik;
}


/*
 * mseGP:
 *
 * returns the mean-squared-prediction-error sequential 
 * design criterial given the stored GP parameterization 
 * at locations Xcand averaging over reference locations Xref
 */

void mspeGP(GP *gp, unsigned int ncand, double **Xcand, unsigned int nref,
	    double **Xref, int fi, int verb, double *mspe)
{
  unsigned int i;
  double *dmu, *ds2, *s2, *pref;
  double dnp, dnp2, s2avg, dmu2avg, df;

  /* calculate reduction in variance, temporarily stored in mspe */
  alcGP(gp, ncand, Xcand, nref, Xref, verb, mspe);

  /* predict at reference locations */
  pref = new_vector(nref);
  predGP_lite(gp, nref, Xref, NULL, pref, &df, NULL);
  s2avg = meanv(pref, nref);

  /* get derivative of mean at reference locations */
  dmus2GP(gp, nref, Xref, NULL, pref, NULL, NULL, NULL, NULL);
  dmu2avg = 0.0;
  for(i=0; i<nref; i++) dmu2avg += sq(pref[i]);
  dmu2avg /= ((double) nref);
  free(pref);

  /* allocate memory */
  if(fi) {
    dmu = new_vector(ncand);
    ds2 = new_vector(ncand);
    s2 = new_vector(ncand); 
    
    /* calculate the first derivatives */
    dmus2GP(gp, ncand, Xcand, NULL, dmu, NULL, s2, ds2, NULL);
  } else dmu = ds2 = s2 = NULL;

  /* calculate mspe */
  dnp = (df + 1.0)/(df- 1.0);
  dnp2 = dnp*(df - 2.0)/df;
  for(i=0; i<ncand; i++) {
    mspe[i] = dnp*s2avg - dnp2*mspe[i];
    if(fi && gp->F > 0) 
      mspe[i] += dmu2avg / (gp->F + 0.5*sq(ds2[i]/s2[i]) + sq(dmu[i]/s2[i]));
  }

  /* clean up */
  if(fi) {
    free(s2);
    free(dmu);
    free(ds2);
  }
}


/*
 * mspeGP_R:
 *
 * R interface to C-side function that returns the
 * mean-squared-prediction-error sequential design 
 * criterial given the stored GP parameterization 
 * at locations Xcand averaging over reference 
 * locations Xref
 */

void mspeGP_R(/* inputs */
	     int *gpi_in,
	     int *m_in,
	     double *Xcand_in,
	     int *ncand_in,
	     double *Xref_in,
	     int *nref_in,
	     int *fi_in,
	     int *verb_in,
	     
	     /* outputs */
	     double *mspe_out)
{
  GP *gp;
  unsigned int gpi;
  double **Xcand, **Xref;

  /* get the gp */
  gpi = *gpi_in;
  if(gps == NULL || gpi >= NGP || gps[gpi] == NULL) 
    error("gp %d is not allocated\n", gpi);
  gp = gps[gpi];
  if((unsigned) *m_in != gp->m) 
    error("ncol(X)=%d does not match GP/C-side (%d)", *m_in, gp->m);
  
  /* double check that derivatives have been calculated */
  if(! gp->dK) error("derivative info not in gp; use newGP with dK=TRUE");

  /* make matrix bones */
  Xcand = new_matrix_bones(Xcand_in, *ncand_in, *m_in);
  Xref = new_matrix_bones(Xref_in, *nref_in, *m_in);

  mspeGP(gp, *ncand_in, Xcand, *nref_in, Xref, *fi_in, *verb_in, mspe_out);

  /* clean up */
  free(Xcand);
  free(Xref);
}
