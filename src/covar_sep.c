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
#include "linalg.h"
#include <assert.h>
#include <stdlib.h>
#ifdef RPRINT
#include <R.h>
#else 
#include <math.h>
#endif

/*
 * covar_sep_symm:
 *
 * calculate the correlation (K) between X1 and X2 with 
 * a separable power exponential correlation function 
 * with range d and nugget g
 */

void covar_sep_symm(const int col, double **X, const int n, 
		    double *d, const double g, double **K)
{
  int i, j, k;

  /* calculate the covariance */
  for(i=0; i<n; i++) {
    K[i][i] = 1.0 + g;
    for(j=i+1; j<n; j++) {
      K[i][j] = 0.0;
      for(k=0; k<col; k++) K[i][j] += sq(X[i][k] - X[j][k])/d[k];
      K[i][j] = exp(0.0 - K[i][j]);
      K[j][i] = K[i][j];
    }
  }
}



/*
 * covar_sep:
 *
 * calculate the correlation (K) between X1 and X2 with 
 * an isotropic power exponential correlation function 
 * with range d and nugget g
 */

void covar_sep(const int col, double **X1, const int n1, double **X2,
     const int n2, double *d, double **K)
{
  int i, j, k;

  /* calculate the covariance */
  for(i=0; i<n1; i++)
    for(j=0; j<n2; j++) {
      K[i][j] = 0.0;
      for(k=0; k<col; k++) K[i][j] += sq(X1[i][k] - X2[j][k])/d[k];
      K[i][j] = exp(0.0 - K[i][j]);
    }
}


/*
 * diff_covar_sep:
 *
 * calculate the first and second derivative (wrt d) of the correlation (K)
 * between X1 and X2 with an separable power exponential 
 * correlation function with range d and nugget g (though g not
 * needed)
 */

void diff_covar_sep(const int col, double **X1, const int n1, 
        double **X2, const int n2, double *d, double **K, double ***dK)
{
  int i, j, k;
  double d2k;

  /* first copy K into each dK */
  for(k=0; k<col; k++) {
    d2k = sq(d[k]);
    for(i=0; i<n1; i++) {
      for(j=0; j<n2; j++) {
        dK[k][i][j] = K[i][j]*sq(X1[i][k] - X2[j][k])/d2k;
      }
    }
  }
}


/*
 * diff_covar_sep_symm:
 *
 * calculate the first and second derivative (wrt d) of the correlation (K)
 * between X1 and X2 with an separable power exponential 
 * correlation function with range d and nugget g (though g not
 * needed) -- assumes symmetric matrix
 */

void diff_covar_sep_symm(const int col, double **X, const int n, 
        double *d, double **K, double ***dK)
{
  int i, j, k;
  double d2k;

  /* first copy K into each dK */
  for(k=0; k<col; k++) {
    d2k = sq(d[k]);
    for(i=0; i<n; i++) {
      for(j=i+1; j<n; j++) {
        dK[k][j][i] = dK[k][i][j] = K[i][j]*sq(X[i][k] - X[j][k])/d2k;
      }
      dK[k][i][i] = 0.0;
    }
  }
}


/*
 * calc_g_mui_kxy_sep:
 *
 * function for calculating the g vector, mui scalar, and
 * kxy vector for the IECI calculation; kx is length-n 
 * utility space -- only implements isotropic case; separable
 * version implemented in plgp source tree;
 */

void calc_g_mui_kxy_sep(const int col, double *x, double **X, 
        const int n, double **Ki, double **Xref, 
        const int m, double *d, double g, double *gvec, 
        double *mui, double *kx, double *kxy)
{
  double mu_neg;
  int i;

  /* sanity check */
  if(m == 0) assert(!kxy && !Xref);

  /* kx <- drop(covar(X1=pall$X, X2=x, d=Zt$d, g=Zt$g)) */
  covar_sep(col, &x, 1, X, n, d, &kx);
  /* kxy <- drop(covar(X1=x, X2=Xref, d=Zt$d, g=0)) */
  if(m > 0) covar_sep(col, &x, 1, Xref, m, d, &kxy);

  /* Kikx <- drop(util$Ki %*% kx) stored in gvex */
  linalg_dsymv(n,1.0,Ki,n,kx,1,0.0,gvec,1);

  /* mui <- drop(1 + Zt$g - t(kx) %*% Kikx) */
  *mui = 1.0 + g - linalg_ddot(n, kx, 1, gvec, 1);
  
  /* gvec <- - Kikx/mui */
  mu_neg = 0.0 - 1.0/(*mui);
  for(i=0; i<n; i++) gvec[i] *= mu_neg;
}
