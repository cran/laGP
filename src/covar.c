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
#include <stdlib.h>
#ifdef RPRINT
#include <R.h>
#else 
#include <math.h>
#endif


/*
 * covar_symm:
 *
 * calculate the correlation (K) between X1 and X2 with 
 * an isotropic power exponential correlation function 
 * with range d and nugget g -- assumes symetric matrix
 */

void covar_symm(const int col, double **X, const int n, 
                double d, double g, double **K)
{
  int i, j, k;

  /* calculate the covariance */
  for(i=0; i<n; i++) {
    for(j=i+1; j<n; j++) {
      K[i][j] = 0.0;
      for(k=0; k<col; k++) K[i][j] += sq(X[i][k] - X[j][k]);
      K[j][i] = K[i][j] = exp(0.0 - K[i][j]/d);
    }
    K[i][i] = 1.0 + g;
  }
}


/*
 * covar:
 *
 * calculate the correlation (K) between X1 and X2 with 
 * an isotropic power exponential correlation function 
 * with range d and nugget g
 */

void covar(const int col, double **X1, const int n1, double **X2,
	   const int n2, double d, double g, double **K)
{
  int i, j, k;

  /* calculate the covariance */
  for(i=0; i<n1; i++)
    for(j=0; j<n2; j++) {
      K[i][j] = 0.0;
      for(k=0; k<col; k++) K[i][j] += sq(X1[i][k] - X2[j][k]);
      if(K[i][j] == 0.0) K[i][j] = 1.0 + g;
      else K[i][j] = exp(0.0 - K[i][j]/d);
    }
}


/*
 * diff_covar_symm:
 *
 * calculate the first and 2nd derivative (wrt d) of the correlation (K)
 * between X1 and X2 with an isotropic power exponential 
 * correlation function with range d and nugget g (though g not
 * needed) -- assymes symmetric matrix
 */

void diff_covar_symm(const int col, double **X, const int n, 
                     double d, double **dK, double **d2K)
{
  int i, j, k;
  double d2, dist;

  /* calculate the covariance */
  d2 = sq(d);
  for(i=0; i<n; i++) {
    for(j=i+1; j<n; j++) {
      dist = 0.0;
      for(k=0; k<col; k++) dist += sq(X[i][k] - X[j][k]);
      dK[j][i] = dK[i][j] = dist*exp(0.0 - dist/d)/d2;
      if(d2K) d2K[j][i] = d2K[i][j] = dK[i][j]*(dist - 2.0*d)/d2; 
    }
    d2K[i][i] = dK[i][i] = 0.0;
  }
}


/*
 * diff_covar:
 *
 * calculate the first and 2nd derivative (wrt d) of the correlation (K)
 * between X1 and X2 with an isotropic power exponential 
 * correlation function with range d and nugget g (though g not
 * needed)
 */

void diff_covar(const int col, double **X1, const int n1, double **X2,
		const int n2, double d, double **dK, double **d2K)
{
  int i, j, k;
  double d2, dist;

  /* calculate the covariance */
  d2 = sq(d);
  for(i=0; i<n1; i++)
    for(j=0; j<n2; j++) {
      dist = 0.0;
      for(k=0; k<col; k++) dist += sq(X1[i][k] - X2[j][k]);
      dK[i][j] = dist*exp(0.0 - dist/d)/d2;
      if(d2K) d2K[i][j] = dK[i][j]*(dist - 2.0*d)/d2; 
    }
}


/*
 * distance:
 * 
 * C-side version of distance_R
 */

void distance(double **X1, const unsigned int n1, double **X2,
	      const unsigned int n2, const unsigned int m,
	      double **D)
{
  unsigned int i,j,k;

  /* for each row of X1 and X2 */
  for(i=0; i<n1; i++) {
    for(j=0; j<n2; j++) {

      /* sum the squared entries */
      D[i][j] = 0.0;
      for(k=0; k<m; k++) {
	      D[i][j] += sq(X1[i][k] - X2[j][k]);
      }

    }
  }
}



/*
 * distance_R:
 *
 * function for calculating the distance matrix between
 * the rows of X1 and X2, with output in D_out -- using
 * a built-in R interface
 */
 
void distance_R(double *X1_in, int *n1_in, double *X2_in, 
		int *n2_in, int *m_in, double *D_out)
{
  double **X1, **X2, **D;
  
  /* make matrix bones */
  X1 = new_matrix_bones(X1_in, *n1_in, *m_in);
  X2 = new_matrix_bones(X2_in, *n2_in, *m_in);
  D = new_matrix_bones(D_out, *n1_in, *n2_in);

  distance(X1, *n1_in, X2, *n2_in, *m_in, D);

  /* clean up */
  free(X1);
  free(X2);
  free(D);
}


/*
 * distance_symm_R:
 *
 * function for calculating the distance matrix between
 * the rows of X1 and itself, with output in the symmetric
 * D_out matrix -- using a built-in R interface
 */

void distance_symm_R(double *X_in, int *n_in, int *m_in, double *D_out)
{
  int n, m, i, j, k;
  double **X, **D;
  
  /* copy integers */
  n = *n_in;
  m = *m_in;

  /* make matrix bones */
  X = new_matrix_bones(X_in, n, m);
  D = new_matrix_bones(D_out, n, n);

  /* for each row of X and itself */
  for(i=0; i<n; i++) {
    D[i][i] = 0.0;
    for(j=i+1; j<n; j++) {
      D[i][j] = 0.0;
      for(k=0; k<m; k++) 
	      D[i][j] += sq(X[i][k] - X[j][k]);
        D[j][i] = D[i][j];
    }
  }

  /* clean up */
  free(X);
  free(D);
}


/*
 * dist2covar_R:
 *
 * function for converting a distance matrix (D) into a 
 * covariance matrix (K) using an isotropic power expoential 
 * covariance function with range d and nugget g
 */

void dist2covar_R(double *D_in, int *n1_in, int *n2_in, double *d_in,
		  double *g_in, double *K_out)
{
  int n1, n2, i, j;
  double **D, **K;
  double d, g;

  /* copy integers */
  n1 = *n1_in;
  n2 = *n2_in;
  d = *d_in;
  g = *g_in;

  /* make matrix bones */
  D = new_matrix_bones(D_in, n1, n2);
  K = new_matrix_bones(K_out, n1, n2);

  /* calculate the covariance */
  for(i=0; i<n1; i++)
    for(j=0; j<n2; j++) {
      if(D[i][j] == 0) K[i][j] = 1.0 + g;
      else K[i][j] = exp(0.0 - D[i][j]/d);
    }

  /* clean up */
  free(D);
  free(K);
}



/*
 * dist2covar_symm_R:
 *
 * function for converting a symmetric distance matrix (D) 
 * into a covariance matrix (K) using an isotropic power 
 * expoential covariance function with range d and nugget d 
 */

void dist2covar_symm_R(double *D_in, int *n_in, double *d_in,
		       double *g_in, double *K_out)
{

  int n, i, j;
  double **D, **K;
  double d, g;

  /* copy integers */
  n = *n_in;
  d = *d_in;
  g = *g_in;

  /* make matrix bones */
  D = new_matrix_bones(D_in, n, n);
  K = new_matrix_bones(K_out, n, n);

  /* calculate the covariance */
  for(i=0; i<n; i++) {
    K[i][i] = 1.0 + g;
    for(j=i+1; j<n; j++) {
      K[i][j] = exp(0.0 - D[i][j]/d);
      K[j][i] = K[i][j];
    }
  }

  /* clean up */
  free(D);
  free(K);
}
