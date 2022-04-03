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
#include "rhelp.h"
#include <stdlib.h>
#ifdef RPRINT
#include <R.h>
#include <Rmath.h>
#endif

#if FALSE
/*
 * rbetter_R:
 *
 * R interface to rbetter function, used to sample uniformly
 * from the (diagonal) half of a rectable where the sum of
 * the rows of X (the sample) are less than ystar
 */

void rbetter_R(int *n_in, int *m_in, double *rect_in, double *ystar_in, 
	       double *X_out)
{
  unsigned int n, m, i, j;
  double **X, **rect;
  unsigned int *ro;
  double *Xi;
  double ystar, cumsum, left;

  /* copy inputs */
  n = (unsigned int) *n_in;
  m = (unsigned int) *m_in;
  ystar = *ystar_in;

  GetRNGstate();

  /* construct big X matrix */
  X = new_matrix_bones(X_out, n, m);
  rect = new_matrix_bones(rect_in, 2, m);
  Xi = new_vector(m);

  for(i=0; i<n; i++) {
    cumsum = 0.0;
    /* sample ith row */
    for(j=0; j<m; j++) {
      left = ystar - cumsum;
      if(rect[0][j] > left) X[i][j] = R_NegInf;
      X[i][j] = runif(rect[0][j], MYfmin(rect[1][j], left));
      cumsum += X[i][j];
    }
    /* randomly reorder ith row */
    ro = rand_indices(m);
    dupv(Xi, X[i], m);
    for(j=0; j<m; j++) X[i][j] = Xi[ro[j]];
    free(ro);
  }

  PutRNGstate();

  /* clean up */
  free(X);
  free(Xi);
  free(rect);
}
#endif


/*
 * rbetter_R:
 *
 * R interface to rbetter function, used to sample uniformly
 * from the (diagonal) half of a rectable where the sum of
 * the rows of X (the sample) are less than ystar; rejection
 * sampling method
 */

void rbetter_R(int *n_in, int *m_in, double *rect_in, double *ystar_in, 
	       double *X_out)
{
  unsigned int n, m, i, j;
  double **X, **rect;
  double ystar, cumsum;

  /* copy inputs */
  n = (unsigned int) *n_in;
  m = (unsigned int) *m_in;
  ystar = *ystar_in;

  GetRNGstate();

  /* construct big X matrix */
  X = new_matrix_bones(X_out, n, m);
  rect = new_matrix_bones(rect_in, 2, m);
  for(j=0; j<m; j++) if(rect[1][j] > ystar) rect[1][j] = ystar;

  for(i=0; i<n; i++) {
    /* sample ith row */
    do {
      cumsum = 0.0;
      for(j=0; j<m; j++) {
	X[i][j] = runif(rect[0][j], rect[1][j]);
	cumsum += X[i][j];
	if(cumsum > ystar) break;
      }
    } while(cumsum > ystar);
  }

  PutRNGstate();

  /* clean up */
  free(X);
  free(rect);
}
