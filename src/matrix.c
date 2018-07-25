/******************************************************************************** 
 *
 * Bayesian Regression and Adaptive Sampling with Gaussian Process Trees
 * Copyright (C) 2005, University of California
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
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
 *
 * Questions? Contact Robert B. Gramacy (rbgramacy@ams.ucsc.edu)
 *
 ********************************************************************************/


#include "rhelp.h"

#include "matrix.h"
#include <assert.h>
#include <math.h>
#include <stdlib.h>


#define DEBUG

/*
 * get_data_rect:
 *
 * compute and return the rectangle implied by the X data
 */

double **get_data_rect(double **X, unsigned int N, unsigned int d)
{
  unsigned int i,j;
  double ** rect = new_matrix(2, d);
  
  for(i=0; i<d; i++) {
    rect[0][i] = X[0][i];
    rect[1][i] = X[0][i];
    for(j=1; j<N; j++) {
      if(X[j][i] < rect[0][i]) rect[0][i] = X[j][i];
      else if(X[j][i] > rect[1][i]) rect[1][i] = X[j][i];
    }
  }
  return(rect);
}


/*
 * replace matrix with zeros
 */

void zero(double **M, unsigned int n1, unsigned int n2)
{
  unsigned int i, j;
  for(i=0; i<n1; i++) for(j=0; j<n2; j++) M[i][j] = 0;
}


/*
 * check if a (square) matrix is zeros
 */

int isZero(double **M, unsigned int m, int sym)
{
  unsigned int i,j, upto;
  for(j=0; j<m; j++) {
    upto = m;
    if(sym) upto = j+1;
    for(i=0; i<upto; i++)
      if(M[j][i] != 0.0) return 0;
  }
  return 1;
}


/*
 * replace square matrix with identity 
 */

void id(double **M, unsigned int n)
{
  unsigned int i;
  zero(M, n, n);
  for(i=0; i<n; i++) M[i][i] = 1.0;
}


/*
 * same as new_matrix below, but for creating
 * n x n identity matrices
 */

double ** new_id_matrix(unsigned int n)
{
  unsigned int i;
  double** m = new_zero_matrix(n, n);
  for(i=0; i<n; i++) m[i][i] = 1.0;
  return m;
}


/*
 * same as new_matrix below, but zeros out the matrix
 */

double ** new_zero_matrix(unsigned int n1, unsigned int n2)
{
  unsigned int i, j;
  double **m = new_matrix(n1, n2);
  for(i=0; i<n1; i++) for(j=0; j<n2; j++) m[i][j] = 0.0;
  return m;
}


/*
 * same as new_imatrix, but zeros out the matrix
 */

int ** new_zero_imatrix(unsigned int n1, unsigned int n2)
{
  unsigned int i, j;
  int **m = new_imatrix(n1, n2);
  for(i=0; i<n1; i++) for(j=0; j<n2; j++) m[i][j] = 0;
  return m;
}


/*
 * create a new n1 x n2 matrix which is allocated like
 * an n1*n2 array, but can be referenced as a 2-d array
 */

double ** new_matrix(unsigned int n1, unsigned int n2)
{
  int i;
  double **m;
  
  if(n1 == 0 || n2 == 0) return NULL;
  
  m = (double**) malloc(sizeof(double*) * n1);
  assert(m);
  m[0] = (double*) malloc(sizeof(double) * (n1*n2));
  assert(m[0]);
  
  for(i=1; i<n1; i++) m[i] = m[i-1] + n2;
  
  return m;
}


/*
 * create a double ** Matrix from a double * vector
 * should be freed with the free command, rather than
 * delete_matrix
 */

double ** new_matrix_bones(double *v, unsigned int n1, unsigned int n2)
{
  double **M;
  int i;
  M = (double **) malloc(sizeof(double*) * n1);
  M[0] = v;
  for(i=1; i<n1; i++) M[i] = M[i-1] + n2;
  return(M);
}


/*
 * create an integer ** Matrix from a double * vector
 * should be freed with the free command, rather than
 * delete_matrix
 */

int ** new_imatrix_bones(int *v, unsigned int n1, unsigned int n2)
{
  int **M;
  int i;
  M = (int **) malloc(sizeof(double*) * n1);
  M[0] = v;
  for(i=1; i<n1; i++) M[i] = M[i-1] + n2;
  return(M);
}


/*
 * create a new n1 x n2 integer matrix which is allocated like
 * an n1*n2 array, but can be referenced as a 2-d array
 */

int ** new_imatrix(unsigned int n1, unsigned int n2)
{
  int i;
  int **m;
  
  if(n1 == 0 || n2 == 0) return NULL;
  
  m = (int**) malloc(sizeof(double*) * n1);
  assert(m);
  m[0] = (int*) malloc(sizeof(double) * (n1*n2));
  assert(m[0]);
  
  for(i=1; i<n1; i++) m[i] = m[i-1] + n2;
  
  return m;
}


/*
 * create a new n2 x n1 integer matrix which is allocated like
 * an n1*n2 array, and copy the TRANSPOSE of n1 x n2 M into it.
 */

int ** new_t_imatrix(int** M, unsigned int n1, unsigned int n2)
{
  int i,j;
  int **m;
  
  if(n1 <= 0 || n2 <= 0) {
    assert(M == NULL);
    return NULL;
  }
  
  m = new_imatrix(n2, n1);
  for(i=0; i<n1; i++) for(j=0; j<n2; j++)  m[j][i] = M[i][j];
  return m;
}


/*
 * create a new n1 x n2 matrix which is allocated like
 * an n1*n2 array, and copy the of n1 x n2 M into it.
 */

double ** new_dup_matrix(double** M, unsigned int n1, unsigned int n2)
{
  double **m;

  if(n1 <= 0 || n2 <= 0) {
    /* assert(M == NULL); */
    return NULL;
  }
 
  m = new_matrix(n1, n2);
  dup_matrix(m, M, n1, n2);
  return m;
}


/*
 * create a new n1 x n2 matrix which is allocated like
 * an n1*n2 array, and copy the of n1 x n2 M into it.
 */

int ** new_dup_imatrix(int** M, unsigned int n1, unsigned int n2)
{
  int **m;

  if(n1 <= 0 || n2 <= 0) {
    /* assert(M == NULL); */
    return NULL;
  }
 
  m = new_imatrix(n1, n2);
  dup_imatrix(m, M, n1, n2);
  return m;
}


/*
 * create a new n1 x (n2-1) matrix which is allocated like
 * an n1*(n2-1) array, and copy M[n1][2:n2] into it.
 */

double ** new_shift_matrix(double** M, unsigned int n1, unsigned int n2)
{
  double **m;
  unsigned int i, j;
  if(n1 <= 0 || n2 <= 1) {
    assert(M == NULL);
    return NULL;
  }
  m = new_matrix(n1, (n2-1));
  /* printMatrix(M, n1, n2, MYstdout); */
  for(i=0; i<n1; i++) for(j=0; j<(n2-1); j++) m[i][j] = M[i][j+1];
  /* printMatrix(m, n1, (n2-1), MYstdout); */
  return m;
}


/*
 * copy M2 to M1
 */

void dup_matrix(double** M1, double **M2, unsigned int n1, unsigned int n2)
{
  unsigned int i;
  if(n1 == 0 || n2 == 0) return;
  assert(M1 && M2);
  for(i=0; i<n1; i++) dupv(M1[i], M2[i], n2);
}


/*
 * copy M2 to M1 for integer matrices
 */

void dup_imatrix(int** M1, int **M2, unsigned int n1, unsigned int n2)
{
  unsigned int i;
  if(n1 == 0 || n2 == 0) return;
  assert(M1 && M2);
  for(i=0; i<n1; i++) dupiv(M1[i], M2[i], n2);
}


/*
 * swap the pointers of M2 to M1, and vice-versa
 * (tries to aviod dup_matrix when unnecessary)
 */

void swap_matrix(double **M1, double **M2, unsigned int n1, unsigned int n2)
{
  unsigned int i;
  double *temp;
  temp = M1[0];
  M1[0] = M2[0];
  M2[0] = temp;
  for(i=1; i<n1; i++) {
    M1[i] = M1[i-1] + n2;
    M2[i] = M2[i-1] + n2;
  }
}


/*
 * create a bigger n1 x n2 matrix which is allocated like
 * an n1*n2 array, and copy the of n1 x n2 M into it.
 * deletes the old matrix
 */

double ** new_bigger_matrix(double** M, unsigned int n1, unsigned int n2, 
		unsigned int n1_new, unsigned int n2_new)
{
  int i;
  double **m;
  
  assert(n1_new >= n1);
  assert(n2_new >= n2);
  
  if(n1_new <= 0 || n2_new <= 0) {
    assert(M == NULL);
    return NULL;
  }
  
  if(M == NULL) {
    assert(n1 == 0 || n2 == 0);
    return new_zero_matrix(n1_new, n2_new);
  }
  
  if(n2 == n2_new) {
    m = (double**) malloc(sizeof(double*) * n1_new);
    assert(m);
    m[0] = realloc(M[0], sizeof(double) * n1_new * n2_new);
    free(M);
    assert(m[0]);
    for(i=1; i<n1_new; i++) m[i] = m[i-1] + n2_new;
    zerov(m[n1], (n1_new-n1)*n2_new);
  } else {
    m = new_zero_matrix(n1_new, n2_new);
    dup_matrix(m, M, n1, n2);
    delete_matrix(M);
  }
  return m;
}


/*
 * create a bigger n1 x n2 matrix which is allocated like
 * an n1*n2 array, and copy the of n1 x n2 M into it.
 * deletes the old matrix -- integer version
 */

int ** new_bigger_imatrix(int** M, unsigned int n1, unsigned int n2, 
			  unsigned int n1_new, unsigned int n2_new)
{
  int i;
  int **m;
  
  assert(n1_new >= n1);
  assert(n2_new >= n2);
  
  if(n1_new <= 0 || n2_new <= 0) {
    assert(M == NULL);
    return NULL;
  }
  
  if(M == NULL) {
    assert(n1 == 0 || n2 == 0);
    return new_zero_imatrix(n1_new, n2_new);
  }
  
  if(n2 == n2_new) {
    m = (int**) malloc(sizeof(int*) * n1_new);
    assert(m);
    m[0] = realloc(M[0], sizeof(int) * n1_new * n2_new);
    free(M);
    assert(m[0]);
    for(i=1; i<n1_new; i++) m[i] = m[i-1] + n2_new;
    zeroiv(m[n1], (n1_new-n1)*n2_new);
  } else {
    m = new_zero_imatrix(n1_new, n2_new);
    dup_imatrix(m, M, n1, n2);
    delete_imatrix(M);
  }
  return m;
}


/*
 * create a new n1 x n2 matrix which is allocated like
 * an n1*n2 array, and copy the of n1 x n2 M into it.
 */

double ** new_normd_matrix(double** M, unsigned int n1, unsigned int n2, 
		double **rect, double normscale)
{
  double **m;
  m = new_dup_matrix(M, n1, n2);
  normalize(m, rect, n1, n2, normscale);
  return m;
}


/*
 * create a new n2 x n1 matrix which is allocated like
 * an n1*n2 array, and copy the TRANSPOSE of n1 x n2 M into it.
 */

double ** new_t_matrix(double** M, unsigned int n1, unsigned int n2)
{
  int i,j;
  double **m;
  
  if(n1 <= 0 || n2 <= 0) {
    assert(M == NULL);
    return NULL;
  }
  
  m = new_matrix(n2, n1);
  for(i=0; i<n1; i++) for(j=0; j<n2; j++)  m[j][i] = M[i][j];
  return m;
}


/*
 * delete a matrix allocated as above
 */

void delete_matrix(double** m)
{
  if(m == NULL) return;
  assert(*m);
  free(*m);
  assert(m);
  free(m);
}


/*
 * delete an integer matrix allocated as above
 */

void delete_imatrix(int** m)
{
  if(m == NULL) return;
  assert(*m);
  free(*m);
  assert(m);
  free(m);
}


/*
 * print an n x col matrix allocated as above out an opened outfile.
 * actually, this routine can print any double** 
 */

void printMatrix(double **M, unsigned int n, unsigned int col, FILE *outfile)
{
  int i,j;
  assert(outfile);
  if(n > 0 && col > 0) assert(M);
  for(i=0; i<n; i++) {
    for(j=0; j<col; j++) {
#ifdef DEBUG
      if(j==col-1) MYprintf(outfile, "%.20f\n", M[i][j]);
      else MYprintf(outfile, "%.20f ", M[i][j]);
#else
      if(j==col-1) MYprintf(outfile, "%g\n", M[i][j]);
      else MYprintf(outfile, "%g ", M[i][j]);
#endif
    }
  }
}


/*
 * print an n x col integer matrix allocated as above out an opened outfile.
 * actually, this routine can print any double** 
 */

void printIMatrix(int **M, unsigned int n, unsigned int col, FILE *outfile)
{
  int i,j;
  assert(outfile);
  if(n > 0 && col > 0) assert(M);
  for(i=0; i<n; i++) {
    for(j=0; j<col; j++) {
#ifdef DEBUG
      if(j==col-1) MYprintf(outfile, "%d\n", M[i][j]);
      else MYprintf(outfile, "%d ", M[i][j]);
#else
      if(j==col-1) MYprintf(outfile, "%d\n", M[i][j]);
      else MYprintf(outfile, "%d ", M[i][j]);
#endif
    }
  }
}


/*
 * print the transpose of an n x col matrix allocated as above out an
 * opened outfile.  actually, this routine can print any double**
 */

void printMatrixT(double **M, unsigned int n, unsigned int col, FILE *outfile)
{
  int i,j;
  assert(outfile);
  if(n > 0 && col > 0) assert(M);
  for(i=0; i<col; i++) {
    for(j=0; j<n; j++) {
      if(j==n-1) MYprintf(outfile, "%g\n", M[j][i]);
      else MYprintf(outfile, "%g ", M[j][i]);
    }
  }
}


/*
 * add two matrices of the same size 
 * M1 = a*M1 + b*M2
 */

void add_matrix(double a, double **M1, double b, double **M2, 
		unsigned int n1, unsigned int n2)
{
  unsigned int i,j;
  assert(n1 > 0 && n2 > 0);
  assert(M1 && M2);
  for(i=0; i<n1; i++)
    for(j=0; j<n2; j++)
      M1[i][j] = a*M1[i][j] + b*M2[i][j];
}


/*
 * add_p_matrix:
 *
 * add v[n1][n2] to V into the positions specified by p1[n1] and
 * p2[n2]
 */

void add_p_matrix(double a, double **V, int *p1, int *p2, double b, double **v, 
		unsigned int n1, unsigned int n2)
{
  int i,j;
  assert(V); assert(p1); assert(p2); assert(n1 > 0 && n2 > 0);
  for(i=0; i<n1; i++) for(j=0; j<n2; j++) 
    V[p1[i]][p2[j]] = a*V[p1[i]][p2[j]] + b*v[i][j];
}


/*
 * subtract off (element-wise) the center[n2] vector from each
 * column of the M[n1][n2] matrix
 */

void center_columns(double **M, double *center, unsigned int n1, unsigned int n2)
{
  unsigned int i,j;

  /* sanity checks */
  if(n1 <= 0 || n2 <= 0) {return;}
  assert(center && M);

  for(i=0; i<n2; i++)
     for(j=0; j<n1; j++) 
       M[j][i] -= center[i];
}


/*
 * subtract off (element-wise) the center[n1] vector from each
 * row of the M[n1][n2] matrix
 */

void center_rows(double **M, double *center, unsigned int n1, unsigned int n2)
{
  unsigned int i;

  /* sanity checks */
  if(n1 <= 0 || n2 <= 0) {return;}
  assert(center && M);

  for(i=0; i<n1; i++) centerv(M[i], n2, center[i]); 
}


/*
 * subtract off (element-wise) the center[n2] vector from each
 * column of the M[n1][n2] matrix
 */

void norm_columns(double **M, double *norm, unsigned int n1, unsigned int n2)
{
  unsigned int i,j;

  /* sanity checks */
  if(n1 <= 0 || n2 <= 0) {return;}
  assert(norm && M);

  for(i=0; i<n2; i++)
     for(j=0; j<n1; j++) 
       M[j][i] /= norm[i];
}


/*
 * wmean_of_columns:
 *
 * fill mean[n2] with the weighted mean of the columns of M (n1 x n2);
 * weight vector should have length n1;
 */

void wmean_of_columns(double *mean, double **M, unsigned int n1, unsigned int n2,
		      double *weight)
{
  unsigned int i,j;
  double sw;
 
  /* sanity checks */
  if(n1 <= 0 || n2 <= 0) {return;}
  assert(mean && M);
  
  /* find normailzation constant */
  if(weight) sw = sumv(weight, n1); 
  else sw = (double) n1;

  /* calculate mean of columns */
  for(i=0; i<n2; i++) {
    mean[i] = 0;
    if(weight) for(j=0; j<n1; j++) mean[i] += weight[j] * M[j][i];	
    else for(j=0; j<n1; j++) mean[i] += M[j][i];	
    mean[i] = mean[i] / sw;
  }
}

/*
 * wvar_of_columns:
 *
 * fill var[n2] with the weighted variance of the columns of M (n1 x n2);
 * weight vector should have length n1;
 */

void wvar_of_columns(double *var, double **M, unsigned int n1, unsigned int n2,
		      double *weight)
{
  unsigned int i,j;
  double sw;
  double *mean = new_vector(n2);
  /* sanity checks */
  if(n1 <= 0 || n2 <= 0) {return;}
  assert(mean && M && var);
  
  /* find normailzation constant */
  if(weight) sw = sumv(weight, n1); 
  else sw = (double) n1;

  /* calculate mean of columns */
  for(i=0; i<n2; i++) {
    mean[i] = 0;
    if(weight) for(j=0; j<n1; j++) mean[i] += weight[j] * M[j][i];	
    else for(j=0; j<n1; j++) mean[i] += M[j][i];	
    mean[i] = mean[i]/sw;
  }

  /* calculate variance of columns */
  for(i=0; i<n2; i++) {
    var[i] = 0;
    if(weight) for(j=0; j<n1; j++) var[i] += weight[j] * (M[j][i] - mean[i]) * (M[j][i] - mean[i]);	
    else for(j=0; j<n1; j++) var[i] +=  (M[j][i] - mean[i]) * (M[j][i] - mean[i]);	
    var[i] = var[i]/sw;
  }

  free(mean);
}


/*
 * sum_of_columns_f:
 *
 * fill sum[n1] with the sum of the columns of M (n1 x n2);
 * each element of which is sent through function f() first;
 */

void sum_of_columns_f(double *s, double **M, unsigned int n1, unsigned int n2,
		      double(*f)(double))
{
  unsigned int i,j;

  /* sanity checks */
  if(n1 <= 0 || n2 <= 0) {return;}
  assert(s && M);
  
  /* calculate sum of columns */
  for(i=0; i<n2; i++) {
    s[i] = f(M[0][i]);
    for(j=1; j<n1; j++) s[i] += f(M[j][i]);
  }
}


/*
 * sum_of_columns:
 *
 * fill s[n1] with the sum of the columns of M (n1 x n2);
 */

void sum_of_columns(double *s, double **M, unsigned int n1, unsigned int n2)
{
  unsigned int i,j;

  /* sanity checks */
  if(n1 <= 0 || n2 <= 0) {return;}
  assert(s && M);
  
  /* calculate sum of columns */
  for(i=0; i<n2; i++) {
    s[i] = M[0][i];
    for(j=1; j<n1; j++) s[i] += M[j][i];
  }
}



/*
 * min_of_columns:
 *
 * fill s[n1] with the min of the columns of M (n1 x n2);
 */

void min_of_columns(double *s, double **M, unsigned int n1, unsigned int n2)
{
  unsigned int i,j;

  /* sanity checks */
  if(n1 <= 0 || n2 <= 0) {return;}
  assert(s && M);
  
  /* calculate sum of columns */
  for(i=0; i<n2; i++) {
    s[i] = M[0][i];
    for(j=1; j<n1; j++) if(M[j][i] < s[i]) s[i] = M[j][i];
  }
}


/*
 * sum_of_each_column_f:
 *
 * fill sum[n1] with the sum of the columns of M (n1[i] x n2);
 * each element of which is sent through function f() first;
 * n1 must have n2 entries
 */

void sum_of_each_column_f(double *s, double **M, unsigned int *n1, 
			  unsigned int n2, double(*f)(double))
{
  unsigned int i,j;

  /* sanity checks */
  if(n2 <= 0) {return;}
  assert(s && M);
  
  /* calculate sum of columns */
  for(i=0; i<n2; i++) {
    if(n1[i] > 0) s[i] = f(M[0][i]);
    else s[i] = 0;
    for(j=1; j<n1[i]; j++) s[i] += f(M[j][i]);
  }
}


/*
 * wmean_of_columns_f:
 *
 * fill mean[n1] with the weighted mean of the columns of M (n1 x n2);
 * weight vector should have length n1 -- each element of which is
 * sent through function f() first;
 */

void wmean_of_columns_f(double *mean, double **M, unsigned int n1, unsigned int n2,
		      double *weight, double(*f)(double))
{
  unsigned int i,j;
  double sw;

  /* sanity checks */
  if(n1 <= 0 || n2 <= 0) {return;}
  assert(mean && M);
  
  /* find normailzation constant */
  if(weight) sw = sumv(weight, n1);
  else sw = (double) n1;

  /* calculate mean of columns */
  for(i=0; i<n2; i++) {
    mean[i] = 0;
    if(weight) for(j=0; j<n1; j++) mean[i] += weight[j] * f(M[j][i]);
    else for(j=0; j<n1; j++) mean[i] += f(M[j][i]);
    mean[i] = mean[i] / sw;
  }
}


/*
 * wmean_of_rows_f:
 *
 * fill mean[n1] with the weighted mean of the rows of M (n1 x n2);
 * weight vector should have length n2 -- each element of which is
 * sent through function f() first; 
 */

void wmean_of_rows_f(double *mean, double **M, unsigned int n1, unsigned int n2,
		  double *weight, double(*f)(double))
{
  unsigned int i,j;
  double sw;

  /* sanity checks */
  if(n1 <= 0 || n2 <= 0) {return;}
  assert(mean && M);

  /* calculate the normalization constant */
  if(weight) sw = sumv(weight, n2);
  else sw = (double) n2;

  /* calculate the mean of rows */
  for(i=0; i<n1; i++) {
    mean[i] = 0;
    if(weight) for(j=0; j<n2; j++) mean[i] += weight[j] * f(M[i][j]);	
    else for(j=0; j<n2; j++) mean[i] += f(M[i][j]);	
    mean[i] = mean[i] / sw;
  }
}


/*
 * wmean_of_rows_f:
 *
 * fill mean[n1] with the weighted mean of the rows of M (n1 x n2);
 * weight vector should have length n2;
 */

void wmean_of_rows(double *mean, double **M, unsigned int n1, unsigned int n2,
		  double *weight)
{
  unsigned int i,j;
  double sw;

  /* sanity checks */
  if(n1 <= 0 || n2 <= 0) {return;}
  assert(mean && M);

  /* calculate the normalization constant */
  if(weight) sw =  sumv(weight, n2);
  else sw = (double) n2;

  /* calculate the mean of rows */
  for(i=0; i<n1; i++) {
    mean[i] = 0;
    if(weight) for(j=0; j<n2; j++) mean[i] += weight[j] * M[i][j];	
    else for(j=0; j<n2; j++) mean[i] += M[i][j];	
    mean[i] = mean[i] / sw;
  }
}


/*
 * wcov_of_columns:
 *
 * fill cov[n1,n1] with the weighted covariance of the columns of M (n1 x n2);
 * weight vector should have length n1;
 */

void wcov_of_columns(double **cov, double **M, double *mean, unsigned int n1, 
		      unsigned int n2, double *weight)
{
  unsigned int i,j,t;
  double sw;
 
  /* sanity checks */
  if(n1 <= 0 || n2 <= 0) {return;}
  assert(cov && M && mean);
  
  /* find normailzation constant */
  if(weight) sw = sumv(weight, n1); 
  else sw = (double) n1;

  /* calculate mean of columns */
  for(i=0; i<n2; i++) {
    zerov(cov[i], n2);
    if(weight) {
      for(t=0; t<n1; t++) {
	for(j=i; j<n2; j++) /* using weights */
	  cov[i][j] += weight[t]*(M[t][i]*M[t][j] - M[t][i]*mean[j] - 
				  M[t][j]*mean[i]) + mean[i]*mean[j];
      }
    } else {
      for(t=0; t<n1; t++) {
	for(j=i; j<n2; j++) /*  not using weights */
	  cov[i][j] += (M[t][i]*M[t][j] - M[t][i]*mean[j] - 
			M[t][j]*mean[i] + mean[i]*mean[j]);
      }
    }
    scalev(cov[i], n2, 1.0/sw);

    /* fill in the other half */
    for(j=0; j<i; j++) cov[i][j] = cov[j][i]; 
  }
}



/*
 * wcovx_of_columns:
 *
 * fill mean[n1] with the weighted covariance of the columns of M1 (T x n1)
 * to those of M2 (T x n2); weight vector should have length T;
 */

void wcovx_of_columns(double **cov, double **M1, double **M2, double *mean1, 
		      double *mean2, unsigned int T,  unsigned int n1, 
		      unsigned int n2, double *weight)
{
  unsigned int i,j,t;
  double sw;
 
  /* sanity checks */
  if(T <= 0 || n1 <= 0 || n2 <= 0) {return;}
  assert(cov && M1 && M2 && mean1 && mean2);
  
  /* find normailzation constant */
  if(weight) sw = sumv(weight, T); 
  else sw = (double) T;

  /* calculate mean of columns */
  for(i=0; i<n1; i++) {
    zerov(cov[i], n2);
    if(weight) {
      for(t=0; t<T; t++) {
	for(j=0; j<n2; j++) /* using weights */
	  cov[i][j] += weight[t] * (M1[t][i]*M2[t][j] - M1[t][i]*mean2[j] - 
				    M2[t][j]*mean1[i]) + mean1[i]*mean2[j];
      }
    } else {
      for(t=0; t<T; t++) {
	for(j=0; j<n2; j++) /*  not using weights */
	  cov[i][j] += (M1[t][i]*M2[t][j] - M1[t][i]*mean2[j] - 
			M2[t][j]*mean1[i] + mean1[i]*mean2[j]);
      }
    }
    scalev(cov[i], n2, 1.0/sw);
  }
}


/*
 * fill the q^th quantile for each column of M (n1 x n2)
 * if non-null, the w argument should contain NORMALIZED
 * weights to be used in a bootstrap calculation of the 
 * quantiles
 */

void quantiles_of_columns(double **Q, double *q, unsigned int m,
			  double **M, unsigned int n1, unsigned int n2, 
			  double *w)
{
  unsigned int i,j;
  double *Mc, *wnorm, *qs;
  double W;

  /* check if there is anything to do */
  if(n1 == 0) return;

  /* allocate vector representing the current column,
   * and for storing the m quantiles */
  Mc = new_vector(n1);
  qs = new_vector(m);

  /* if non-null, create a normalized weight vector */
  if(w != NULL) {
    W = sumv(w, n1);
    wnorm = new_dup_vector(w, n1);
    scalev(wnorm, n1, 1.0/W);
  } else {
    wnorm = NULL;
  }

  /* for each column */
  for(i=0; i<n2; i++) {

    /* copy the column into a vector */
    for(j=0; j<n1; j++) Mc[j] = M[j][i];

    quantiles(qs, q, m, Mc, wnorm, n1);
    for(j=0; j<m; j++) Q[j][i] = qs[j];
  }

  /* clean up */
  if(w != NULL) { assert(wnorm); free(wnorm); }
  free(Mc);
  free(qs);
}


/* 
 * data structure for sorting weighted samples
 * to estimate quantiles 
 */

typedef struct wsamp
{
  double w;
  double x;
} Wsamp;


/*
 * comparison function for weighted samples
 */

int compareWsamp(const void* a, const void* b)
{
  Wsamp* aa = (Wsamp*)(*(Wsamp **)a); 
  Wsamp* bb = (Wsamp*)(*(Wsamp **)b); 
  if(aa->x < bb->x) return -1;
  else return 1;
}


/*
 * calculate the quantiles of v[1:n] specified in q[1:m], and store them
 * in qs[1:m]; If non-null weights, then use the sorting method; assume
 * that the weights are NORMALIZED, it is also assumed that the q[1:m] 
 * is specified in increasing order
 */

void quantiles(double *qs, double *q, unsigned int m, double *v,
	       double *w, unsigned int n)
{
  unsigned int i, k, j;
  double wsum;
  Wsamp **wsamp;

  /* create and fill pointers to weighted sample structures */
  if(w != NULL) {
    wsamp = (Wsamp**) malloc(sizeof(struct wsamp*) * n);
    for(i=0; i<n; i++) {
      wsamp[i] = malloc(sizeof(struct wsamp));
      wsamp[i]->w = w[i];
      wsamp[i]->x = v[i];
    }

    /* sort by v; and implicity report the associated weights w */
    qsort((void*)wsamp, n, sizeof(Wsamp*), compareWsamp);
  } else wsamp = NULL;

  /* for each quantile in q */
  wsum = 0.0;
  for(i=0, j=0; j<m; j++) {
    
    /* check to make sure the j-th quantile requested is valid */
    assert(q[j] > 0 && q[j] <1);
    
    /* find the (non-weighted) quantile using select */
    if(w == NULL) {
      
      /* calculate the index-position of the quantile */
      k = (unsigned int) n*q[j];
      qs[j] = quick_select_index(v, NULL, n, k);

    } else { /* else using sorting method */

      /* check to make sure the qs are ordered */
      assert(wsamp);
      if(j > 0) assert(q[j] > q[j-1]);

      /* find the next quantile in the q-array */
      for(; i<n; i++) {
	
	/* to cover the special case where n<=m */
	if(i > 0 && wsum >= q[j]) { qs[j] = wsamp[i-1]->x; break; }

	/* increment with the next weight */
	wsum += wsamp[i]->w;

	/* see if we've found the next quantile */
	if(wsum >= q[j]) { qs[j] = wsamp[i]->x; break; }
      }

      /* check to make sure we actually had founda  quantile */
      if(i == n) warning("unable to find quanile q[%d]=%g", j, q[j]);
    }
  }

  /* clean up */
  if(w) {
    assert(wsamp);
    for(i=0; i<n; i++) free(wsamp[i]);
    free(wsamp);
  }
}


/*
 * allocate and return an array of length n with scale*1 at
 * each entry
 */

double* ones(unsigned int n, double scale)
{
  double *o;
  unsigned int i;
  /* o = (double*) malloc(sizeof(double) * n); */
  o = new_vector(n);
  /* assert(o); */
  for(i=0; i<n; i++) o[i] = scale;
  return o;
}


/*
 * allocate and return an array containing
 * the seqence of doubles [from...to] with steps of
 * size by
 */

double* dseq(double from, double to, double by)
{
  unsigned int n,i;
  double *s = NULL;
  
  by = fabs(by);
  
  if(from <= to) n = (unsigned int) (to - from)/fabs(by) + 1;
  else n = (unsigned int) (from - to)/fabs(by) + 1;
  
  if( n == 0 ) return NULL;
  
  s = (double*) malloc(sizeof(double) * n);
  assert(s);
  s[0] = from;
  for(i=1; i<n; i++) {
    s[i] = s[i-1] + by;
  }
  return s;
}


/*
 * allocate and return an array containing
 * the integer seqence [from...to]
 */

int* iseq(double from, double to)
{
  unsigned int n,i;
  int by;
  int *s = NULL;
  
  if(from <= to) {
    n = (unsigned int) (to - from) + 1;
    by = 1;
  } else {
    assert(from > to);
    n = (unsigned int) (from - to) + 1;
    by = -1;
  }
  
  if(n == 0) return NULL;
  
  s = new_ivector(n);
  s[0] = from;
  for(i=1; i<n; i++) {
    s[i] = s[i-1] + by;
  }
  return s;
}


/*
 * find:
 *
 * return an integer of length (*len) with indexes into V which
 * satisfy the relation "V op val" where op is one of 
 * LT(<) GT(>) EQ(==) LEQ(<=) GEQ(>=) NE(!=)
 */

int* find(double *V, unsigned int n, FIND_OP op, double val, unsigned int* len)
{
  unsigned int i,j;
  int *tf;
  int *found;
  
  tf = new_ivector(n);
  
  (*len) = 0;
  switch (op) {
  case GT:  
    for(i=0; i<n; i++) {
      if(V[i] >  val) tf[i] = 1; 
      else tf[i] = 0; 
      if(tf[i] == 1) (*len)++;
    }
    break;
  case GEQ: 
    for(i=0; i<n; i++) {
      if(V[i] >= val) tf[i] = 1; 
      else tf[i] = 0; 
      if(tf[i] == 1) (*len)++;
    }
    break;
  case EQ:  
    for(i=0; i<n; i++) {
      if(V[i] == val) tf[i] = 1; 
      else tf[i] = 0; 
      if(tf[i] == 1) (*len)++;
    }
    break;
  case LEQ: 
    for(i=0; i<n; i++) {
      if(V[i] <= val) tf[i] = 1; 
      else tf[i] = 0; 
      if(tf[i] == 1) (*len)++;
    }
    break;
  case LT:  
    for(i=0; i<n; i++) {
      if(V[i] <  val) tf[i] = 1; 
      else tf[i] = 0; 
      if(tf[i] == 1) (*len)++;
    }
    break;
  case NE:  
    for(i=0; i<n; i++) {
      if(V[i] != val) tf[i] = 1; 
      else tf[i] = 0; 
      if(tf[i] == 1) (*len)++;
    }
    break;
  default: error("OP not supported");
  }
  
  if(*len == 0) found = NULL;
  else {
    found = new_ivector(*len);
    for(i=0,j=0; i<n; i++) {
      if(tf[i]) {
	found[j] = i;
	j++;
      }
    }
  }
  
  free(tf);
  return found;
}


/*
 * return an integer of length (*len) wih indexes into V[][col] which
 * satisfy the relation "V op val" where op is one of LT(<) GT(>)
 * EQ(==) LEQ(<=) GEQ(>=) NE(!=)
 */

int* find_col(double **V, int *pv, unsigned int n, 
	      unsigned int var, FIND_OP op, double val, 
	      unsigned int* len)
{
  unsigned int i,j;
  int *tf, *p;
  int *found;
  
  tf = new_ivector(n);
  if(pv) p = pv;
  else p = iseq(0,n-1);
  
  (*len) = 0;
  switch (op) {
  case GT:  
    for(i=0; i<n; i++) {
      if(V[p[i]][var] >  val) tf[i] = 1; 
      else tf[i] = 0; 
      if(tf[i] == 1) (*len)++;
    }
    break;
  case GEQ: 
    for(i=0; i<n; i++) {
      if(V[p[i]][var] >= val) tf[i] = 1; 
      else tf[i] = 0; 
      if(tf[i] == 1) (*len)++;
    }
    break;
  case EQ:  
    for(i=0; i<n; i++) {
      if(V[p[i]][var] == val) tf[i] = 1; 
      else tf[i] = 0; 
      if(tf[i] == 1) (*len)++;
    }
    break;
  case LEQ: 
    for(i=0; i<n; i++) {
      if(V[p[i]][var] <= val) tf[i] = 1; 
      else tf[i] = 0; 
      if(tf[i] == 1) (*len)++;
    }
    break;
  case LT:  
    for(i=0; i<n; i++) {
      if(V[p[i]][var] <  val) tf[i] = 1; 
      else tf[i] = 0; 
      if(tf[i] == 1) (*len)++;
    }
    break;
  case NE:  
    for(i=0; i<n; i++) {
      if(V[p[i]][var] != val) tf[i] = 1; 
      else tf[i] = 0; 
      if(tf[i] == 1) (*len)++;
    }
    break;
  default: error("OP not supported");
  }
  
  if(*len == 0) found = NULL;
  else {
    found = new_ivector(*len);
    for(i=0,j=0; i<n; i++) {
      if(tf[i]) {
	found[j] = i;
	j++;
      }
    }
  }
  
  free(tf);
  if(!pv) free(p);
  return found;
}


/*
 * Returns the kth smallest value in the array arr[1..n].  The input
 * array will be rearranged to have this value in location arr[k] ,
 * with all smaller elements moved to arr[1..k-1] (in arbitrary order)
 * and all larger elements in arr[k+1..n] (also in arbitrary order).
 * (from Numerical Recipies in C)
 *
 * This Quickselect routine is based on the algorithm described in
 * "Numerical recipes in C", Second Edition, Cambridge University
 * Press, 1992, Section 8.5, ISBN 0-521-43108-5 This code by Nicolas
 * Devillard - 1998. Public domain.
 */

#define ELEM_SWAP(a,b) { register double t=(a);(a)=(b);(b)=t; }
#define IELEM_SWAP(a,b) { register int t=(a);(a)=(b);(b)=t; }

double quick_select_index(double arr[], int iarr[], int n, int k) 
{
  int low, high ;
  int middle, ll, hh;
  
  low = 0 ; high = n-1 ; 
  assert(k >= low && k <= high);
  for (;;) {
    if (high <= low) /* One element only */
      return arr[k] ;
    
    if (high == low + 1) {  /* Two elements only */
      if (arr[low] > arr[high]) {
        ELEM_SWAP(arr[low], arr[high]) ;
        if(iarr) IELEM_SWAP(iarr[low], iarr[high]) ;
      }
      return arr[k] ;
    }
    
    /* Find kth of low, middle and high items; swap into position low */
    middle = (low + high) / 2;
    if (arr[middle] > arr[high]) {
      ELEM_SWAP(arr[middle], arr[high]) ;
      if(iarr) IELEM_SWAP(iarr[middle], iarr[high]) ;
    }
    if (arr[low] > arr[high]) {
      ELEM_SWAP(arr[low], arr[high]) ;
      if(iarr) IELEM_SWAP(iarr[low],iarr[high]) ;
    }
    if (arr[middle] > arr[low])  {
      ELEM_SWAP(arr[middle], arr[low]) ;
      if(iarr) IELEM_SWAP(iarr[middle],iarr[low]) ;
    }

    /* Swap low item (now in position middle) into position (low+1) */
    ELEM_SWAP(arr[middle], arr[low+1]) ;
    if(iarr) IELEM_SWAP(iarr[middle], iarr[low+1]) ;

    /* Nibble from each end towards middle, swapping items when stuck */
    ll = low + 1;
    hh = high;
    for (;;) {
        do ll++; while (arr[low] > arr[ll]) ;
        do hh--; while (arr[hh]  > arr[low]) ;

        if (hh < ll)
        break;

        ELEM_SWAP(arr[ll], arr[hh]) ;
        if(iarr) IELEM_SWAP(iarr[ll], iarr[hh]) ;
    }

    /* Swap middle item (in position low) back into correct position */
    ELEM_SWAP(arr[low], arr[hh]) ;
    if(iarr) IELEM_SWAP(iarr[low], iarr[hh]) ;

    /* Re-set active partition */
    if (hh <= k)
        low = ll;
    if (hh >= k)
        high = hh - 1;
  }
}

/*
 * same as the quick_select algorithm above, but less
 * efficient.  Not currently used in tgp
 */

double kth_smallest(double a[], int n, int k)
{
  int i,j,l,m ;
  double x ;
  
  l=0 ; m=n-1 ;
  while (l<m) {
    x=a[k] ;
    i=l ;
    j=m ;
    do {
      while (a[i]<x) i++ ;
      while (x<a[j]) j-- ;
      if (i<=j) {
	ELEM_SWAP(a[i],a[j]) ;
	i++ ; j-- ;
      }
    } while (i<=j) ;
    if (j<k) l=i ;
    if (k<i) m=j ;
  }
  return a[k] ;
}
#undef ELEM_SWAP


/* 
 * send mean of the columns of the matrix M
 * out to a file 
 */

void mean_to_file(const char *file_str, double **M, unsigned int T, unsigned int n)
{
  double *Mm;
  FILE *MmOUT;
  unsigned int i;
  
  Mm = (double*) malloc(sizeof(double) * n);
  wmean_of_columns(Mm, M, T, n, NULL);
  MmOUT = fopen(file_str, "w");
  assert(MmOUT);
  for(i=0; i<n; i++) MYprintf(MmOUT, "%g\n", Mm[i]);
  fclose(MmOUT);
  free(Mm);
}


/* 
 * send a vector
 * of the matrix M out to a file 
 */

void vector_to_file(const char* file_str, double* vector, unsigned int n)
{
  FILE* VOUT;
  unsigned int i;
  
  VOUT = fopen(file_str, "w");
  assert(VOUT);
  for(i=0; i<n; i++) MYprintf(VOUT, "%g\n", vector[i]);
  fclose(VOUT);
}


/* 
 * open file with the given name
 * and print the passed matrix to it
 */

void matrix_to_file(const char* file_str, double** matrix, unsigned int n1, unsigned int n2)
{
  FILE* MOUT;
  
  MOUT = fopen(file_str, "w");
  assert(MOUT);
  printMatrix(matrix, n1, n2, MOUT); 
  fclose(MOUT);
}


/* 
 * open file with the given name
 * and print the passed integer matrix to it
 */

void intmatrix_to_file(const char* file_str, int** matrix, unsigned int n1, unsigned int n2)
{
  FILE* MOUT;
  
  MOUT = fopen(file_str, "w");
  assert(MOUT);
  printIMatrix(matrix, n1, n2, MOUT); 
  fclose(MOUT);
}


/* 
 * open file with the given name
 * and print transpose of the passed matrix to it
 */

void matrix_t_to_file(const char* file_str, double** matrix, unsigned int n1, 
		      unsigned int n2)
{
  FILE* MOUT;
  
  MOUT = fopen(file_str, "w");
  assert(MOUT);
  printMatrixT(matrix, n1, n2, MOUT); 
  fclose(MOUT);
}


/*
 * sub_p_matrix:
 *
 * copy the cols v[1:n1][p[n2]] to V.  
 * must have nrow(v) == nrow(V) and ncol(V) >= lenp
 * and ncol(v) >= max(p)
 */

void sub_p_matrix(double **V, int *p, double **v, 
		  unsigned int nrows, unsigned int lenp, 
		  unsigned int col_offset)
{
  int i,j;
  assert(V); assert(p); assert(v); assert(nrows > 0 && lenp > 0);
  for(i=0; i<nrows; i++) for(j=0; j<lenp; j++) 
    V[i][j+col_offset] = v[i][p[j]];
}


/*
 * new_p_subvector:
 *
 * create a new vector from the entries of v, specified
 * by p.  Must have have len(v) == len(V) and ncol(V) >= ncols
 * and ncol(v) >= max(p)
 */

double **new_p_submatrix(int *p, double **v, unsigned int nrows, 
			 unsigned int ncols, unsigned int col_offset)
{
  double **V;
  if(nrows == 0 || ncols+col_offset == 0) return NULL;
  V = new_matrix(nrows, ncols + col_offset);
  if(ncols > 0) sub_p_matrix(V, p, v, nrows, ncols, col_offset);
  return(V);
}


/*
 * sub_p_matrix_rows:
 *
 * copy the rows v[1:n1][p[n2]] to V.  
 * must have ncol(v) == ncol(V) and nrow(V) >= lenp
 * and nrow(v) >= max(p)
 */

void sub_p_matrix_rows(double **V, int *p, double **v, 
		       unsigned int ncols, unsigned int lenp, 
		       unsigned int row_offset)
{
  int i;
  assert(V); assert(p); assert(v); assert(ncols > 0 && lenp > 0);
  for(i=0; i<lenp; i++) 
    dupv(V[i+row_offset], v[p[i]], ncols);
}


/*
 * new_p_submatrix_rows:
 *
 * create a new matrix from the rows of v, specified
 * by p.  Must have have ncol(v) == ncol(V) and nrow(V) >= nrows
 * and nrow(v) >= max(p)
 */

double **new_p_submatrix_rows(int *p, double **v, unsigned int nrows, 
			      unsigned int ncols, unsigned int row_offset)
{
  double **V;
  if(nrows+row_offset == 0 || ncols == 0) return NULL;
  V = new_matrix(nrows + row_offset, ncols);
  if(nrows > 0) sub_p_matrix_rows(V, p, v, ncols, nrows, row_offset);
  return(V);
}

/*
 * copy_p_matrix:
 *
 * copy v[n1][n2] to V into the positions specified by p1[n1] and p2[n2]
 */

void copy_p_matrix(double **V, int *p1, int *p2, double **v, 
		unsigned int n1, unsigned int n2)
{
  int i,j;
  assert(V); assert(p1); assert(p2); assert(n1 > 0 && n2 > 0);
  for(i=0; i<n1; i++) for(j=0; j<n2; j++) 
    V[p1[i]][p2[j]] = v[i][j];
}


/*
 * enforce that means should lie within the quantiles,
 * to guard agains numerical instabilities arising in
 * prediction.  when violated, replace with median
 */

void check_means(double *mean, double *q1, double *median, 
		 double *q2, unsigned int n)
{
  unsigned int i;
  int replace = 0;
  for(i=0; i<n; i++) {
    if(mean[i] > q2[i] || mean[i] < q1[i]) {
      MYprintf(MYstdout, "replacing %g with (%g,%g,%g)\n", 
	       mean[i], q1[i], median[i], q2[i]);
      mean[i] = median[i];
      replace++;
    }
  }
  
  /* let us know what happened */
  if(replace > 0) 
    MYprintf(MYstdout, "NOTICE: %d predictive means replaced with medians\n", 
	     replace);
}


/*
 * pass back the indices (through p) into the matrix X which lie
 * within the boundaries described by rect; return the number of true
 * indices.  X is treated as n1 x n2, and p is an n1 (preallocated)
 * array
 */  

unsigned int matrix_constrained(int *p, double **X, unsigned int n1, 
				unsigned int n2, Rect *rect)
{
  unsigned int i,j, count;
  count = 0;
  /* printRect(MYstderr, rect->d, rect->boundary); */
  for(i=0; i<n1; i++) {
    p[i] = 1;
    for(j=0; j<n2; j++) {
      if(rect->opl[j] == GT) {
	assert(rect->opr[j] == LEQ);
	p[i] = (int) (X[i][j] > rect->boundary[0][j] && 
		      X[i][j] <= rect->boundary[1][j]);
      }
      else if(rect->opl[j] == GEQ) {
	if(rect->opr[j] == LEQ)
	  p[i] = (int) (X[i][j] >= rect->boundary[0][j] && 
			X[i][j] <= rect->boundary[1][j]);
	else if(rect->opr[j] == LT)
	  p[i] = (int) (X[i][j] >= rect->boundary[0][j] && 
			X[i][j] < rect->boundary[1][j]);
	else assert(0);
      }
      else assert(0);
      if(p[i] == 0) break;
    }
    if(p[i] == 1) count++;     
  }
  return count;
}


/*
 * create a new rectangle structure without any of the fields filled
 * in
 */

Rect* new_rect(unsigned int d)
{
  Rect* rect = (Rect*) malloc(sizeof(struct rect));
  rect->d = d;
  rect->boundary = new_matrix(2, d);
  rect->opl = (FIND_OP *) malloc(sizeof(FIND_OP) * d);
  rect->opr = (FIND_OP *) malloc(sizeof(FIND_OP) * d);
  return rect;
}


/*
 * create a new rectangle structure with the boundary populated
 * by the contents of a double array
 */

Rect* new_drect(double **drect, int d)
{
  unsigned int i;
  Rect *rect = new_rect(d);
  for(i=0; i<d; i++) {
    rect->boundary[0][i] = drect[0][i];
    rect->boundary[1][i] = drect[1][i];
    rect->opl[i] = GEQ;
    rect->opr[i] = LEQ;
  }
  return rect;
}


/*
 * return a pointer to a duplicated rectangle structure
 */

Rect* new_dup_rect(Rect* oldR)
{
  unsigned int i;
  Rect* rect = (Rect*) malloc(sizeof(struct rect));
  rect->d = oldR->d;
  rect->boundary = new_dup_matrix(oldR->boundary, 2, oldR->d);
  rect->opl = (FIND_OP *) malloc(sizeof(FIND_OP) * rect->d);
  rect->opr = (FIND_OP *) malloc(sizeof(FIND_OP) * rect->d);
  for(i=0; i<rect->d; i++) {
    rect->opl[i] = oldR->opl[i];
    rect->opr[i] = oldR->opr[i];
  }
  return rect;
}


/*
 * calculate and return the area depicted by
 * the rectangle boundaries
 */

double rect_area(Rect* rect)
{
  unsigned int i;
  double area;
  
  area = 1.0;
  for(i=0; i<rect->d; i++)
    area *= rect->boundary[1][i] - rect->boundary[0][i];
  return area;
}


/*
 * calculate and return the area depicted by
 * the rectangle boundaries, using only dimensions 0,...,maxd-1
 */

double rect_area_maxd(Rect* rect, unsigned int maxd)
{
  unsigned int i;
  double area;
  
  assert(maxd <= rect->d);
  area = 1.0;
  for(i=0; i<maxd; i++)
    area *= rect->boundary[1][i] - rect->boundary[0][i];
  return area;
}


/*
 * print a rectangle structure out to
 * the file denoted by "outfile"
 */

void print_rect(Rect *r, FILE* outfile)
{
  unsigned int i;
  MYprintf(outfile, "# %d dim rect (area=%g) with boundary:\n", 
	   r->d, rect_area(r));
  printMatrix(r->boundary, 2, r->d, outfile);
  MYprintf(outfile, "# opl and opr\n");
  for(i=0; i<r->d; i++) MYprintf(outfile, "%d ", r->opl[i]);
  MYprintf(outfile, "\n");
  for(i=0; i<r->d; i++) MYprintf(outfile, "%d ", r->opr[i]);
  MYprintf(outfile, "\n");
}


/*
 * free the memory associated with a
 * rectangle structure
 */

void delete_rect(Rect *rect)
{
  delete_matrix(rect->boundary);
  free(rect->opl);
  free(rect->opr);
  free(rect);
}


/*
 * make it so that the data lives in
 * [0,1]^d.
 */

void normalize(double **X, double **rect, int N, int d, double normscale)
{
  int i, j;
  double norm;
  if(N == 0) return;
  assert(d != 0);
  for(i=0; i<d; i++) {
    norm = fabs(rect[1][i] - rect[0][i]);
    if(norm == 0) norm = fabs(rect[0][i]);
    for(j=0; j<N; j++) {
      if(rect[0][i] < 0) 
	X[j][i] = (X[j][i] + fabs(rect[0][i])) / norm;
      else
	X[j][i] = (X[j][i] - rect[0][i]) / norm;
      X[j][i] = normscale * X[j][i];
      /* if(!(X[j][i] >=0 && X[j][i] <= normscale))
	MYprintf(MYstdout, "X[%d][%d] = %g, normscale = %g\n", j, i, X[j][i], normscale);
	assert(X[j][i] >=0 && X[j][i] <= normscale); */
    }
  }
}


/*
 * put Rect r on the scale of double rect
 * r should be form 0 to NORMSCALE
 */

void rect_unnorm(Rect* r, double **rect, double normscale)
{
  int i;
  double norm;
  for(i=0; i<r->d; i++) {
    assert(r->boundary[0][i] >= 0 && r->boundary[1][i] <= normscale);
    norm = fabs(rect[1][i] - rect[0][i]);
    if(norm == 0) norm = fabs(rect[0][i]);
    r->boundary[1][i] = normscale * r->boundary[1][i];
    r->boundary[0][i] = rect[0][i] + norm * r->boundary[0][i];
    r->boundary[1][i] = rect[1][i] - norm * (1.0 - r->boundary[1][i]);
  }
}


/*
 * allocates a new double array of size n1
 */

double* new_vector(unsigned int n)
{
  double *v;
  if(n == 0) return NULL;
  v = (double*) malloc(sizeof(double) * n);
  return v;
}


/*
 * allocates a new double array of size n1
 * and fills it with zeros
 */

double* new_zero_vector(unsigned int n)
{
  double *v;
  v = new_vector(n);
  zerov(v, n);
  return v;
}


/*
 * allocates a new double array of size n1
 * and fills it with the contents of vold
 */

double* new_dup_vector(double* vold, unsigned int n)
{
  double *v;
  v = new_vector(n);
  dupv(v, vold, n);
  return v;
}


/*
 * copies vold to v 
 * (assumes v has already been allcocated)
 */

void dupv(double *v, double* vold, unsigned int n)
{
  unsigned int i;
  for(i=0; i<n; i++) v[i] = vold[i];
}


/*
 * copies v into the col-th column of M
 * (assumes M and v are properly allocated already)
 */

void dup_col(double **M, unsigned int col, double *v, unsigned int n)
{
  unsigned int i;
  for(i=0; i<n; i++) M[i][col] = v[i];
}


/* 
 * sumv:
 *
 * return the sum of the contents of the vector
 */

double sumv(double *v, unsigned int n)
{
  unsigned int i;
  double s;
  if(n==0) return 0;
  assert(v);
  s = 0;
  for(i=0; i<n; i++) s += v[i];
  return(s);
}


/* 
 * meanv:
 *
 * return the mean of the contents of the vector
 */

double meanv(double *v, unsigned int n)
{
  return(sumv(v, n)/n);
}


/*
 * equalv:
 *
 * returns 1 if the vectors are equal, 0 otherwise
 */

int equalv(double *v1, double *v2, int n)
{
  unsigned int i;
  for(i=0; i<n; i++) if(v1[i] != v2[i]) return(0);
  return(1);
}


/* 
 * sumiv:
 *
 * return the sum of the contents of the integer vector
 */

int sumiv(int *iv, unsigned int n)
{
  unsigned int i;
  int s;
  if(n==0) return 0;
  assert(iv);
  s = 0;
  for(i=0; i<n; i++) s += iv[i];
  return(s);
}


/* 
 * meaniv:
 *
 * return the mean of the contents of the integer vector
 */

int meaniv(int *iv, unsigned int n)
{
  return((int) (sumiv(iv, n)/n));
}


/* 
 * sum_fv:
 *
 * return the sum of the contents of the vector
 * each entry of which is applied to function f
 */

double sum_fv(double *v, unsigned int n, double(*f)(double))
{
  unsigned int i;
  double s;
  if(n==0) return 0;
  assert(v);
  s = 0;
  for(i=0; i<n; i++) s += f(v[i]);
  return(s);
}


/*
 * swaps the pointer of v2 to v1, and vice-versa
 * (avoids copying via dupv)
 */

void swap_vector(double **v1, double **v2)
{
  double* temp;
  temp = (double*) *v1;
  *v1 = *v2;
  *v2 = (double*) temp;
}


/*
 * zeros out v
 * (assumes that it has already been allocated)
 */

void zerov(double*v, unsigned int n)
{
  unsigned int i;
  for(i=0; i<n; i++) v[i] = 0;
}


/*
 * multiply the contents of vector v[n]
 * by the scale parameter
 */

void scalev(double *v, unsigned int n, double scale)
{
  int i;
  assert(v);
  for(i=0; i<n; i++) v[i] = v[i]*scale;
}


/*
 * multiply the contents of vector v[n]
 * by the scale[n] parameter
 */

void scalev2(double *v, unsigned int n, double *scale)
{
  int i;
  assert(v);
  assert(scale);
  for(i=0; i<n; i++) v[i] = v[i]*scale[i];
}


/*
 * subtract the center value from each component
 * of v[n]
 */

void centerv(double *v, unsigned int n, double center)
{
  int i;
  assert(v);
  for(i=0; i<n; i++) v[i] = v[i] - center;
}

 
/*
 * divide off the norm[n] value from each component
 * of v[n]
 */

void normv(double *v, unsigned int n, double* norm)
{
  int i;
  assert(v);
  assert(norm);
  for(i=0; i<n; i++) v[i] /= norm[i];
}


/*
 * copy v[n] to V into the positions specified by p[n]
 */

void copy_p_vector(double *V, int *p, double *v, unsigned int n)
{
  int i;
  assert(V); assert(p); assert(n > 0);
  for(i=0; i<n; i++) V[p[i]] = v[i];
}


/*
 * copy v[p[i]] to V[n]
 */

void copy_sub_vector(double *V, int *p, double *v, unsigned int n)
{
  int i;
  if(n == 0) return;
  assert(V); assert(p); assert(n > 0);
  for(i=0; i<n; i++) V[i] = v[p[i]];
}


/*
 * new n-vector V; copy v[p[i]] to V [n]
 */

double* new_sub_vector(int *p, double *v, unsigned int n)
{
  double *V = new_vector(n);
  copy_sub_vector(V, p, v, n);
  return V;
}


/*
 * add two vectors of the same size 
 * v1 = a*v1 + b*v2
 */

void add_vector(double a, double *v1, double b, double *v2, unsigned int n)
{
  if(n == 0) return;
  assert(n > 0);
  assert(v1 && v2);
  add_matrix(a, &v1, b, &v2, 1, n);
}


/*
 * add two integer vectors of the same size 
 * v1 = v1 + v2
 */

void add_ivector(int *v1, int *v2, unsigned int n)
{
  unsigned int i;
  if(n == 0) return;
  assert(n > 0);
  assert(v1 && v2);
  for(i=0; i<n; i++) v1[i] += v2[i];
}


/*
 * add_p_vector:
 *
 * add v[n1] to V into the positions specified by p[n1]
 */

void add_p_vector(double a, double *V, int *p, double b, double *v, unsigned int n)
{
  int i = 0;
  if(n == 0) return;
  assert(V); assert(p);
  add_p_matrix(a, &V, &i, p, b, &v, 1, n);
}


/*
 * printing a vector out to outfile
 */

void printVector(double *v, unsigned int n, FILE *outfile, PRINT_PREC type)
{
  unsigned int i;
  if(type==HUMAN) for(i=0; i<n; i++) MYprintf(outfile, "%g ", v[i]);
  else if(type==MACHINE) for(i=0; i<n; i++) MYprintf(outfile, "%.20f ", v[i]);
  else error("bad PRINT_PREC type");
  MYprintf(outfile, "\n");
}


/*
 * printing a symmetric matrix in vector format out to outfile
 */

void printSymmMatrixVector(double **m, unsigned int n, FILE *outfile, 
			   PRINT_PREC type)
{
  unsigned int i,j;
  if(type==HUMAN)
    for(i=0; i<n; i++) 
      for(j=i; j<n; j++) 
	MYprintf(outfile, "%g ", m[i][j]);
  else if(type==MACHINE) 
    for(i=0; i<n; i++) 
      for(j=i; j<n; j++) 
	MYprintf(outfile, "%.20f ", m[i][j]);
  else error("bad PRINT_PREC type");
  MYprintf(outfile, "\n");
}


/*
 * return the minimum element in the vector.
 * pass back the index of the minimum through 
 * the which pointer
 */

double min(double *v, unsigned int n, unsigned int *which)
{
  unsigned int i;
  double min;
  
  *which = 0;
  min = v[0];
  
  for(i=1; i<n; i++) {
    if(v[i] < min)  {
      min = v[i];
      *which = i;
    }
  }
  
  return min;
}


/*
 * return the maximum element in the vector.  pass back the index of
 * the maximum through the which pointer
 */

double max(double *v, unsigned int n, unsigned int *which)
{
  unsigned int i;
  double max;
  
  *which = 0;
  max = v[0];
  
  for(i=1; i<n; i++) {
    if(v[i] > max)  {
      max = v[i];
      *which = i;
    }
  }
  
  return max;
}


/*
 * new vector of integers of length n
 */

int *new_ivector(unsigned int n)
{
  int *iv;
  if(n == 0) return NULL;
  iv = (int*)  malloc(sizeof(int) * n);
  assert(iv);
  return iv;
}


/*
 * duplicate the integer contents of iv of length n into the already
 * allocated vector iv_new, also of length n
 */

void dupiv(int *iv_new, int *iv, unsigned int n)
{
  unsigned int i;
  if(n > 0) assert(iv && iv_new);
  for(i=0; i<n; i++) iv_new[i] = iv[i];
}


/*
 * zeros out v
 * (assumes that it has already been allocated)
 */

void zeroiv(int*v, unsigned int n)
{
  unsigned int i;
  for(i=0; i<n; i++) v[i] = 0;
}


/*
 * swaps the pointer of v2 to v1, and vice-versa
 * (avoids copying via dupv)
 */

void swap_ivector(int **v1, int **v2)
{
  int* temp;
  temp = (int*) *v1;
  *v1 = *v2;
  *v2 = (int*) temp;
}


/*
 * allocate a new integer vector of length n and copy the integer
 * contents of iv into it
 */

int *new_dup_ivector(int *iv, unsigned int n)
{
  int* iv_new = new_ivector(n);
  dupiv(iv_new, iv, n);
  return iv_new;
}


/*
 * create a new integer vector of length n, fill it with ones,
 * multiplied by the scale parameter-- for a vector of 5's, use
 * scale=5
 */

int *new_ones_ivector(unsigned int n, int scale)
{
  int *iv = new_ivector(n);
  iones(iv, n, scale);
  return iv;
}


/*
 * create a new integer vector of length n, fill it with zeros
 */

int *new_zero_ivector(unsigned int n)
{
  int *iv = new_ivector(n);
  zeroiv(iv, n);
  return iv;
}


/*
 * write n ones into iv (pre-allocated), and then multiply by the
 * scale parameter-- for a vector of 5's, use scale=5
 */

void iones(int *iv, unsigned int n, int scale)
{
  unsigned int i;
  if(n > 0) assert(iv);
  for(i=0; i<n; i++) iv[i] = scale;
}


/*
 * printing an integer vector out to outfile
 */

void printIVector(int *iv, unsigned int n, FILE *outfile)
{
  unsigned int i;
  for(i=0; i<n; i++) MYprintf(outfile, "%d ", iv[i]);
  MYprintf(outfile, "\n");
}


/* 
 * send an integer vector
 * of the matrix M out to a file 
 */

void ivector_to_file(const char* file_str, int* vector, unsigned int n)
{
  FILE* VOUT;
  unsigned int i;
  
  VOUT = fopen(file_str, "w");
  assert(VOUT);
  for(i=0; i<n; i++) MYprintf(VOUT, "%d\n", vector[i]);
  fclose(VOUT);
}


/*
 * copy v[n] to V into the positions specified by p[n]
 */

void copy_p_ivector(int *V, int *p, int *v, unsigned int n)
{
  int i;
  assert(V); assert(p); assert(n > 0);
  for(i=0; i<n; i++) V[p[i]] = v[i];
}


/*
 * copy v[p[i]] to V[n]
 */

void copy_sub_ivector(int *V, int *p, int *v, unsigned int n)
{
  int i;
  assert(V); assert(p); assert(n > 0);
  for(i=0; i<n; i++) V[i] = v[p[i]];
}


/*
 * new n-vector V; copy v[p[i]] to V [n]
 */

int* new_sub_ivector(int *p, int *v, unsigned int n)
{
  int *V = new_ivector(n);
  copy_sub_ivector(V, p, v, n);
  return V;
}


/*
 * casting used on above function for vectors
 * of UNSIGNED integers.
 */

unsigned int *new_uivector(unsigned int n)
{ return (unsigned int*) new_ivector(n); }

void dupuiv(unsigned int *iv_new, unsigned int *iv, unsigned int n)
{ dupiv((int *) iv_new, (int*) iv, n); }

unsigned int *new_dup_uivector(unsigned int *iv, unsigned int n)
{ return (unsigned int*) new_dup_ivector((int*) iv, n); }

unsigned int *new_ones_uivector(unsigned int n, unsigned int scale)
{ return (unsigned int*) new_ones_ivector(n, (int) scale); }

unsigned int *new_zero_uivector(unsigned int n)
{ return (unsigned int*) new_zero_ivector(n); }

void uiones(unsigned int *iv, unsigned int n, unsigned int scale)
{ iones((int*) iv, n, (int) scale); }

void zerouiv(unsigned int *iv, unsigned int n)
{ zeroiv((int*) iv, n); }

void printUIVector(unsigned int *iv, unsigned int n, FILE *outfile)
{ printIVector((int*) iv, n, outfile); }

void uivector_to_file(const char *file_str, unsigned int *iv, unsigned int n)
{ ivector_to_file(file_str, (int*) iv, n); }

void copy_p_uivector(unsigned int *V, int *p, unsigned int *v, unsigned int n)
{ copy_p_ivector((int*)V, p, (int*)v, n); }

void copy_sub_uivector(unsigned int *V, int *p, unsigned int *v, unsigned int n)
{ copy_sub_ivector((int*)V, p, (int*)v, n); }

unsigned int* new_sub_uivector(int *p, unsigned int *v, unsigned int n)
{ return (unsigned int*) new_sub_ivector(p, (int*)v, n); }

unsigned int sumuiv(unsigned int *v, unsigned int n)
{ return (unsigned int) sumiv((int*)v, n); }

unsigned int meanuiv(unsigned int *v, unsigned int n)
{ return (unsigned int) meaniv((int*)v, n); }


/*
 * sq:
 * 
 * calculate the square of x
 */

double sq(double x)
{
  return x*x;
}


/*
 * MYfmax:
 *
 * seems like some systems are missing the prototype
 * for the fmax function which should be in math.h --
 * so I wrote MY own
 */

double MYfmax(double a, double b)
{
  if(a >= b) return a;
  else return b;
}


/*
 * MYfmin:
 *
 * seems like some systems are missing the prototype
 * for the fmin function which should be in math.h --
 * so I wrote MY own
 */

double MYfmin(double a, double b)
{
  if(a <= b) return a;
  else return b;
}


/*
 * vmult: 
 *
 * returns the product of its arguments
 */

double vmult(double *v1, double *v2, int n)
{
  double v = 0.0;
  int i;
  for(i=0; i<n; i++) v += v1[i]*v2[i]; 
  return v;
}
