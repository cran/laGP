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


#include <stdlib.h>
#include <assert.h>
#include "linalg.h"
#include "matrix.h"
#include "rhelp.h"

#ifdef FORTPACK
char uplo = 'U';
#endif
/* #define DEBUG */

/*
 * linalg_dtrsv:
 *
 * analog of dtrsv in cblas nad blas
 * assumed row-major lower-tri and non-unit
 */

void linalg_dtrsv(TA, n, A, lda, Y, ldy)
const enum CBLAS_TRANSPOSE TA;
int n, lda, ldy;
double **A;
double *Y;
{
	#ifdef FORTBLAS
	char ta;
	char diag = 'N';
	if(TA == CblasTrans) ta = 'T'; else ta = 'N';
	dtrsv(&uplo, &ta, &diag, &n, *A, &lda, Y, &ldy);
	#else
	cblas_dtrsv(CblasRowMajor,CblasLower,TA,CblasNonUnit,
    	/*cblas_dtrsv(CblasColMajor,CblasUpper,CblasNoTrans,CblasNonUnit,*/
    		n,*A,lda,Y,ldy);
	#endif
}


/*
 * linalg_ddot:
 *
 * analog of ddot in cblas nad blas
 */

double linalg_ddot(n, X, ldx, Y, ldy)
int n, ldx, ldy;
double *X, *Y;
{
  double result;

#ifdef FORTBLAS
  size_t n64,ldx64,ldy64;
  n64 = n; ldx64 = ldx; ldy64=ldy;
  result = ddot(&n64,X,&ldx64,Y,&ldy64);
#else
  result = cblas_ddot(n, X, ldx, Y, ldy);
#endif
  return result;
}


/*
 * linalg_daxpy:
 *
 * analog of daxpy in cblas nad blas
 */

void linalg_daxpy(n,alpha,X,ldx,Y,ldy)
int n, ldx, ldy;
double alpha;
double *X, *Y;
{
#ifdef FORTBLAS
  size_t n64, ldx64, ldy64;
  n64 = n; ldx64 = ldx; ldy64 = ldy;
  /* daxpy(&n,&alpha,X,&ldx,Y,&ldy); */
  daxpy(&n64,&alpha,X,&ldx64,Y,&ldy64);
#else
  cblas_daxpy(n, alpha, X, ldx, Y, ldy);
#endif
}


/*
 * linalg_dgemm:
 *
 * analog of dgemm in cblas nad blas
 * assumed column major representation
 */

void linalg_dgemm(TA, TB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
const enum CBLAS_TRANSPOSE TA, TB;
int m, n, k, lda, ldb, ldc;
double alpha, beta;
double **A, **B, **C;
{
#ifdef FORTBLAS
  size_t m64, n64, k64, lda64, ldb64, ldc64;
  char ta, tb;
  m64 = m; n64 = n; k64 = k; lda64 = lda; ldb64 = ldb; ldc64 = ldc;
  if(TA == CblasTrans) ta = 'T'; else ta = 'N';
  if(TB == CblasTrans) tb = 'T'; else tb = 'N';
  /* dgemm(&ta,&tb,&m,&n,&k,&alpha,*A,&lda,*B,&ldb,&beta,*C,&ldc); */
  dgemm(&ta,&tb,&m64,&n64,&k64,&alpha,*A,&lda64,*B,&ldb64,&beta,*C,&ldc64);
#else
  cblas_dgemm(CblasColMajor,TA,TB,m,n,k,alpha,*A,lda,*B,ldb,beta,*C,ldc);
#endif
}


/*
 * linalg_dgemv:
 *
 * analog of dgemv in cblas nad blas
 * assumed column major representation
 */

void linalg_dgemv(TA, m, n, alpha, A, lda, X, ldx, beta, Y, ldy)
const enum CBLAS_TRANSPOSE TA;
int m, n, lda, ldx, ldy;
double alpha, beta;
double **A;
double *X, *Y;
{
#ifdef FORTBLAS
  size_t m64, n64, lda64, ldx64, ldy64;
  char ta;
  m64 = m; n64 = n, lda64 = lda; ldx64 = ldx; ldy64 = ldy;
  if(TA == CblasTrans) ta = 'T'; else ta = 'N';
  /* dgemv(&ta,&m,&n,&alpha,*A,&lda,X,&ldx,&beta,Y,&ldy); */
  dgemv(&ta,&m64,&n64,&alpha,*A,&lda64,X,&ldx64,&beta,Y,&ldy64);
#else
  cblas_dgemv(CblasColMajor,TA,m,n,alpha,*A,lda,X,ldx,beta,Y,ldy);
#endif
}


/*
 * linalg_dsymm:
 *
 * analog of dsymm in cblas nad blas
 * assumed column major and upper-triangluar representation
 */


void linalg_dsymm(SIDE, m, n, alpha, A, lda, B, ldb, beta, C, ldc)
const enum CBLAS_SIDE SIDE;
int m, n, lda, ldb, ldc;
double alpha, beta;
double **A, **B, **C;
{
#ifdef FORTBLAS
  size_t m64, n64, lda64, ldb64, ldc64;
  char side;
  m64 = m; n64 = n; lda64 = lda; ldb64 = ldb; ldc64 = ldc;
  if(SIDE == CblasRight) side = 'R'; else side = 'L';
  /* dsymm(&side,&uplo,&m,&n,&alpha,*A,&lda,*B,&ldb,&beta,*C,&ldc); */
  dsymm(&side,&uplo,&m64,&n64,&alpha,*A,&lda64,*B,&ldb64,&beta,*C,&ldc64);
#else
  cblas_dsymm(CblasColMajor,SIDE,CblasUpper,m,n,alpha,*A,lda,*B,ldb,beta,*C,ldc);
#endif
}


/*
 * linalg_dsymv:
 *
 * analog of dsymv in cblas and blas
 * assumed column major representation
 */


void linalg_dsymv(n, alpha, A, lda, X, ldx, beta, Y, ldy)
int n, lda, ldx, ldy;
double alpha, beta;
double **A;
double *X, *Y;
{
#ifdef FORTBLAS
  size_t n64, lda64, ldy64, ldx64;
  n64 = n; lda64 = lda; ldx64 = ldx; ldy64 = ldy;
  /* dsymv(&uplo,&n,&alpha,*A,&lda,X,&ldx,&beta,Y,&ldy); */
  dsymv(&uplo,&n64,&alpha,*A,&lda64,X,&ldx64,&beta,Y,&ldy64);
#else
  cblas_dsymv(CblasColMajor,CblasUpper,n,alpha,*A,lda,X,ldx,beta,Y,ldy);
#endif
}

/*
 * linalg_dposv:
 *
 * analog of dposv in clapack and lapack where 
 * Mutil is with colmajor and uppertri or rowmajor
 * and lowertri
 */

int linalg_dposv(n, Mutil, Mi)
int n;
double **Mutil, **Mi;
{
  long info;
	
  /* then use LAPACK */
#ifdef FORTPACK
  size_t n64;
  n64 = n;
  dposv(&uplo,&n64,&n64,*Mutil,&n64,*Mi,&n64,&info);
#else
  /*info = clapack_dposv(CblasColMajor,CblasUpper,n,n,*Mutil,n,*Mi,n);*/
  info = clapack_dposv(CblasRowMajor,CblasLower,n,n,*Mutil,n,*Mi,n);
#endif
  
#ifdef DEBUG
  if(info != 0) {
    matrix_to_file("M.dump", Mutil, n, n);
    error("offending matrix dumped into matrix.dump");
  }
#endif
  
  return (int) info;
}


/*
 * linalg_dgesv:
 *
 * analog of dgesv in clapack and lapack;
 * row or col major doesn't matter because it is
 * assumed that Mutil is symmetric
 * 
 * inverse_lu used this with RowMajor, other with ColMajor
 */

int linalg_dgesv(n, Mutil, Mi)
int n;
double **Mutil, **Mi;
{
	int info;
	int *p;

	p = new_ivector(n);
	#ifdef FORTPACK
	dgesv(&n,&n,*Mutil,&n,p,*Mi,&n,&info);
	#else
	info = clapack_dgesv(CblasColMajor,n,n,*Mutil,n,p,*Mi,n);
	/*info = clapack_dgesv(CblasRowMajor,n,n,*Mutil,n,p,*Mi,n);*/
	#endif
	free(p);
	#ifdef DEBUG
	assert(info == 0);
	#endif

	return info;
}


/*
 *
 * analog of dpotrf in clapack and lapack where 
 * var is with colmajor and uppertri or rowmajor
 * and lowertri
 */

int linalg_dpotrf(n, var)
int n;
double **var;
{
  long info;

#ifdef FORTPACK
  size_t n64;
  n64 = n;
  dpotrf(&uplo,&n64,*var,&n64,&info); 
#else
  info = clapack_dpotrf(CblasRowMajor,CblasLower,n,*var,n);
  /*info = clapack_dpotrf(CblasColMajor,CblasUpper,n,*var,n);*/
#endif
#ifdef DEBUG
  assert(info == 0);
#endif
  
  return (int) info;
}



#ifndef FORTPACK
/*
 * solve_cg_symm:
 * 
 * solve Ax=b by inverting A and computing using the conjugate 
 * gradient method from Skilling (also takes advantage of symmetry in C) 
 * C[n][n] double u[n], y[n], y_star[n]
 */

int solve_cg_symm(y, y_star, C, u, theta, n)
unsigned int n;
double **C;
double *u, *y, *y_star;
double theta;
{
	double g[n], g_star[n], h[n], h_star[n], Ch[n], Ch_star[n]; 
	double Cy[n], Cy_star[n], Ag_star[n], ACh_star[n], Ay_star[n], CAy_star[n];
	double **A;
	double gamma, gamma_star, lambda, lambda_star, g_old_norm, g_old_norm_star, Q, Q_star, u_norm, upper;
	unsigned int k, i, j;/*, iter;*/

	A = new_matrix(n, n);
	u_norm = linalg_ddot(n, u, 1, u, 1);

	/* initialize */
	for(i=0; i<n; i++) {
		for(j=0; j<n; j++) A[i][j] = C[i][j];
		A[i][i] -= theta;
		y[i] = y_star[i] = 0;
		g[i] = g_star[i] = h[i] = h_star[i] = u[i];
		Cy[i] = Cy_star[i] = Ag_star[i] = ACh_star[i] = Ch[i] = Ch_star[i] = Ay_star[i] = CAy_star[i];
	}
	g_old_norm =  linalg_ddot(n, g, 1, g, 1);
	linalg_dsymv(n,1.0,A,n,g_star,1,0.0,Ag_star,1);
	g_old_norm_star =  linalg_ddot(n, g_star, 1, Ag_star, 1);

	/* the main loop */
	for(k=0; k<n; k++) {

		/* Ch = C * h */
		linalg_dsymv(n,1.0,C,n,h,1,0.0,Ch,1);
		linalg_dsymv(n,1.0,C,n,h_star,1,0.0,Ch_star,1);

		/* lambda = g^t * g / ( g^t * Ch) */
		lambda = g_old_norm / linalg_ddot(n, g, 1, Ch, 1)  ;
		lambda_star = g_old_norm / linalg_ddot(n, Ag_star, 1, Ch_star, 1)  ;

		/* y = y + lambda * h */
		for(i=0; i<n; i++) y[i] = y[i] + lambda * h[i];
		for(i=0; i<n; i++) y_star[i] = y[i] + lambda_star * h_star[i];

		/* Q = y^t*u - 0.5 y^t*C*y */
		Q = linalg_ddot(n, y, 1, u, 1);
		linalg_dsymv(n,1.0,C,n,y,1,0.0,Cy,1);
		Q -= 0.5 * linalg_ddot(n, y, 1, Cy, 1);

		/* Q_star = y_star^t*A*u - 0.5 y_star^t*C*A*y */
		linalg_dsymv(n,1.0,A,n,y_star,1,0.0,Ay_star,1);
		Q_star = linalg_ddot(n, Ay_star, 1, u, 1);
		linalg_dsymv(n,1.0,C,n,Ay_star,1,0.0,CAy_star,1);
		Q_star -= 0.5 * linalg_ddot(n, y_star, 1, CAy_star, 1);

		/* see if we're close */
		upper = (0.5*u_norm - Q_star)/theta;
		if((upper - Q) / Q < 1e-6) break;

		/* stuff for next round */

		/* g = g - lambda * C * h */
		for(i=0; i<n; i++) g[i] = g[i] - lambda * Ch[i];
		for(i=0; i<n; i++) g_star[i] = g_star[i] - lambda_star * Ch_star[i];

		/* gamma = (g*g) / g_old_norm */
		gamma = 1.0 / g_old_norm;
		g_old_norm = linalg_ddot(n, g, 1, g, 1);
		gamma *= g_old_norm;
		gamma_star = 1.0 / g_old_norm_star;
		linalg_dsymv(n,1.0,A,n,g_star,1,0.0,Ag_star,1);
		g_old_norm_star = linalg_ddot(n, g_star, 1, Ag_star, 1);
		gamma_star *= g_old_norm_star;

		/* h = g + gamma * h */
		for(i=0; i<n; i++) h[i] = g[i] + gamma * h[i];
		for(i=0; i<n; i++) h_star[i] = g_star[i] + gamma_star * h_star[i];
	}

	delete_matrix(A);
	return k;
}
#endif


