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
#include "linalg.h"
#include "rhelp.h"
#include "covar.h"
#include "ieci.h"
#include <stdlib.h>
#include <assert.h>
#ifdef RPRINT
#include <R.h>
#include <Rmath.h>
#else 
#include <math.h>
#endif


#ifdef RPRINT
/*
 * EI:
 *
 * calculates the expected improvement following
 * Williams et al by integrating over the parameters
 * to the GP predictive
 */

double EI(const double m, const double s2, const int df, 
	  const double fmin)
{
  double diff, sd, diffs, scale, ei;

  diff = fmin - m;
  sd = sqrt(s2);
  diffs = diff/sd;
  scale = (df*sd + sq(diff)/sd)/(df-1.0);
  ei = diff*pt(diffs, (double) df, 1, 0);
  ei += scale*dt(diffs, (double) df, 0);

  return(ei);
}
#else /* this is so we can compile this file for mex/Matlab */
double EI(const double m, const double s2, const int df, 
	  const double fmin)
{
  error("EI not defined due to missing pt( and dt(");
  return(0);
}
#endif


/*
 * calc_ecis:
 *
 * function that iterates over the m Xref locations, and the
 * statis calculated by previous calc_* function in order to 
 * use the EI function to calculate the ECI relative to the
 * reference locations -- writes over ktKik input
 */

void calc_ecis(const int m, double *ktKik, double *s2p, const double phi, 
	       const double g, double *badj, double *tm, const double tdf, 
	       const double fmin)
{
  int i;
  double zphi, ts2;

  for(i=0; i<m; i++) {
    zphi = (s2p[1] + phi)*(1.0 + g - ktKik[i]);
    ts2 = badj[i] * zphi / (s2p[0] + tdf);
    ktKik[i] = EI(tm[i], ts2, tdf, fmin);
  }
}


/*
 * calc_ieci:
 *
 * function that iterates over the m Xref locations, and the
 * statis calculated by previous calc_* function in order to 
 * use the EI function to calculate the ECI relative to the
 * reference locations, and then averaves to create the IECI
 */

double calc_ieci(const int m, double *ktKik, double *s2p, const double phi, 
		 const double g, double *badj, double *tm, const double tdf, 
		 const double fmin, double *w)
{

  int i;
  double zphi, ts2, eci, ieci;

  ieci = 0.0;
  for(i=0; i<m; i++) {
    zphi = (s2p[1] + phi)*(1.0 + g - ktKik[i]);
    ts2 = badj[i] * zphi / (s2p[0] + tdf);
    eci = EI(tm[i], ts2, tdf, fmin);
    if(w) ieci += w[i]*eci;
    else ieci += eci;
  }

  return (ieci/m);
}


/*
 * calc_alc:
 *
 * function that iterates over the m Xref locations, and the
 * stats calculated by previous calc_* function in order to 
 * calculate the reduction in variance
 */

double calc_alc(const int m, double *ktKik, double *s2p, const double phi, 
		double *badj, const double tdf, double *w)
{
  int i;
  double zphi, ts2, alc, dfrat;
  
  dfrat = tdf/(tdf - 2.0);
  alc = 0.0;
  for(i=0; i<m; i++) {
    zphi = (s2p[1] + phi)*ktKik[i];
    if(badj) ts2 = badj[i] * zphi / (s2p[0] + tdf);
    else ts2 = zphi / (s2p[0] + tdf);
    if(w) alc += w[i]*dfrat*ts2;
    else alc += ts2*dfrat; 
  }

  return (alc/m);
}


/*
 * calc_g_mui_kxy:
 *
 * function for calculating the g vector, mui scalar, and
 * kxy vector for the IECI calculation; kx is length-n 
 * utility space -- only implements isotropic case; separable
 * version implemented in plgp source tree;
 */

void calc_g_mui_kxy(const int col, double *x, double **X, 
		    const int n, double **Ki, double **Xref, 
		    const int m, double d, const double g, double *gvec, 
        double *mui, double *kx, double *kxy)
{
  double mu_neg;
  int i;

  /* sanity check */
  if(m == 0) assert(!kxy && !Xref);

  /* kx <- drop(covar(X1=pall$X, X2=x, d=Zt$d, g=Zt$g)) */
  covar(col, &x, 1, X, n, d, g, &kx);
  /* kxy <- drop(covar(X1=x, X2=Xref, d=Zt$d, g=0)) */
  if(m > 0) covar(col, &x, 1, Xref, m, d, 0.0, &kxy);

  /* Kikx <- drop(util$Ki %*% kx) stored in gvex */
  linalg_dsymv(n,1.0,Ki,n,kx,1,0.0,gvec,1);

  /* mui <- drop(1 + Zt$g - t(kx) %*% Kikx) */
  *mui = 1.0 + g - linalg_ddot(n, kx, 1, gvec, 1);
  
  /* gvec <- - Kikx/mui */
  mu_neg = 0.0 - 1.0/(*mui);
  for(i=0; i<n; i++) gvec[i] *= mu_neg;
}


/*
 * calc_ktKikx:
 *
 * function for calculating the ktKikx vector used in the
 * IECI calculation -- writes over the KtKik input --
 * R interface (calc_ktKikx_R) available in plgp source tree
 */

void calc_ktKikx(double *ktKik, const int m, double **k, const int n,
		 double *g, const double mui, double *kxy, double **Gmui_util,
		 double *ktGmui_util, double *ktKikx)
{
  int i;
  double **Gmui;
  double *ktGmui;

  /* first calculate Gmui = g %*% t(g)/mu */
  if(!Gmui_util) Gmui = new_matrix(n, n);
  else Gmui = Gmui_util;
  linalg_dgemm(CblasNoTrans,CblasTrans,n,n,1,
               mui,&g,n,&g,n,0.0,Gmui,n);

  /* used in the for loop below */
  if(!ktGmui_util) ktGmui = new_vector(n);
  else ktGmui = ktGmui_util;

  /* loop over all of the m candidates */
  for(i=0; i<m; i++) {

    /* ktGmui = drop(t(k) %*% Gmui) */
    /* zerov(ktGmui, n); */
    linalg_dsymv(n,1.0,Gmui,n,k[i],1,0.0,ktGmui,1);

    /* ktKik += diag(t(k) %*% (g %*% t(g) * mui) %*% k) */
    if(ktKik) ktKikx[i] = ktKik[i] + linalg_ddot(n, ktGmui, 1, k[i], 1);
    else ktKikx[i] = linalg_ddot(n, ktGmui, 1, k[i], 1);

    /* ktKik.x += + 2*diag(kxy %*% t(g) %*% k) */
    ktKikx[i] += 2.0*linalg_ddot(n, k[i], 1, g, 1)*kxy[i];

    /* ktKik.x + kxy^2/mui */
    ktKikx[i] += sq(kxy[i])/mui;
  }

  /* clean up */
  if(!ktGmui_util) free(ktGmui);
  if(!Gmui_util) delete_matrix(Gmui);
}


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
  for(j=0; j<=m; j++) if(rect[1][j] > ystar) rect[1][j] = ystar;

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
