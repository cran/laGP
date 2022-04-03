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


#include <assert.h>
#include <stdlib.h>
#include <math.h>
#ifdef RPRINT
#include <R.h>
#include <Rmath.h>
#endif
#include <R_ext/Applic.h>
#include "rhelp.h"
#include "util.h"
#include "matrix.h"
#ifdef _OPENMP
  #include <omp.h>
#endif

/*
 * log_determinant_chol:
 *
 * returns the log determinant of the n x n
 * choleski decomposition of a matrix M
 */

double log_determinant_chol(double **M, const unsigned int n)
{
  double log_det;
  unsigned int i;

  /* det = prod(diag(R)) .^ 2 */
  log_det = 0;
  for(i=0; i<n; i++) log_det += log(M[i][i]);
  log_det = 2*log_det;

  return log_det;
}


/* the C-function behind uniroot from R */
#ifndef BRENT_FROM_R
double Brent_fmin(double ax, double bx, double (*f)(double, void *),
      void *info, double tol)
{
    /*  c is the squared inverse of the golden ratio */
    const double c = (3. - sqrt(5.)) * .5;

    /* Local variables */
    double a, b, d, e, p, q, r, u, v, w, x;
    double t2, fu, fv, fw, fx, xm, eps, tol1, tol3;

/*  eps is approximately the square root of the relative machine precision. */
    eps = DBL_EPSILON;
    tol1 = eps + 1.;/* the smallest 1.000... > 1 */
    eps = sqrt(eps);

    a = ax;
    b = bx;
    v = a + c * (b - a);
    w = v;
    x = v;

    d = 0.;/* -Wall */
    e = 0.;
    fx = (*f)(x, info);
    fv = fx;
    fw = fx;
    tol3 = tol / 3.;

/*  main loop starts here ----------------------------------- */

    for(;;) {
  xm = (a + b) * .5;
  tol1 = eps * fabs(x) + tol3;
  t2 = tol1 * 2.;

  /* check stopping criterion */

  if (fabs(x - xm) <= t2 - (b - a) * .5) break;
  p = 0.;
  q = 0.;
  r = 0.;
  if (fabs(e) > tol1) { /* fit parabola */

      r = (x - w) * (fx - fv);
      q = (x - v) * (fx - fw);
      p = (x - v) * q - (x - w) * r;
      q = (q - r) * 2.;
      if (q > 0.) p = -p; else q = -q;
      r = e;
      e = d;
  }

  if (fabs(p) >= fabs(q * .5 * r) ||
      p <= q * (a - x) || p >= q * (b - x)) { /* a golden-section step */

      if (x < xm) e = b - x; else e = a - x;
      d = c * e;
  }
  else { /* a parabolic-interpolation step */

      d = p / q;
      u = x + d;

      /* f must not be evaluated too close to ax or bx */

      if (u - a < t2 || b - u < t2) {
    d = tol1;
    if (x >= xm) d = -d;
      }
  }

  /* f must not be evaluated too close to x */

  if (fabs(d) >= tol1)
      u = x + d;
  else if (d > 0.)
      u = x + tol1;
  else
      u = x - tol1;

  fu = (*f)(u, info);

  /*  update  a, b, v, w, and x */

  if (fu <= fx) {
      if (u < x) b = x; else a = x;
      v = w;    w = x;   x = u;
      fv = fw; fw = fx; fx = fu;
  } else {
      if (u < x) a = u; else b = u;
      if (fu <= fw || w == x) {
        v = w; fv = fw;
        w = u; fw = fu;
      } else if (fu <= fv || v == x || v == w) {
        v = u; fv = fu;
      }
  }
    }
    /* end of main loop */

    return x;
}
#endif


double MYlbfgsb(int n, double *x, double *l, double *u, optimfn fn, 
  optimgr gr, int *fail, void *ex, double pgtol, int *counts, int maxit, 
  char *msg, int trace, int fromR)
{
  int *nbd;
  int k;
  double val = 0;

  nbd = new_ivector(n);
  for(k=0; k<n; k++) nbd[k] = 2;

  if(fromR) {
    lbfgsb(n, 5, x, l, u, nbd, &val, fn, gr, fail, ex, 
    1e7, pgtol, counts, counts+1, maxit, msg, trace, 10);
  } else {

#ifdef _OPENMP
    #pragma omp critical
    {
#endif  
      lbfgsb(n, 5, x, l, u, nbd, &val, fn, gr, fail, ex, 
        1e7, pgtol, counts, counts+1, maxit, msg, trace, 10);
#ifdef _OPENMP
    }
#endif
  }
  free(nbd);

  /* return the objective value */
  return(val);
}
