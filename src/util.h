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

#ifndef __UTIL_H__
#define __UTIL_H__

double log_determinant_chol(double **M, const unsigned int n);
#ifdef BRENT_FROM_R
extern double Brent_fmin(double ax, double bx, 
          double (*f)(double x, void *info), void *info,
          double tol);
#else
double Brent_fmin(double ax, double bx, double (*f)(double, void *),
      void *info, double tol);
#endif

double MYlbfgsb(int n, double *x, double *l, double *u, 
	double (*fn)(int, double*, void*), 
	void (*gr)(int, double *, double *, void *), 
	int *fail, void *ex, double pgtol, int *counts, int maxit, 
	char *msg, int trace, int fromR);

#endif
