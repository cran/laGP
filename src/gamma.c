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


#include "gamma.h"
#ifdef RPRINT
#include <Rmath.h>
#endif
#include <assert.h>

/*
 * Cgamma:
 *
 * (complete) gamma function and its logarithm (all logarithms are base 10)
 * from UCS
 */

double Cgamma(const double a, const int ulog)
{
  double r;
  if(ulog) r = lgammafn(a) / M_LN10;
  else r = gammafn(a);
  /* MYprintf(MYstdout, "Cgamma: a=%g, ulog=%d, r=%g\n", a, ulog, r); */
  assert(!isnan(r));
  return(r);
}


/*
 * Rgamma_inv:
 *
 * regularized gamma inverse function from UCS
 */

double Rgamma_inv(const double a, const double y, const int lower,
                  const int ulog)
{
  double r;
  if(ulog) r = qgamma(y*M_LN10, a, /*scale=*/ 1.0, lower, ulog);
  else r = qgamma(y, a, /*scale=*/ 1.0, lower, ulog);
  /*MYprintf(MYstdout, "Rgamma_inv: a=%g, y=%g, lower=%d, ulog=%d, r=%g\n",
    a, y, lower, ulog, r); */
  assert(!isnan(r));
  return(r);
}


/*
 * Igamma_inv:
 *
 * incomplete gamma inverse function from UCS
 */

double Igamma_inv(const double a, const double y, const int lower,
                  const int ulog)
{
  double r;
  if(ulog) r = Rgamma_inv(a, y - Cgamma(a, ulog), lower, ulog);
  else r = Rgamma_inv(a, y / Cgamma(a, ulog), lower, ulog);
  assert(!isnan(r));
  /* MYprintf(MYstdout, "Rgamma_inv: a=%g, y=%g, lower=%d, ulog=%d, r=%g\n",
     a, y, lower, ulog, r); */
  return(r);
}

/*
 * Igamma_inv_R:
 *
 * function to test the Igamma_inv function in R to compare
 * with the Igamma.inv function in the UCS library
 */

void Igamma_inv_R(double *a, double *y, int *lower, int *ulog, double *result)
{
  *result = Igamma_inv(*a, *y, *lower, *ulog);
}
