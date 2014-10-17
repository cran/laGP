#include "matrix.h"
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
 * an separable power exponential correlation function 
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
 * diff_covar_sep_symm:
 *
 * calculate the first and 2nd derivative (wrt d) of the correlation (K)
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
