#ifndef __GAMMA_H__
#define __GAMMA_H__

double Rgamma_inv(const double a, const double y, const int lower,
                  const int ulog);
double Igamma_inv(const double a, const double y, const int lower,
                  const int ulog);
void Igamma_inv_R(double *a, double *y, int *lower, int *ulog, double *result);

#endif

