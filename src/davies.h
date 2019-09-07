/*
 * This header file was produced for the laGP package, providing prototypes
 * for functions in davies.c copied (with minor edits and enhancements)
 * from the CompQuadForm package for R
 */

#ifndef __DAVIES_H__
#define __DAVIES_H__

void  qfc(double* lb1, double* nc1, int* n1, int *r1, double *sigma, double *c1, int *lim1, double *acc, double* trace, int* ifault, double *res);
void  davies(double* lb1, double* nc1, int* n1, int *r1, double *sigma, 
   double *c1, int *lq, int *lim1, double *acc, double* trace, int* ifault, double *res);

#endif

