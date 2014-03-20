#ifndef __ALC_GPU_H__
#define __ALC_GPU_H__

void alc_gpu(double d, double g, double phi, int m, int n, double *X,
             double *Ki, int ncand, double *Xcand, int nref, double *Xref, 
             double *k, double *alcv, int gpu);
int num_gpus(void);

#endif

