// module load cuda
// nvcc -arch=sm_20 -c -Xcompiler -fPIC alc_gpu.cu -o alc_gpu.o


#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
extern "C" {
  #include "alc_gpu.h"
}

// #define SUM2

// #define TIMINGS
#ifdef TIMINGS
#include <sys/time.h>
#endif

#define SDEPS sqrt(2.220446e-16)

/*
 * check_gpu_error:
 *
 * checking for errors after kernel calls 
 */

static void check_gpu_error(const char *msg) {
    cudaError_t err = cudaGetLastError ();
    if (cudaSuccess != err)
        printf("Cuda error: %s: %s\n", msg, cudaGetErrorString (err));
}

/*
 * NearestPowerOf2:
 *
 * finds the nearest power of 2 greater than n
 */

__device__ int NearestPowerOf2 (int n)
  {
    if (!n) return n;  //(0 == 2^0)
 
    int x = 1;
    while(x < n) { x <<= 1; }
  	
    return x;
}


/*
 * alc_kernel:
 *
 * return s2' component of the ALC calculation of the
 * expected reduction in variance calculation at locations 
 * Xcand averaging over reference locations Xref: 
 * ds2 = s2 - s2', where the s2s are at Xref and the
 * s2' incorporates Xcand, and everything is averaged
 * over Xref.
 */

__global__ void alc_kernel(const int n, const int m, double *X, double *Ki, 
         const double d, const double g, const double phi, const int ncand, 
         double *Xcand, const int nref, double *Xref, double *k, double *alc)
{
  unsigned tid, tid2, halfPoint, bid, bidm, nTotalThreads;
  double mui, kx_tid, gvec_tid, /* dot, ktGmuik, */ ktKikx_sum;

  extern __shared__ double s[];
  double* gvec = s; /* size n */
  double* kxy = &s[n];  /* size nref */
  double* kx = &s[n+nref];  /* size n */
  double* kx_gvec = &s[2*n+nref]; /* size n */
  double* Xc = &s[3*n+nref]; /* size m */
  double* k_g = kx; // &s[3*n+2*nref+m]; /* size n */
  double* ktGmui_k = kx_gvec; // &s[4*n+2*nref+m]; /* size n */

  /* thread indices stored in registers */
  tid = threadIdx.x; 
  bid = blockIdx.x;
  bidm = bid*m;

  /* copy row of Xcand into faster shared memory */
  if(tid < m) Xc[tid] = Xcand[bidm + tid];
  __syncthreads(); 
 
  /* calculate covariance between candidate and tid-th data point */
  /* each thread accesses global memory m times; might make sense to copy 
     like with Ki */
  kx_tid = 0.0;
  for(unsigned int k=0; k<m; k++)
     kx_tid += (Xc[k] - X[tid*m+k])*(Xc[k] - X[tid*m+k]);
  kx[tid] = exp(0.0 - kx_tid/d);
  __syncthreads();

  /* gvec calculation and dot product preparation */
  gvec_tid = 0;
  for(unsigned int i=0; i<n; i++)	gvec_tid += kx[i] * Ki[i*n+tid];
  kx_gvec[tid] = gvec_tid * kx[tid];
  __syncthreads();

  /* kx is no longer needed as of this point in the program;
     will later be used by k_g  */

  /* reduction for dot product */
  nTotalThreads = NearestPowerOf2(blockDim.x);
  while(nTotalThreads > 1) {
    halfPoint = (nTotalThreads >> 1);
    if(tid < halfPoint) {
      tid2 = tid + halfPoint;
      if(tid2 < blockDim.x) kx_gvec[tid] += kx_gvec[tid2];
    }
    __syncthreads();
    nTotalThreads = halfPoint;
  } 

  /* mui <- drop(1 + Zt$g - t(kx) %*% Kikx) */
  mui = 1.0 + g - kx_gvec[0]; 
  /* finish gvec calculation */
  gvec[tid] = 0.0 - gvec_tid/mui;
  // ktKikx_sum = 0.0;
  /* no need to sync threads */

  /* kx_gvec is no longer needed as of this point in the program;
     will later be used by ktGmui_k  */

  /* preparation of kxy: could thread for nref */
  if(tid == 0) {
     for(unsigned int i=0; i<nref; i++) {
       kx_tid = 0.0;  // re-using kx_tid from above 
       for(unsigned int k=0; k<m; k++)
          kx_tid += (Xc[k] - Xref[i*m+k])*(Xc[k] - Xref[i*m+k]);
       kxy[tid + i] = exp(0.0 - kx_tid/d);
     }
  }

  /* preparation of alc */
  if(tid == 1) alc[bid] = 0.0;
   __syncthreads();

 
  /* skip if numerical problems */
  if(mui > SDEPS) {
         
    /* use g, mu, and kxy to calculate ktKik.x */
    /* loop over all of the nref reference locations: 
       when nref is bigger we might want to thread this */
    for(unsigned int r=0; r<nref; r++) {

      /* ktGmui = t(k) %*% Gmui %*% k */
      double ktGmui_tid = 0;
      for(unsigned int j=0; j<n; j++)
        ktGmui_tid += k[r*n+j] * gvec[tid]*gvec[j]*mui;
      ktGmui_k[tid] = ktGmui_tid * k[r*n+tid];
      k_g[tid] = k[r*n+tid]*gvec[tid];
      __syncthreads();

      /* two reductions for ktGimuk and kg (re-using dot)  */
      nTotalThreads = NearestPowerOf2(blockDim.x);
#ifdef SUM2
      /* Option 1: each active thread does double work */
      while(nTotalThreads > 1) {
         halfPoint = (nTotalThreads >> 1);
         if(tid < halfPoint) {
           tid2 = tid + halfPoint;
           if(tid2 < blockDim.x) {
                 ktGmui_k[tid] += ktGmui_k[tid2];
                 k_g[tid] += k_g[tid2];
           }
         }
         __syncthreads();
         nTotalThreads = halfPoint;
       }
#else
       /* Option 2: the > n/2 idle threads pick up the second sum */
       /* first the threads to double-duty until power of 2 */
       halfPoint = (nTotalThreads >> 1);
       if(tid < halfPoint) {
          tid2 = tid + halfPoint;
          if(tid2 < blockDim.x) {
            ktGmui_k[tid] += ktGmui_k[tid2];
            k_g[tid] += k_g[tid2];
          }
        }
        __syncthreads();

        /* now its a power of to so we can do 2 parallel reductions */
        if(tid < halfPoint){
          if(tid < halfPoint/2) { // The first half of the threads work on ktGmui_k
            for(unsigned int s=halfPoint/2; s>0; s>>=1) {
              if(tid < s) ktGmui_k[tid] += ktGmui_k[tid + s];
              __syncthreads();
            } 
          } else { // The second half of the threads works on k_g
            tid = tid - (halfPoint/2);
            for(unsigned int s=halfPoint/2; s>0; s>>=1) {
              if (tid < s) k_g[tid] += k_g[tid + s];
              __syncthreads();
            }
          }
        }
#endif

      /* finish ktKikx calculation */
      if(tid == 0) ktKikx_sum /*+*/= ktGmui_k[0] + 2.0*k_g[0]*kxy[r] + kxy[r]*kxy[r]/mui;
      // __syncthreads();
    }
    
    /* calculate the ALC */
    /* when nref is bigger we might want to thread-reduce this */
    if(tid == 0) alc[bid] = phi*ktKikx_sum/((n-2.0)*((double) nref));
  }
}


extern "C" {

/*
 * num_gpus:
 *
 * a wrapper function to check how many gpus there are
 */

int num_gpus(void)  
  {
    int count;
    cudaError_t success = cudaGetDeviceCount(&count);
    if(success == cudaSuccess) return count;
    else return 0;
  }

/*
 * alcGP_gpu:
 *
 * calculate ALC stats on a GPU; for C-version and more comments 
 * see alcGP in gp.c 
 */

void alc_gpu(double d, double g, double phi, int m, int n, double *X,
             double *Ki, int ncand, double *Xcand, int nref, double *Xref, 
             double *k, double *alcv, int gpu)
  {
    double *d_k, *d_X, *d_Ki, *d_Xref, *d_Xcand, *d_alcv;

#ifdef TIMINGS
    struct timeval t1, t2;

    /* for timing memory copies */
    gettimeofday(&t1, 0);
#endif

    cudaError_t devsuccess = cudaSetDevice(gpu);
    assert(devsuccess == cudaSuccess);

    /* copy to GPU */
    cudaMalloc((void**) &d_X, (n*m) * sizeof(double));
    cudaMemcpy(d_X, X, (n*m) * sizeof(double), cudaMemcpyHostToDevice);
    check_gpu_error("X copy");
    cudaMalloc((void**) &d_Ki, (n*n) * sizeof(double));
    cudaMemcpy(d_Ki, Ki, (n*n) * sizeof(double), cudaMemcpyHostToDevice);
    check_gpu_error("Ki copy");
    cudaMalloc((void**) &d_Xref, (nref*m) * sizeof(double));
    cudaMemcpy(d_Xref, Xref, (nref*m) * sizeof(double), cudaMemcpyHostToDevice);
    check_gpu_error("Xref copy");
    cudaMalloc((void**) &d_Xcand, (ncand*m) * sizeof(double));
    cudaMemcpy(d_Xcand, Xcand, (ncand*m) * sizeof(double), cudaMemcpyHostToDevice);
    check_gpu_error("Xcand copy");
    cudaMalloc((void**) &d_k, (nref*n) * sizeof(double));
    cudaMemcpy(d_k, k, (nref*n) * sizeof(double), cudaMemcpyHostToDevice);
    check_gpu_error("k copy");
    /* allocate output on GPU */
    cudaMalloc((void**) &d_alcv, ncand * sizeof(double));
    // cudaMemset((void**) &d_alcv, 0, ncand * sizeof(double));
    check_gpu_error("alcv copy");

#ifdef TIMINGS
    /* finish timing memory copies */
    gettimeofday(&t2, 0);
    double time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000000.0;
    printf("GPU allocate/copy-in  time: %4.3fs\n", time); 
#endif

    /* run the kernel */
    int nBlocks = ncand; 
    dim3 dimBlock(n, 1, 1);
    dim3 dimGrid(nBlocks, 1, 1);

#ifdef TIMINGS
    /* begin timing of GPU calculation */
    gettimeofday(&t1, 0);
#endif

    /* begin GPU calculation */
    int sh_size = (3*n + nref + n*m)*sizeof(double);
    alc_kernel<<<dimGrid,dimBlock,sh_size>>>(n, m, d_X, d_Ki, d, g, phi, ncand, 
          d_Xcand, nref, d_Xref, d_k, d_alcv);
    cudaDeviceSynchronize();

    /* check for errors */
    check_gpu_error("alc_kernel");

#ifdef TIMINGS
    /* finish calculation timing */
    gettimeofday(&t2, 0);
    time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000000.0;
    printf("GPU compute time: %4.3fs\n", time); 

    /* begin timing of output copies */
    gettimeofday(&t1, 0);
#endif

    /* copy from GPU */
    cudaMemcpy(alcv, d_alcv, ncand * sizeof(double), cudaMemcpyDeviceToHost);

    /* clean up CUDA */
    cudaFree(d_X);
    cudaFree(d_alcv);
    cudaFree(d_Ki);
    cudaFree(d_Xref);
    cudaFree(d_Xcand);
    cudaFree(d_k);


#ifdef TIMINGS
    /* end timing of output copies */
    gettimeofday(&t2, 0);
    time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000000.0;
    printf("GPU free/copy-out time: %4.2fs\n", time); 
#endif
  }
}
