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


#include "laGP.h"
#include "rhelp.h"
#include "matrix.h"
#include "gp.h"
#include "covar.h"
#include "order.h"
#include <assert.h>
#ifdef _GPU
  #include "alc_gpu.h"
#endif
#ifdef _OPENMP
  #include <omp.h>
#endif

/*
 * aGP_R:
 * 
 * R-interface to C-version of R function aGP.R: 
 * uses ALC to adaptively select a small (size n) subset 
 * of (X,Z) from which to predict by (thus a in approx) kriging 
 * equations at eack XX row; returns  mean and variance of 
 * those Student-t equations.
 */


void aGP_R(/* inputs */
		int *m_in,
		int *start_in,
		int *end_in,
		double *XX_in,
		int *nn_in,
		int *n_in,
		double *X_in,
		double *Z_in,
		double *d_in,
		double *darg_in,
		double *g_in,
		double *garg_in,
		int *imethod_in,
		int *close_in,
		int *ompthreads_in,
		int *numgpus_in,
		int *gputhreads_in,
		int *nngpu_in,
    int *numrays_in,
    double *rect_in,
		int *verb_in,
		int *Xiret_in,
		
		/* outputs */
		int *Xi_out,
		double *mean_out,
		double *var_out,
		double *dmle_out,
		int *dits_out,
		double *gmle_out,
		int *gits_out,
		double *llik_out)
{
  int j, verb, dmle, gmle, mxth;
  double **X, **XX, **rect;
  Method method;

#ifdef _OPENMP
  mxth = omp_get_max_threads();
#else
  mxth = 1;
#endif

  /* check gpu input */
#ifndef _GPU
  if(*numgpus_in || *nngpu_in) error("laGP not compiled with GPU support");
  if(*gputhreads_in != 0) 
    myprintf(mystdout, "NOTE: gpu.threads(%d) > 0 but GPUs not enabled\n");
#else
  int ngpu = num_gpus();
  if(*numgpus_in > ngpu) 
    error("%d GPUs requested, but %d available", *numgpus_in, ngpu);
  if(*nngpu_in < 0) error("must have nngpu (%d) >= 0", *nngpu_in);
  if(*nngpu_in < *numgpus_in)
    warning("number of GPUs (%d) greater than nngpu (%d)", *numgpus_in, *nngpu_in);
  if(*gputhreads_in == 0 && *numgpus_in > 0)
    error("requested 0 GPU threads but indicated %d GPUs", *numgpus_in);
  if(*gputhreads_in > 0 && *numgpus_in == 0)
    error("requested %d GPU threads but indicated 0 GPUs", *gputhreads_in);
  if(*gputhreads_in > mxth) {
    myprintf(mystdout, "NOTE: GPU threads(%d) > max(%d), using %d\n", 
      *gputhreads_in, mxth,   mxth);
    *gputhreads = mxth;
  }
  if(*nngpu_in < *nn_in && *ompthreads_in < 1)
    error("must have non-zero ompthreads (%d) when nngpu (%d) < nn (%d)", 
      *ompthreads_in, *nngpu_in, *nn_in);
#endif

  /* copy method */
  method = ALC; /* to guarentee initializaion */
  if(*imethod_in == 1) method = ALC;
  else if(*imethod_in == 2) method = ALCRAY;
  else if(*imethod_in == 3) method = MSPE;
  else if(*imethod_in == 4) method = EFI;
  else if(*imethod_in == 5) method = NN;
  else error("imethod %d does not correspond to a known method\n", *imethod_in);

  /* make matrix bones */
  X = new_matrix_bones(X_in, *n_in, *m_in);
  XX = new_matrix_bones(XX_in, *nn_in, *m_in);

  /* check rect input */
  if(method == ALCRAY) {
    assert(*numrays_in >= 1);
    rect = new_matrix_bones(rect_in, 2, *m_in);
  } else rect = NULL;

  /* check for mle */
  gmle = dmle = 0;
  if(darg_in[0] > 0) dmle = 1;
  if(garg_in[0] > 0) gmle = 1;

  /* for each predictive location */
#ifdef _OPENMP

  /* check verb argument; can't print too much in OpenMP */
  if(*verb_in > 1) {
    myprintf(mystdout, "NOTE: verb=%d but only verb=1 allowed with OpenMP\n", 
      *verb_in);
    verb = 1;
  } else verb = *verb_in;

  /* check ompthreads_in against max */
  if(*ompthreads_in > mxth) {
    myprintf(mystdout, "NOTE: omp.threads(%d) > max(%d), using %d\n", 
      *ompthreads_in, mxth, mxth);
    *ompthreads_in = mxth;
  }
  /* check combined threads */
  if(*ompthreads_in + *gputhreads_in > mxth) {
    myprintf(mystdout, "NOTE: combined GPU/OMP threads(%d) > max(%d), reducing OMP to %d\n", 
      *ompthreads_in + *gputhreads_in, mxth, mxth - *gputhreads_in);
    *ompthreads_in = mxth - *gputhreads_in;
  }

  /* ready to parallelize */
  #pragma omp parallel num_threads(*ompthreads_in + *gputhreads_in)
  {
    int i, me, start, step, end, gpu;

    /* get thread information */
    me = omp_get_thread_num();
    // nth = omp_get_num_threads();

    /* decide on gpu usage and range of XX locations for thread */
    if(me < *gputhreads_in) { 
      gpu = *numgpus_in; 
      start = me; 
      step = *gputhreads_in; 
      end = *nngpu_in;
    } else { 
      gpu = 0; 
      start = me - *gputhreads_in + *nngpu_in; 
      step = *ompthreads_in;
      end = *nn_in;
    }

#else
    int i, start, step, end, gpu;
    
    start = 0; step = 1; end = *nn_in; 
    verb = *verb_in;

    /* checking if threads arguments are reasonable */
    if(*ompthreads_in != 1) /* mxth should be 1; this is for compiler */
      warning("NOTE: omp.threads > %d, but source not compiled for OpenMP", mxth);
    if(*gputhreads_in > 1)
      warning("NOTE: using gpu.threads > 1 requires OpenMP compilation");

    /* gpu usage */
    gpu = *numgpus_in;
#endif

    /* initialization of data required for each thread */
    double ** Xref = new_matrix(1, *m_in);
    double dvec[6], gvec[6];
    for(j=1; j<6; j++) { dvec[j] = darg_in[j-1]; gvec[j] = garg_in[j-1]; }
    double df;
    double *dmlei, *gmlei;  dmlei = gmlei = NULL;
    int *ditsi, *gitsi, *Xi; ditsi = gitsi = Xi = NULL;

    /* loop over each predictive location */
    for(i=start; i<end; i+=step) {

      /* copy in predictive location; MAYBE move inside laGP */
      dupv(*Xref, XX[i], *m_in);
      dvec[0] = d_in[i]; gvec[0] = g_in[i];

      /* deal with mle and Xi pointers */
      if(dmle) { dmlei = &(dmle_out[i]); ditsi = &(dits_out[i]); }
      if(gmle) { gmlei = &(gmle_out[i]); gitsi = &(gits_out[i]); }
      if(*Xiret_in) { Xi = Xi_out+i*(*end_in); }

      /* call C-only code */
      laGP(*m_in, *start_in, *end_in, Xref, 1, *n_in, X, Z_in, dvec,
        gvec, method, *close_in, gpu, *numrays_in, rect, verb-1, Xi, &(mean_out[i]), 
        &(var_out[i]), &df, dmlei, ditsi, gmlei, gitsi, &(llik_out[i]));
      var_out[i] *= df/(df-2.0);

      /* progress meter */
#ifdef _OPENMP
      #pragma omp master
#endif
      if(verb > 0) {
        myprintf(mystdout, "i = %d (of %d)", i+1, end);
        if(dmle) myprintf(mystdout, ", d = %g, its = %d", *dmlei, *ditsi);
        if(gmle) myprintf(mystdout, ", g = %g, its = %d", *gmlei, *gitsi);
        myprintf(mystdout, "\n");
      }

      /* MAYBE ALLOW R INTERRUPTS HERE */
    }

    /* clean up */
    delete_matrix(Xref);

#ifdef _OPENMP
  }
#endif
    
  /* clean up */
  free(X);
  free(XX);
}


/*
 * closest_indices:
 *
 * returns the close indices into X which are closest (on average) to the
 * element(s) of Xref.  The first start of those indices are the start 
 * closest, otherwise the indecies unordered (unless sorted=true).  
 * Even when sorted=true the incies close+1, ... are not sorted.
 */

int *closest_indices(const unsigned int m, const unsigned int start,
  double **Xref, const unsigned int nref, const unsigned int n, double **X,
  const unsigned int close, const unsigned int sorted)
{
  int i;
  int *oD, *oD2, *temp;
  double **D;

  /* calculate distances to reference location(s), and so-order X & Z */
  D = new_matrix(nref, n);
  distance(Xref, nref, X, n, m, D);
  if(nref > 1) min_of_columns(*D, D, nref, n);

  /* partition based on "close"st */
  if(n > close) {
    oD = iseq(0, n-1);
    quick_select_index(*D, oD, n, close);
  } else oD = NULL;

  /* now partition based on start */
  if(sorted) {
    oD2 = order(*D, close);
    if(oD) {
      temp = new_ivector(close);
      for(i=0; i<close; i++) temp[i] = oD[oD2[i]];
      free(oD);
      oD = temp;
      free(oD2);  
    } else oD = oD2;
  } else {
    if(!oD) oD = iseq(0, n-1);
    quick_select_index(*D, oD, close, start);
  }

  delete_matrix(D);
  return(oD);
}

/*
 * laGP:
 * 
 * C-version of R function laGP.R: uses ALC to adaptively
 * select a small (size n) subset of (X,Z) from which to
 * predict by (thus approx) kriging equations; returns
 * those Student-t equations.
 */

void laGP(const unsigned int m, const unsigned int start, const unsigned int end, 
       double **Xref, const unsigned int nref, const unsigned int n, double **X, 
       double *Z, double *d, double *g, const Method method, const unsigned int close, 
       const int alc_gpu, const unsigned int numrays, double **rect, const int verb, 
       int *Xi, double *mean, double *s2, double *df, double *dmle, int *dits, 
       double *gmle, int *gits, double *llik)
{
  GP *gp;
  unsigned int i, j, ncand, w;
  int *oD, *cand;
  double **Xcand, **Xcand_orig, **x, **Sigma;
  double *al;

  /* temporary space */
  x = new_matrix(1, m);

  /* special cases for close */
  if(close > 0 && close < n-start) ncand = close-start;
  else ncand = n-start;

  /* get the indices of the closest X-locations to Xref */
  oD = closest_indices(m, start, Xref, nref, n, X, close, method == ALCRAY);

  /* build GP with closest start locations */
  gp = newGP_sub(m, start, oD, X, Z, *d, *g, method == MSPE || method == EFI);
  if(Xi) dupiv(Xi, oD, start);

  /* possibly restricted candidate set */
  cand = oD+start;
  Xcand_orig = Xcand = new_p_submatrix_rows(cand, X, ncand, m, 0);

  /* allocate space for active learning criteria */
  if(method != NN) al = new_vector(ncand);
  else al = NULL;

  /* loop over each adaptive sample of end */
  for(i=start; i<end; i++) {
    
    /* perform active learning calculations */
    if(method == ALCRAY) {
      assert(nref == 1);
      int roundrobin = (i-start+1) % ((int) sqrt(i-start+1.0));
      w = lalcrayGP(gp, Xcand, ncand, Xref, roundrobin, numrays, rect, verb-2);
    } else if(method == ALC) {
      if(alc_gpu) {
#ifdef _GPU
	#ifdef _OPENMP
	int gpu = omp_get_thread_num() % alc_gpu;
	#else
	int gpu = 0;
	#endif
        alcGP_gpu(gp, ncand, Xcand, nref, Xref, verb-2, al, gpu);
#else
        error("laGP not compiled for GPU support");
#endif
      } else alcGP(gp, ncand, Xcand, nref, Xref, verb-2, al);
    } else if(method == EFI) efiGP(gp, ncand, Xcand, al); /* no verb argument */
    else if(method == MSPE) mspeGP(gp, ncand, Xcand, nref, Xref, 1, verb-2, al);
    
    /* selecting from the evaluated criteria */
    if(method != ALCRAY) {
      if(method == NN) w = i-start;
      else if(method != MSPE) max(al, ncand, &w);
      else min(al, ncand, &w);
    }
    
    /* record chosen location */
    if(Xi != NULL) Xi[i] = cand[w];

    /* update the GP with the chosen candidate */
    dupv(x[0], Xcand[w], gp->m);
    updateGP(gp, 1, x, &(Z[cand[w]]), verb-1);

    /* remove from candidates */
    if(al && w != ncand-1) { 
      if(method == ALCRAY) { /* preserving distance order */
        if(w == 0) { cand++; Xcand++; }
        else {
          for(j=w; j<ncand-1; j++) { /* by pulling backwards */
            cand[j] = cand[j+1];
            dupv(Xcand[j], Xcand[j+1], gp->m);
          }
        }
      } else { /* simply swap: do not preserve distance order */
        cand[w] = cand[ncand-1]; 
        dupv(Xcand[w], Xcand[ncand-1], gp->m);
      }
    }
    ncand--;
  }

  /* possibly do MLE calculation */
  if(d[1] > 0 && g[1] > 0) {
    if(! gp->dK) newdKGP(gp);
    jmleGP(gp, d+2, g+2, d+4, g+4, dits, gits, verb);
    *dmle = gp->d; *gmle = gp->g;
  } else if(d[1] > 0) {
    if(! gp->dK) newdKGP(gp);
    *dmle = mleGP(gp, LENGTHSCALE, d[2], d[3], d+4, dits, verb);
  } else if(g[1] > 0) {
    *gmle = mleGP(gp, NUGGET, g[2], g[3], g+4, gits, verb);
  }

  /* now predict */
  Sigma = new_matrix(nref,nref);
  predGP(gp, nref, Xref, mean, Sigma, df, llik);
  /* could swap out for LITE version, though it doesn't matter when
     newf is small */
  for(i=0; i<nref; i++) s2[i] = Sigma[i][i];

  /* clean up */
  deleteGP(gp);
  delete_matrix(Xcand_orig);
  delete_matrix(Sigma);
  free(oD);
  if(al) free(al);
  delete_matrix(x);
}


/*
 * laGP_R:
 * 
 * R-interface to C-version of R function laGP.R: 
 * uses ALC to adaptively select a small (size n) subset 
 * of (X,Z) from which to predict by (thus approx) kriging 
 * equations at Xref; returns those Student-t equations.
 */


void laGP_R(/* inputs */
         int *m_in,
         int *start_in,
         int *end_in,
         double *Xref_in,
         int *nref_in,
         int *n_in,
         double *X_in,
         double *Z_in,
         double *d_in,
         double *g_in,
         int *imethod_in,
         int *close_in,
         int *alc_gpu_in,
         int *numrays_in,
         double *rect_in,
         int *verb_in,
         int *Xiret_in,
         
         /* outputs */
         int *Xi_out,
         double *mean_out,
         double *s2_out,
         double *df_out,
         double *dmle_out,
         int *dits_out,
         double *gmle_out,
         int *gits_out,
         double *llik_out)
{
  double **X, **Xref, **rect;
  Method method;

    /* check gpu input */
#ifndef _GPU
  if(*alc_gpu_in) error("laGP not compiled with GPU support");
#endif

  /* copy method */
  method = ALC; /* to guarentee initialization */
  if(*imethod_in == 1) method = ALC;
  else if(*imethod_in == 2) method = ALCRAY;
  else if(*imethod_in == 3) method = MSPE;
  else if(*imethod_in == 4) method = EFI;
  else if(*imethod_in == 5) method = NN;
  else error("imethod %d does not correspond to a known method\n", *imethod_in);

  /* sanity check d and tmax */
  if(d_in[1] > 0 && (d_in[0] > d_in[3] || d_in[0] < d_in[2])) 
    error("d=%g not in [tmin=%g, tmax=%g]\n", d_in[0], d_in[2], d_in[3]);


  /* make matrix bones */
  X = new_matrix_bones(X_in, *n_in, *m_in);
  Xref = new_matrix_bones(Xref_in, *nref_in, *m_in);

  /* check rect input */
  if(method == ALCRAY) {
    assert(*nref_in == 1);
    assert(*numrays_in >= 1);
    rect = new_matrix_bones(rect_in, 2, *m_in);
  } else rect = NULL;

  /* check Xi input */
  if(! *Xiret_in) Xi_out = NULL;

  /* call C-only code */
  laGP(*m_in, *start_in, *end_in, Xref, *nref_in, *n_in, X, Z_in,
    d_in, g_in, method, *close_in, *alc_gpu_in, *numrays_in, rect, *verb_in, Xi_out, 
    mean_out, s2_out, df_out, dmle_out, dits_out, gmle_out, gits_out, 
    llik_out);

  /* clean up */
  free(X);
  free(Xref);
  if(rect) free(rect);
}
         
