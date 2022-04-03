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


#include "laGP.h"
#include "laGP_sep.h"
#include "rhelp.h"
#include "matrix.h"
#include "gp_sep.h"
#include "covar.h"
#include "order.h"
#include <assert.h>
#include <stdlib.h>
#ifdef RPRINT
#include <Rmath.h>
#endif
#ifdef _OPENMP
  #include <omp.h>
#endif


/*
 * aGPsep_R:
 * 
 * R-interface to C-version of R function aGPsep.R: 
 * uses ALC to adaptively select a small (size n) subset 
 * of (X,Z) from which to predict by (thus a in approx) kriging 
 * equations at eack XX row; returns mean and variance of 
 * those Student-t equations.
 */

void aGPsep_R(/* inputs */
    int *m_in,
    int *start_in,
    int *end_in,
    double *XX_in,
    int *nn_in,
    int *n_in,
    double *X_in,
    double *Z_in,
    double *dstart_in,
    double *darg_in,
    double *g_in,
    double *garg_in,
    int *imethod_in,
    int *close_in,
    int *ompthreads_in,
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
  int j, verb, dmle, gmle, mxth, m;
  double **X, **XX, **rect, **dstart, **dmle_mat;
  Method method;

#ifdef _OPENMP
  mxth = omp_get_max_threads();
#else
  mxth = 1;
#endif

  /* copy method */
  method = ALC; /* to guarantee initializaion */
  if(*imethod_in == 1) method = ALC;
  else if(*imethod_in == 2) error("alcopt not supported for separable GPs at this time");
  else if(*imethod_in == 3) method = ALCRAY;
  else if(*imethod_in == 4) error("MSPE not supported for separable GPs at this time");
  else if(*imethod_in == 5) error("EFI not supported for separable GPs at this time");
  else if(*imethod_in == 6) method = NN;
  else error("imethod %d does not correspond to a known method\n", 
             *imethod_in);

  /* make matrix bones */
  m = *m_in;
  X = new_matrix_bones(X_in, *n_in, m);
  XX = new_matrix_bones(XX_in, *nn_in, m);
  dstart = new_matrix_bones(dstart_in, *nn_in, m);

  /* check rect input */
  if(method == ALCRAY) {
    assert(*numrays_in >= 1);
    rect = new_matrix_bones(rect_in, 2, m);
  } else rect = NULL;

  /* check for mle */
  gmle = dmle = 0; dmle_mat = NULL;
  if(darg_in[0] > 0) {
    dmle = 1;
    dmle_mat = new_matrix_bones(dmle_out, *nn_in, m);
  } 
  if(garg_in[0] > 0) gmle = 1; 

  /* for each predictive location */
#ifdef _OPENMP

  /* check verb argument; can't print too much in OpenMP */
  if(*verb_in > 1) {
    MYprintf(MYstdout, "NOTE: verb=%d but only verb=1 allowed with OpenMP\n", 
      *verb_in);
    verb = 1;
  } else verb = *verb_in;

  /* check ompthreads_in against max */
  if(*ompthreads_in > mxth) {
    MYprintf(MYstdout, "NOTE: omp.threads(%d) > max(%d), using %d\n", 
      *ompthreads_in, mxth, mxth);
    *ompthreads_in = mxth;
  }


  /* ready to parallelize */
  #pragma omp parallel num_threads(*ompthreads_in)
  {
    int i, start, step, end;

    /* get thread information */
    start = omp_get_thread_num();
    step = *ompthreads_in;
    end = *nn_in;

#else
    int i, start, step, end;
    
    start = 0; step = 1; end = *nn_in; 
    verb = *verb_in;

    /* checking if threads arguments are reasonable */
    if(*ompthreads_in != 1) /* mxth should be 1; this is for compiler */
      warning("NOTE: omp.threads > %d, but source not compiled for OpenMP", 
        mxth);
#endif

    /* initialization of data required for each thread */
    double ** Xref = new_matrix(1, m);
    double df;
    double *dmlei, *gmlei; dmlei = gmlei = NULL;
    int *gitsi, *Xi; gitsi = Xi = NULL;
    int ditsi[2];

    /* fill prior part of dvec and gvec */
    double gvec[6], *dvec;
    dvec = new_vector(3*m+3);
    dupv(dvec+m, darg_in, 2*m+3);
    dupv(gvec+1, garg_in, 5);

    /* loop over each predictive location */
    for(i=start; i<end; i+=step) {

      /* copy in predictive location; MAYBE move inside laGP */
      dupv(*Xref, XX[i], m);
      dupv(dvec, dstart[i], m);
      gvec[0] = g_in[i];

      /* deal with mle and Xi pointers */
      if(dmle) { dmlei = dmle_mat[i]; }
      if(gmle) { gmlei = &(gmle_out[i]); gitsi = &(gits_out[i]); }
      if(*Xiret_in) { Xi = Xi_out+i*(*end_in); }

      /* call C-only code */
      laGPsep(m, *start_in, *end_in, Xref, 1, *n_in, X, Z_in, dvec,
        gvec, method, *close_in, *numrays_in, rect, verb-1, Xi, 
        &(mean_out[i]), &(var_out[i]), 1, &df, dmlei, ditsi, gmlei, gitsi, 
        &(llik_out[i]), 0);

      /* copy outputs */
      if(dmle) dits_out[i] = ditsi[0];
      var_out[i] *= df/(df-2.0);

      /* progress meter */
#ifdef _OPENMP
      #pragma omp master
#endif
      if(verb > 0) {
        MYprintf(MYstdout, "i = %d (of %d)", i+1, end);
        if(dmle) {
          MYprintf(MYstdout, ", d = [%g", dmlei[0]);
          for(j=1; j<m; j++) MYprintf(MYstdout, ",%g", dmlei[j]); 
          MYprintf(MYstdout, "], its = %d", *ditsi);
        } if(gmle) MYprintf(MYstdout, ", g = %g, its = %d", *gmlei, *gitsi);
        MYprintf(MYstdout, "\n");
      }

      /* MAYBE ALLOW R INTERRUPTS HERE */
    }

    /* clean up */
    free(dvec);
    delete_matrix(Xref);

#ifdef _OPENMP
  }
#endif
    
  /* clean up */
  if(rect) free(rect);
  if(dmle_mat) free(dmle_mat);
  free(X);
  free(XX);
  free(dstart);
}


/*
 * laGPsep:
 * 
 * C-version of R function laGPsep.R: uses ALC to adaptively
 * select a small (size n) subset of (X,Z) from which to
 * predict by (thus approx) kriging equations; returns
 * those Student-t equations.
 */

void laGPsep(const unsigned int m, const unsigned int start, 
  const unsigned int end, double **Xref, const unsigned int nref, 
  const unsigned int n, double **X, double *Z, double *d, double *g, 
  const Method method, unsigned int close, const unsigned int numstart, 
  double **rect, const int verb, int *Xi, double *mean, double *s2, 
  const unsigned int lite, double *df, double *dmle, int *dits, double *gmle, 
  int *gits, double *llik, int fromR)
{
  GPsep *gpsep;
  unsigned int i, j, ncand, w, free_rect;
  int *oD, *cand;
  int dconv;
  char msg[60];
  double **Xcand, **Xcand_orig, **x, **Sigma;
  double *al;

  /* temporary space */
  x = new_matrix(1, m);

  /* special cases for close */
  if(method == NN && close > end) close = end;
  if(close > 0 && close < n-start) ncand = close-start;
  else ncand = n-start;

  /* get the indices of the closest X-locations to Xref */
  oD = closest_indices(m, start, Xref, nref, n, X, close, 
                       method == ALCRAY || method == ALCOPT);

  /* build separable GP with closest start locations */
  gpsep = newGPsep_sub(m, start, oD, X, Z, d, *g, 0);
  if(Xi) dupiv(Xi, oD, start);

  /* possibly restricted candidate set */
  cand = oD+start;
  Xcand_orig = Xcand = new_p_submatrix_rows(cand, X, ncand, m, 0);
  
  /* potentially calculate rect on the fly */
  if((method == ALCRAY || method == ALCOPT) && rect == NULL) {
    rect = get_data_rect(Xcand, ncand, m);
    free_rect = 1;
  } else free_rect = 0;

  /* allocate space for active learning criteria */
  if(method != NN) al = new_vector(ncand);
  else al = NULL;

  /* loop over each adaptive sample of end */
  for(i=start; i<end; i++) {
    
    /* perform active learning calculations */
    if(method == ALCRAY) {
      assert(nref == 1);
      int roundrobin = (i-start+1) % ((int) sqrt(i-start+1.0));
      w = lalcrayGPsep(gpsep, Xcand, ncand, Xref, roundrobin, numstart, rect, 
            verb-2);
    } else if(method == ALCOPT) {
      int roundrobin = (i-start); /* +1) % ((int) sqrt(i-start+1.0)); */
    w = lalcoptGPsep(gpsep, Xcand, ncand, Xref, nref, roundrobin, numstart, rect, 
                  100, verb-2, fromR);
    }else if(method == ALC)
    alcGPsep(gpsep, ncand, Xcand, nref, Xref, verb-2, al);
  
    /* selecting from the evaluated criteria */
    if(method != ALCRAY && method != ALCOPT) {
      if(method == NN) w = i-start;
      else if(method != MSPE) max(al, ncand, &w);
      else min(al, ncand, &w);
    }
    
    /* record chosen location */
    if(Xi != NULL) Xi[i] = cand[w];

    /* update the GP with the chosen candidate */
    dupv(x[0], Xcand[w], gpsep->m);
    updateGPsep(gpsep, 1, x, &(Z[cand[w]]), verb-1);

    /* remove from candidates */
    if(al && w != ncand-1) { 
      if(method == ALCRAY || method == ALCOPT) { /* preserving distance order */
        if(w == 0) { cand++; Xcand++; }
        else {
          for(j=w; j<ncand-1; j++) { /* by pulling backwards */
            cand[j] = cand[j+1];
            dupv(Xcand[j], Xcand[j+1], m);
          }
        }
      } else { /* simply swap: do not preserve distance order */
        cand[w] = cand[ncand-1]; 
        dupv(Xcand[w], Xcand[ncand-1], m);
      }
    }
    ncand--;
  }

  /* possibly do MLE calculation */
  if(d[m] > 0 && g[1] > 0) {
    if(! gpsep->dK) newdKGPsep(gpsep);
    jmleGPsep(gpsep, 100, d+m+1, d+2*m+1, g+2, d+3*m+1, g+4, verb, 
              dits, gits, &dconv, fromR);
    dupv(dmle, gpsep->d, m); 
    *gmle = gpsep->g;
  } else if(d[m] > 0) {
    if(! gpsep->dK) newdKGPsep(gpsep);
    mleGPsep(gpsep, d+m+1, d+2*m+1, d+3*m+1, 100, verb, dmle, dits, msg, 
             &dconv, fromR);
  } else if(g[1] > 0) {
    *gmle = mleGPsep_nug(gpsep, g[2], g[3], g+4, verb, gits);
  }

  /* now predict */
  if(lite) predGPsep_lite(gpsep, nref, Xref, 0, mean, s2, df, llik); 
  else {
    Sigma = new_matrix_bones(s2, nref, nref);
    predGPsep(gpsep, nref, Xref, 0, mean, Sigma, df, llik);
    free(Sigma); 
  }

  /* clean up */
  deleteGPsep(gpsep);
  delete_matrix(Xcand_orig);
  free(oD);
  if(al) free(al);
  if(free_rect) delete_matrix(rect);
  delete_matrix(x);
}


/*
 * laGPsep_R:
 * 
 * R-interface to C-version of R function laGPsep.R: 
 * uses ALC to adaptively select a small (size n) subset 
 * of (X,Z) from which to predict by (thus approx) kriging 
 * equations at Xref; returns those Student-t equations.
 */

void laGPsep_R(/* inputs */
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
         int *numstart_in,
         double *rect_in,
         int *lite_in,
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
  unsigned int j, m;
  int dits[2];
  Method method;

  /* copy method */
  method = ALC; /* to guarantee initialization */
  if(*imethod_in == 1) method = ALC;
  else if(*imethod_in == 2) method = ALCOPT;
  else if(*imethod_in == 3) method = ALCRAY;
  else if(*imethod_in == 4) error("MSPE not supported for separable GPs at this time");
  else if(*imethod_in == 5) error("EFI not supported for separable GPs at this time"); 
  else if(*imethod_in == 6) method = NN;
  else error("imethod %d does not correspond to a known method\n", 
             *imethod_in);

  /* sanity check d and tmax */
  m = *m_in;
  if(d_in[m] > 0)
    for(j=0; j<m; j++) 
      if(d_in[j] > d_in[2*m+1+j] || d_in[j] < d_in[m+1+j])
        error("d[%d]=%g not in [tmin=%g, tmax=%g]\n", 
          j, d_in[j], d_in[2*m+1+j], d_in[m+1+j]);

  /* make matrix bones */
  X = new_matrix_bones(X_in, *n_in, *m_in);
  Xref = new_matrix_bones(Xref_in, *nref_in, m);

  /* check rect input */
  if(method == ALCRAY || method == ALCOPT) {
    if(method == ALCRAY) assert(*nref_in == 1);
    assert(*numstart_in >= 1);
    if(rect_in[0] < rect_in[*m_in]) 
      rect = new_matrix_bones(rect_in, 2, *m_in);
    else rect = NULL;
  } else rect = NULL;

  /* check Xi input */
  if(! *Xiret_in) Xi_out = NULL;

  /* call C-only code */
  laGPsep(m, *start_in, *end_in, Xref, *nref_in, *n_in, X, Z_in,
    d_in, g_in, method, *close_in, *numstart_in, rect, *verb_in, Xi_out,
    mean_out, s2_out, *lite_in, df_out, dmle_out, dits, gmle_out, gits_out, 
    llik_out, 1);

  /* return first component of dits */
  *dits_out = dits[0];

  /* clean up */
  free(X);
  free(Xref);
  if(rect) free(rect);
}
         
