/******************************************************************************** 
 *
 * Bayesian Regression and Adaptive Sampling with Gaussian Process Trees
 * Copyright (C) 2005, University of California
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
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
 *
 * Questions? Contact Robert B. Gramacy (rbgramacy@ams.ucsc.edu)
 *
 ********************************************************************************/

#include "matrix.h"
#include "order.h"
#include "rhelp.h"
#include <stdlib.h>
#ifdef RPRINT
#include <Rmath.h>
#endif


/*
 * structure for ranking
 */

typedef struct rank
{
  double s;
  int r;
} Rank;



/*
 * compareRank:
 *
 * comparison function for ranking
 */

int compareRank(const void* a, const void* b)
{
  Rank* aa = (Rank*)(*(Rank **)a);
  Rank* bb = (Rank*)(*(Rank **)b); 
  if(aa->s < bb->s) return -1;
  else if(aa->s > bb->s) return 1;
  else return 0;
}


/*
 * order:
 *
 * obtain the integer order of the indices of s
 * from least to greatest.  the returned indices o
 * applied to s, (e.g. s[o]) would resort in a sorted list
 */

int* order(double *s, unsigned int n)
{
  int j;
  int *r;
  Rank ** sr;
  
  r = new_ivector(n);
  sr = (Rank**) malloc(sizeof(Rank*) * n);
  for(j=0; j<n; j++) {
    sr[j] = (Rank*) malloc(sizeof(Rank));
    sr[j]->s = s[j];
    sr[j]->r = j;
  }
  
  qsort((void*)sr, n, sizeof(Rank*), compareRank);
  
  /* assign ranks */
  for(j=0; j<n; j++) {
    r[j] = sr[j]->r;
    /* r[j] = sr[j]->r +1; */
    free(sr[j]);
  }
  free(sr);
  
  return r;
}


/*
 * rank:
 *
 * obtain the integer rank of the elemts of s
 */

int* rank(double *s, unsigned int n)
{
  int j;
  int *r;
  Rank ** sr;
  
  r = new_ivector(n);
  sr = (Rank**) malloc(sizeof(Rank*) * n);
  for(j=0; j<n; j++) {
    sr[j] = (Rank*) malloc(sizeof(Rank));
    sr[j]->s = s[j];
    sr[j]->r = j;
  }
  
  qsort((void*)sr, n, sizeof(Rank*), compareRank);
  
  /* assign ranks */
  for(j=0; j<n; j++) {
    /* r[sr[j]->r] = j+1; */
    r[sr[j]->r] = j;
    free(sr[j]);
  }
  free(sr);
  
  return r;
}


/*
 * rand_indices:
 *
 * return a random permutation of the
 * indices 1...N
 */

unsigned int* rand_indices(unsigned int N)
{
  int i;
  int *o;
  double *nall = new_vector(N);
  for(i=0; i<N; i++) nall[i] = unif_rand();
  o = order(nall, N);
  free(nall);
  return (unsigned int *) o;
}
