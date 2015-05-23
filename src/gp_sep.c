#include "matrix.h"
#include "gp_sep.h"
#include "util.h"
#include "linalg.h"
#include "rhelp.h"
#include <assert.h>
#include <stdlib.h>
#include <math.h>
#include <Rmath.h>
#include "covar_sep.h"
#include "ieci.h"

#define SDEPS sqrt(DOUBLE_EPS)

/*
 * Global variables used to accumulate data on the C-side
 * as passed from initialization and updtating R calls 
 */

unsigned int NGPsep = 0;
GPsep **gpseps = NULL;

/*
 * get_gpsep:
 *
 * returns an integer reference to a free separable gp 
 * 
 */

unsigned int get_gpsep(void)
{
  unsigned int i;
  if(NGPsep == 0) {
    assert(gpseps == NULL);
    gpseps = (GPsep**) malloc(sizeof(GPsep*));
    gpseps[0] = NULL;
    NGPsep = 1;
    return 0;
  } else {
    for(i=0; i<NGPsep; i++) {
      if(gpseps[i] == NULL) return i;
    }
    gpseps = (GPsep**) realloc(gpseps, sizeof(GPsep*) * (2*NGPsep));
    for(i=NGPsep; i<2*NGPsep; i++) gpseps[i] = NULL;
    NGPsep *= 2;
    return NGPsep/2;
  }
}


/*
 * deletedKGPsep:
 *
 * delete the dK components of the gp 
 *
 * SIMILAR to deleteGP but multiple dKs to delete
 */

void deletedKGPsep(GPsep *gpsep)
{
  unsigned int k;
  for(k=0; k<gpsep->m; k++) delete_matrix(gpsep->dK[k]);
  free(gpsep->dK);  
}

/* 
 * deleteGPsep:
 *
 * free the memory allocated to a separable gp structure 
 *
 * similar to deleteGP except loops over dK
 */


void deleteGPsep(GPsep* gpsep)
{
  assert(gpsep);
  assert(gpsep->X); delete_matrix(gpsep->X);
  assert(gpsep->Z); free(gpsep->Z);
  assert(gpsep->K); delete_matrix(gpsep->K);
  assert(gpsep->Ki); delete_matrix(gpsep->Ki);  
  assert(gpsep->KiZ); free(gpsep->KiZ);
  assert(gpsep->dK); 
  deletedKGPsep(gpsep);
  assert(gpsep->d); free(gpsep->d);
  free(gpsep);
}


/* 
 * deleteGPsep_index:
 *
 * delete the i-th gpsep
 * 
 * SAME as deleteGP except with gpsep
 */

void deleteGPsep_index(unsigned int i)
{
  if(!(gpseps == NULL || i >= NGPsep || gpseps[i] == NULL)) { 
    deleteGPsep(gpseps[i]);
    gpseps[i] = NULL;
  } else error("gpsep %d is not allocated\n", i);
}


/*
 * deleteGPsep_R:
 *
 * R-interface to deleteGPsep
 *
 * SAME as deleteGP_R except with gpsep
 */

void deleteGPsep_R(int *gpsep)
{
  deleteGPsep_index(*gpsep);
}


/*
 * deleteGPseps:
 *
 * delete all of the gpseps stored in
 * the array, and destroy the array
 *
 * SAME as deleteGPs except with gpsep
 */

void deleteGPseps(void)
{
  int i;
  for(i=0; i<NGPsep; i++) {
    if(gpseps[i]) {
      myprintf(mystdout, "removing gpsep %d\n", i);
      deleteGPsep(gpseps[i]);
    }
  }
  if(gpseps) free(gpseps);
  gpseps = NULL;
  NGPsep = 0;
}


/*
 * deleteGPseps_R:
 *
 * R interface to deleteGPseps
 *
 * SAME as deleteGPs_R except with gpsels
 */

void deleteGPseps_R(void)
{ 
  if(gpseps) deleteGPseps();
}


/*
 * calc_ZtKiZ_sep:
 *
 * re-calculates phi = ZtKiZ from Ki and Z stored in
 * the GP object; also update KiZ on which it depends
 *
 * SAME as gp.c but uses GPsep instead
 */

void calc_ZtKiZ_sep(GPsep *gpsep) 
{
  assert(gpsep);
  /* phi <- t(Z) %*% Ki %*% Z */
  if(gpsep->KiZ == NULL) gpsep->KiZ = new_vector(gpsep->n);
  linalg_dsymv(gpsep->n,1.0,gpsep->Ki,gpsep->n,gpsep->Z,1,0.0,gpsep->KiZ,1);
  gpsep->phi = linalg_ddot(gpsep->n, gpsep->Z, 1, gpsep->KiZ, 1);
}


/*
 * newdKGPsep:
 *
 * allocate new space for dK and d2K calculations, and 
 * cancluate derivatives 
 *
 * similar to newdKGP except no 2nd derivatives or fishinfo
 */

void newdKGPsep(GPsep *gpsep)
{
  unsigned int j;
  /* calculate derivatives */
  gpsep->dK = (double ***) malloc(sizeof(double **) * gpsep->m);
  for(j=0; j<gpsep->m; j++) gpsep->dK[j] = new_matrix(gpsep->n, gpsep->n);
  diff_covar_sep_symm(gpsep->m, gpsep->X, gpsep->n, gpsep->d, gpsep->K, gpsep->dK);
}

/*
 * buildGPsep:
 *
 * intended for newly created separable GPs, e.g., via newGPsep
 * does all of the correlation calculations, etc., after data and
 * parameters are defined 
 *
 * similar to buildGP except calculates gradient dK
 */

GPsep* buildGPsep(GPsep *gpsep)
{ 
  double **Kchol, **X;
  unsigned int n, m;
  int info;

  assert(gpsep && gpsep->K == NULL);
  n = gpsep->n;
  m = gpsep->m;
  X = gpsep->X;

  /* build covari ance matrix */
  gpsep->K = new_matrix(n, n);
  covar_sep_symm(m, X, n, gpsep->d, gpsep->g, gpsep->K);
  
  /* invert covariance matrix */
  gpsep->Ki = new_id_matrix(n);
  Kchol = new_dup_matrix(gpsep->K, n, n);
  info = linalg_dposv(n, Kchol, gpsep->Ki);
  if(info) {
#ifdef UNDEBUG
    printMatrix(gpsep->K, n, n, stdout);
#endif
    myprintf(mystdout, "d =");
    printVector(gpsep->d, m, mystdout, HUMAN);
    myprintf(mystdout, "\n");
    error("bad Cholesky decomp (info=%d), g=%g", 
          info, gpsep->g);
  }
  gpsep->ldetK = log_determinant_chol(Kchol, n);
  delete_matrix(Kchol);

  /* phi <- t(Z) %*% Ki %*% Z */
  gpsep->KiZ = NULL;
  calc_ZtKiZ_sep(gpsep);

  /* calculate derivatives */
  newdKGPsep(gpsep);

  /* return new structure */
  return(gpsep);
}


/*
 * newGPsep:
 *
 * allocate a new separable GP structure using the data and parameters
 * provided
 *
 * similar to  newGP except gpseps and pointer to d, and does not have dK
 * flag since gradient is always calculated
 */ 

GPsep* newGPsep(const unsigned int m, const unsigned int n, double **X,
	  double *Z, double *d, const double g)
{
  GPsep* gpsep;

  /* new gp structure */
  gpsep = (GPsep*) malloc(sizeof(GPsep));
  gpsep->m = m;
  gpsep->n = n;
  gpsep->X = new_dup_matrix(X, n, m);
  gpsep->Z = new_dup_vector(Z, n);
  gpsep->d = new_dup_vector(d, m);
  gpsep->g = g;
  gpsep->K = NULL;
  gpsep->dK = NULL;

  return buildGPsep(gpsep);
}


/*
 * newGPsep_R:
 *
 * R-interface initializing a new separable GP, allocating and
 * assigning values to the global variables, which are
 * written over if already in use
 *
 * similar to newGP_R except takes vector d_in and does not have dK flag
 * since the gradient is always calculated
 */

void newGPsep_R(/* inputs */
       int *m_in,
       int *n_in,
       double *X_in,
       double *Z_in,
       double *d_in,
       double *g_in,
       
       /* outputs */
       int *gpsep_index)
{
  double **X;

  /* assign a new gp index */
  *gpsep_index = get_gpsep();

  /* create a new GP; */
  X = new_matrix_bones(X_in, *n_in, *m_in);
  gpseps[*gpsep_index] = newGPsep(*m_in, *n_in, X, Z_in, d_in, *g_in);
  free(X);
}

/*
 * llikGPsep:
 *
 * calculate and return the log marginal likelihood
 *
 * similar to llikGPsep except loops over separable d for
 * prior calculation
 */

double llikGPsep(GPsep *gpsep, double *dab, double *gab)
{
  unsigned int k;
  double llik;

  /* proportional to likelihood calculation */
  llik = 0.0 - 0.5*(((double) gpsep->n) * log(0.5 * gpsep->phi) + gpsep->ldetK);
  // myprintf(mystdout, "d=%g, g=%g, phi=%g, llik=%g\n", gpsep->d, gpsep->g, gpsep->phi, llik); 
  /* llik += lgamma(0.5*((double) gpsep->n)) - ((double) gpsep->n)*M_LN_SQRT_2PI; */

  /* if priors are being used; for lengthscale */
  if(dab && dab[0] > 0 && dab[1] > 0) {
    for(k=0; k<gpsep->m; k++) {
      if(gpsep->d[k] > 0) llik += dgamma(gpsep->d[k], dab[0], 1.0/dab[1], 1);
    }
  }

  /* if priors are being used; for nugget */
  if(gpsep->g > 0 && gab && gab[0] > 0 && gab[1] > 0) 
    llik += dgamma(gpsep->g, gab[0], 1.0/gab[1], 1);

  return(llik);
}


/*
 * llikGPsep_R:
 *
 * R-interface to calculate the marginal likelihood of a 
 * separable GP
 *
 * SAME to llikGP except with gpsep
 */

void llikGPsep_R(/* inputs */
        int *gpsepi_in,
        double *dab_in,
        double *gab_in,

        /* outputs */
        double *llik_out)
{
  GPsep *gpsep;
  unsigned int gpsepi;

  /* get the cloud */
  gpsepi = *gpsepi_in;
  if(gpseps == NULL || gpsepi >= NGPsep || gpseps[gpsepi] == NULL) 
    error("gpsep %d is not allocated\n", gpsepi);
  gpsep = gpseps[gpsepi];

  /* calculate log likelihood */
  *llik_out = llikGPsep(gpsep, dab_in, gab_in);
}


/*
 * dllikGPsep:
 *
 * batch calculation of the gradient of the log likelihood 
 * of a separable gp, with respect to the 
 * lengthscale parameter, d; requires that derivatives
 * be pre-calculated
 *
 * substantially changed from dllikGP and removed d2llik
 */

void dllikGPsep(GPsep *gpsep, double *ab, double *dllik)
{
  double *KiZtwo;
  unsigned int i, j, n, k;
  double dn, phirat ;

  /* sanity check */
  assert(gpsep->dK);
  assert(dllik);

  /* copy dims for fast access */
  n = gpsep->n;
  dn = (double) n;

  KiZtwo = new_vector(n);
  for(k=0; k<gpsep->m; k++) {

    /* deal with possible prior */
    if(ab && ab[0] > 0 && ab[1] > 0) {
      dllik[k] = (ab[0] - 1.0)/gpsep->d[k] - ab[1];
    } else dllik[k] = 0.0;
  
    /* dllik = - 0.5 * tr(Ki %*% dK) */
    for(i=0; i<n; i++) {
      for(j=0; j<i; j++) /* off diagonal */
        dllik[k] -= gpsep->Ki[i][j] * gpsep->dK[k][i][j];

      /* on-diagonal */
      dllik[k] -= 0.5 * gpsep->Ki[i][i] * gpsep->dK[k][i][i];
    }

    /* now third part of the expression, re-using KiZtwo */
    /* KiZtwo = dK %*% KiZ */
    linalg_dsymv(n,1.0,gpsep->dK[k],n,gpsep->KiZ,1,0.0,KiZtwo,1);
    /* now t(KiZ) %*% dK %*% KiZ */
    phirat = linalg_ddot(n, gpsep->KiZ, 1, KiZtwo, 1) / gpsep->phi;
    dllik[k] += 0.5*dn*phirat;
  }
  
  /* clean up */
  free(KiZtwo);
}



/*
 * dllikGPsep_nug:
 *
 * batch calculation of the first derivative
 * of the log likelihood of a gp, with respect to the 
 * NUGGET parameter, d
 *
 * cut code from dllikGP_nug involving 2nd derivative
 */

void dllikGPsep_nug(GPsep *gpsep, double *ab, double *dllik, double *d2llik)
{
  unsigned int i, j, n;
  double *KiZtwo;
  double **two, **dKKidK;
  double dn, phirat, dlp, d2lp;

  /* sanity check */
  assert(dllik); 

  /* deal with possible prior */
  if(ab && ab[0] > 0 && ab[1] > 0) {
    dlp = (ab[0] - 1.0)/gpsep->g - ab[1];
    d2lp = 0.0 - (ab[0] - 1.0)/sq(gpsep->g);
  } else dlp = d2lp = 0.0;

  /* copy dims for fast access */
  n = gpsep->n;
  dn = (double) n;

  if(d2llik) {
    two = new_matrix(n, n);
    dKKidK = gpsep->Ki; 
  } else two = dKKidK = NULL;
  
  /* d2llik = - 0.5 * tr(Ki %*% [0.0 - Ki]); the first expression */
  /* dllik = - 0.5 * tr(Ki) */
  if(d2llik) *d2llik = d2lp;
  *dllik = dlp; 
  for(i=0; i<n; i++) {
    if(d2llik) {
      for(j=0; j<i; j++) { /* off diagonal */
        *d2llik += gpsep->Ki[i][j] * dKKidK[i][j];
        two[i][j] = two[j][i] = 2.0*dKKidK[i][j];
      }
    }
    /* on-diagonal */
    *dllik -= 0.5 * gpsep->Ki[i][i];
    if(d2llik) {
      *d2llik += 0.5 * gpsep->Ki[i][i] * dKKidK[i][i];
      two[i][i] = 2.0*dKKidK[i][i];
    }
  }

  /* now the second part of the expression: */
  /* d2llik -= 0.5 * KiZ %*% two %*% KiZ */
  if(d2llik) {
    KiZtwo = new_vector(n);
    linalg_dsymv(n,1.0,two,n,gpsep->KiZ,1,0.0,KiZtwo,1);
    *d2llik -= 0.5*dn*linalg_ddot(n, gpsep->KiZ, 1, KiZtwo, 1) / gpsep->phi;
    free(KiZtwo);
  }

  /* now third part of the expression, re-using KiZtwo */
  /* now t(KiZ) %*% dK %*% KiZ */
  phirat = linalg_ddot(n, gpsep->KiZ, 1, gpsep->KiZ, 1) / gpsep->phi;
  if(d2llik) *d2llik += 0.5*dn*sq(phirat);
  *dllik += 0.5*dn*phirat;

  /* clean up */
  if(two) delete_matrix(two);
}


/*
 * dllikGPsep_R:
 *
 * R-interface to calculate the derivatives of the
 * likelihood of a GP - wrt lengthscale
 *
 * SIMILAR to dllikGP_R except returns vectorized dllik_out
 * and does not have d2llik_out
 */

void dllikGPsep_R(/* inputs */
               int *gpsepi_in,
               double *ab_in,

               /* outputs */
               double *dllik_out)
{
  GPsep *gpsep;
  unsigned int gpsepi;

  /* get the cloud */
  gpsepi = *gpsepi_in;
  if(gpseps == NULL || gpsepi >= NGPsep || gpseps[gpsepi] == NULL)
    error("gpsep %d is not allocated\n", gpsepi);
  gpsep = gpseps[gpsepi];

  /* double check that derivatives have been calculated */
  if(! gpsep->dK) error("derivative info not in gp; use newGP with dK=TRUE");

  /* calculate log likelihood */
  dllikGPsep(gpsep, ab_in, dllik_out);
}


/*
 * dllikGPsep_nug_R:
 *
 * R-interface to calculate the derivatives of the
 * likelihood of a GP - wrt the NUGGET
 *
 * SIMILAR to dllikGP_nug_R except without d2llik_out
 */

void dllikGPsep_nug_R(/* inputs */
               int *gpsepi_in,
               double *ab_in,

               /* outputs */
               double *dllik_out,
               double *d2llik_out)
{
  GPsep *gpsep;
  double *d2llik;
  unsigned int gpsepi;

  /* get the cloud */
  gpsepi = *gpsepi_in;
  if(gpseps == NULL || gpsepi >= NGPsep || gpseps[gpsepi] == NULL)
    error("gpsep %d is not allocated\n", gpsepi);
  gpsep = gpseps[gpsepi];


  /* check to see if we want 2nd derivative or not */
  if(d2llik_out[0] == 1) d2llik = d2llik_out;
  else d2llik = NULL;

  /* calculate log likelihood */
  dllikGPsep_nug(gpsep, ab_in, dllik_out, d2llik);
}


/*
 * getmGPsep_R:
 *
 * R-interface accessing the input dimension m
 * of a GP
 */

void getmGPsep_R(/* inputs */
              int *gpsepi_in,
               /* outputs */
              int *m_out)
{
  GPsep *gpsep;
  unsigned int gpsepi;

  /* get the cloud */
  gpsepi = *gpsepi_in;
  if(gpseps == NULL || gpsepi >= NGPsep || gpseps[gpsepi] == NULL)
    error("gpsep %d is not allocated\n", gpsepi);
  gpsep = gpseps[gpsepi];

  *m_out = gpsep->m;
}


/*
 * getgGPsep_R:
 *
 * R-interface accessing the snugget of a separable
 * of a GP
 */


void getgGPsep_R(/* inputs */
              int *gpsepi_in,
               /* outputs */
              double *g_out)
{
  GPsep *gpsep;
  unsigned int gpsepi;

  /* get the cloud */
  gpsepi = *gpsepi_in;
  if(gpseps == NULL || gpsepi >= NGPsep || gpseps[gpsepi] == NULL)
    error("gpsep %d is not allocated\n", gpsepi);
  gpsep = gpseps[gpsepi];

  *g_out = gpsep->g;
}

/*
 * getdGPsep_R:
 *
 * R-interface accessing the separable lengthscale parameter
 * of a GP
 */

void getdGPsep_R(/* inputs */
              int *gpsepi_in,
               /* outputs */
              double *d_out)
{
  GPsep *gpsep;
  unsigned int gpsepi;

  /* get the cloud */
  gpsepi = *gpsepi_in;
  if(gpseps == NULL || gpsepi >= NGPsep || gpseps[gpsepi] == NULL)
    error("gpsep %d is not allocated\n", gpsepi);
  gpsep = gpseps[gpsepi];

  /* double check that derivatives have been calculated */
  dupv(d_out, gpsep->d, gpsep->m);
}


/*
 * newparamsGPsep:
 *
 * change the lengthscale and nugget parameters to the gp
 *
 * SIMIAR to newparamsGP except vectorized d and always does
 * gradient
 */ 

void newparamsGPsep(GPsep* gpsep, double *d, const double g)
{
  int info, m, n;
  double **Kchol;

  /* sanity check */
  assert(g >= 0);

  /* build covariance matrix */
  m = gpsep->m; n = gpsep->n;
  dupv(gpsep->d, d, m);
  gpsep->g = g;
  covar_sep_symm(m, gpsep->X, n, gpsep->d, gpsep->g, gpsep->K);
  
  /* invert covariance matrix */
  id(gpsep->Ki, n);
  Kchol = new_dup_matrix(gpsep->K, n, n);
  info = linalg_dposv(n, Kchol, gpsep->Ki);
  if(info) {
#ifdef UNDEBUG
    printMatrix(gpsep->K, n, n, stdout);
#endif
    myprintf(mystdout, "d =");
    printVector(gpsep->d, m, mystdout, HUMAN);
    myprintf(mystdout, "\n");
    error("bad Cholesky decomp (info=%d), g=%g", info, g);
  }
  gpsep->ldetK = log_determinant_chol(Kchol, n);
  delete_matrix(Kchol);

  /* phi <- t(Z) %*% Ki %*% Z */
  calc_ZtKiZ_sep(gpsep);

  /* calculate derivatives and Fisher info based on them ? */
  diff_covar_sep_symm(gpsep->m, gpsep->X, gpsep->n, gpsep->d, gpsep->K, gpsep->dK);    
}


/*
 * newparamsGPsep_R:
 *
 * R-interface allowing the internal/global separable GP representation
 * to change its parameterization without destroying the
 * memory and then re-allocating it
 */

void newparamsGPsep_R(/* inputs */
    int *gpsepi_in,
    double *d_in,
    double *g_in)
{
  GPsep *gpsep;
  unsigned int gpsepi, k;
  int dsame;

  /* get the cloud */
  gpsepi = *gpsepi_in;
  if(gpseps == NULL || gpsepi >= NGPsep || gpseps[gpsepi] == NULL) 
    error("gpsep %d is not allocated\n", gpsepi);
  gpsep = gpseps[gpsepi];

  /* check if any are old */
  dsame = 1;
  for(k=0; k<gpsep->m; k++) {
    if(d_in[k] <= 0) d_in[k] = gpsep->d[k];
    else if(d_in[k] != gpsep->d[k]) dsame = 0;
  }
  if(*g_in < 0) *g_in = gpsep->g;

  /* check if there is nothing to do bc the params are the same */
  if(dsame && *g_in == gpsep->g) return;

  /* call real C routine */
  newparamsGPsep(gpsep, d_in, *g_in);
}



/*
 * utility structure for fcnnllik and defined below
 * for use with Brent_fmin (R's optimize) or uniroot
 *
 * SIMPLIFIED compared to callinfo in gp.c because it only does the nugget
 */

struct callinfo_sep_nug {
  Theta theta;
  GPsep *gpsep;
  double *ab;
  int its;
  int verb;
};


/*
 * fcnnllik_sep_nug:
 * 
 * a utility function for Brent_fmin (R's optimize) to apply to the GP
 * log likelihood after changes to the lengthscale or nugget parameter 
 *
 * SIMPLIFIED compared to fcnnllik in gp.c since it only does the nugget
 */

static double fcnnllik_sep_nug(double x, struct callinfo_sep_nug *info)
{
  double llik;
  (info->its)++;
  newparamsGPsep(info->gpsep, info->gpsep->d, x);
  llik = llikGPsep(info->gpsep, NULL, info->ab);
  if(info->verb > 1)
    myprintf(mystdout, "fmin it=%d, g=%g, llik=%g\n", info->its, info->gpsep->g, llik);
  return 0.0-llik;
} 


/*
 * Ropt_sep_nug:
 *
 * use R's Brent Fmin routine (from optimize) to optimize
 *
 * SIMPLIFIED compared to Ropt in GP because it only does the nugget
 */

double Ropt_sep_nug(GPsep* gpsep, double tmin, double tmax, 
                   double *ab, char *msg, int *its, int verb)
{
  double tnew, th;
  // double ax, bx, fa, fb;
  double Tol = SDEPS;
  // int Maxit = 100;

  /* sanity check */
  assert(tmin < tmax);

  /* get parameter */
  th = gpsep->g;

  /* create structure for Brent_fmin */
  struct callinfo_sep_nug info;
  info.gpsep = gpsep;
  info.ab = ab;
  info.its = 0;
  info.verb = verb;

  /* call the C-routine behind R's optimize function */
  while(1) { /* check to make sure solution is not on boundary */
   tnew = Brent_fmin(tmin, tmax, (double (*)(double, void*)) fcnnllik_sep_nug, &info, Tol);  
   if(tnew > tmin && tnew < tmax) break;
   if(tnew == tmin) { /* left boundary found */
    tmin *= 2;
    if(verb > 0) myprintf(mystdout, "Ropt: tnew=tmin, increasing tmin=%g\n", tmin);
   } else { /* right boundary found */
    tmax /= 2.0;
    if(verb > 0) myprintf(mystdout, "Ropt: tnew=tmax, decreasing tmax=%g\n", tmax);
  }
  /* check that boundaries still valid */
  if(tmin >= tmax) error("unable to opimize in fmin()");
  } 

  /* check that last value agrees with GP parameterization */
  if(gpsep->g != tnew) newparamsGPsep(gpsep, gpsep->d, tnew);

  /* possible print message and return */
  if(verb > 0) myprintf(mystdout, "Ropt %s: told=%g -[%d]-> tnew=%g\n",
      msg, th, info.its, tnew);

  *its += info.its;
  return(tnew);
}


/*
 * mleGPsep_nug:
 *
 * calculate the MLE with respect to the lengthscale parameter;
 * derivatives for the Newton method are calculated on the fly
 */

double mleGPsep_nug(GPsep* gpsep, double tmin, double tmax, double *ab, 
             int *its, int verb)
{
  double tnew, dllik, d2llik, llik_init, llik_new, adj, rat;
  double th;
  double *gab, *dab;
  int restoredKGP;

  /* set priors based on Theta */
  dab = NULL;
  gab = ab;

  /* initialization */
  *its = 0;
  restoredKGP = 0;
  th = gpsep->g;

  /* check how close we are to tmin */
  if(fabs(th - tmin) < SDEPS) {
    if(verb > 0) myprintf(mystdout, "(g=%g) -- starting too close to min (%g)\n", th, tmin);
    goto alldone;
  }

  /* initial likelihood calculation */
  llik_init = llikGPsep(gpsep, dab, gab);

  /* initial printing */
  if(verb > 0) 
      myprintf(mystdout, "(g=%g, llik=%g) ", gpsep->g, llik_init);
  if(verb > 1) myprintf(mystdout, "\n");

  while(1) { /* checking for improved llik */
    while(1) {  /* Newton step(s) */
      llik_new = 0.0-1e300*1e300;
      while(1) {  /* Newton proposal */

        /* calculate first and second derivatives */
        dllikGPsep_nug(gpsep, gab, &dllik, &d2llik);

        /* check for convergence by root */
        if(fabs(dllik) < SDEPS) {
          if(*its == 0) {
            if(verb > 0) myprintf(mystdout, "-- Newton not needed\n");
            goto alldone;
          } else goto newtondone;
        }

        /* Newton update */
        rat = dllik/d2llik; adj = 1.0; (*its)++;

        /* check if we're going the right way */
        if((dllik < 0 && rat < 0) || (dllik > 0 && rat > 0)) {
          if(!gpsep->dK && restoredKGP == 0) { deletedKGPsep(gpsep); restoredKGP = 1; }
          th = Ropt_sep_nug(gpsep, tmin, tmax, ab, "[slip]", its, verb); goto mledone; 
        } else tnew = th - adj*rat;  /* right way: Newton: */

        /* check that we haven't proposed a tnew out of range */
        while((tnew <= tmin || tnew >= tmax) && adj > SDEPS) {
          adj /= 2.0; tnew = th - adj*rat;
        }

        /* if still out of range, restart? */
        if(tnew <= tmin || tnew >= tmax) { 
          if(!gpsep->dK && restoredKGP == 0) { deletedKGPsep(gpsep); restoredKGP = 1; }
          th = Ropt_sep_nug(gpsep, tmin, tmax, ab, "[range]", its, verb);
          goto mledone;
        } else break;
      } /* end inner while -- Newton proposal */

      /* else, resets gpsep->g = tnew */
      if(!gpsep->dK && restoredKGP == 0) { deletedKGPsep(gpsep); restoredKGP = 1; }
      newparamsGPsep(gpsep, gpsep->d, tnew);

      /* print progress */
      if(verb > 1) myprintf(mystdout, "\ti=%d g=%g, c(a,b)=(%g,%g)\n", 
                            *its, tnew, ab[0], ab[1]);      

      /* check for convergence, and break or update */
      if(fabs(tnew - th) < SDEPS) break;
      else th = tnew;

      /* check for max its */
      if(*its >= 100) {
        if(verb > 0) warning("Newton 100/max iterations");
        /* could also call Ropt here as last resort */
       goto alldone;
      }
    } /* end middle while -- Newton step */

    /* sanity check check that we went in the right direction */
newtondone:
    llik_new = llikGPsep(gpsep, dab, gab);
    if(llik_new < llik_init-SDEPS) { 
      if(verb > 0) myprintf(mystdout, "llik_new = %g\n", llik_new);
      llik_new = 0.0-1e300*1e300;
      if(!gpsep->dK && restoredKGP == 0) { deletedKGPsep(gpsep); restoredKGP = 1; }
      th = Ropt_sep_nug(gpsep, tmin, tmax, ab, "[dir]", its, verb); 
      goto mledone;
    } else break;
  } /* outer improved llik check while(1) loop */

  /* capstone progress indicator */
mledone:
  if(!R_FINITE(llik_new)) llik_new = llikGPsep(gpsep, dab, gab);
  if(verb > 0) {
    myprintf(mystdout, "-> %d Newtons -> (g=%g, llik=%g)\n", 
            *its, gpsep->g, llik_new);
  }

  /* return theta-value found */
alldone:
  if(restoredKGP) newdKGPsep(gpsep);
  return th;
}


/*
 * mleGPsep_nug_R:
 *
 * R-interface to update the separable GP to use its MLE 
 * nugget parameterization using the current data
 *
 * SIMPLIFIED compared to mleGPsep_R since there is
 * no lengthscale option
 */

void mleGPsep_nug_R(/* inputs */
       int *gpsepi_in,
       int *verb_in,
       double *tmin_in,
       double *tmax_in,
       double *ab_in,

       /* outputs */
       double *mle_out,
       int *its_out)
{
  GPsep *gpsep;
  unsigned int gpsepi;

  /* get the cloud */
  gpsepi = *gpsepi_in;
  if(gpseps == NULL || gpsepi >= NGPsep || gpseps[gpsepi] == NULL) 
    error("gpsep %d is not allocated\n", gpsepi);
  gpsep = gpseps[gpsepi];

  /* check theta and tmax */
  if(*tmin_in <= 0) *tmin_in = SDEPS; 
  if(gpsep->g >= *tmax_in) error("g=%g >= tmax=%g\n", gpsep->g, *tmax_in);
  else if(gpsep->g <= *tmin_in) error("g=%g <= tmin=%g\n", gpsep->g, *tmin_in);

  /* check a & b */
  if(ab_in[0] < 0 || ab_in[1] < 0) error("ab_in must be a positive 2-vector");

  /* call C-side MLE */
  *mle_out = mleGPsep_nug(gpsep, *tmin_in, *tmax_in, ab_in, its_out, *verb_in);
}


/*
 * updateGPsep:
 *
 * quickly augment (O(n^2)) a gp based on new X-Z pairs.  
 * Uses the Bartlet partition inverse equations
 */

void updateGPsep(GPsep* gpsep, unsigned int nn, double **XX, double *ZZ, 
                 int verb)
{
  unsigned int i, j, l, n, m;
  double *kx, *x, *gvec;
  double mui, Ztg;
  double **Gmui, **temp;
  
  /* allocate space */
  n = gpsep->n; m = gpsep->m;
  kx = new_vector(n);
  gvec = new_vector(n);
  Gmui = new_matrix(n, n);
  temp = new_matrix(1, 1);

  /* for each new location */
  for(j=0; j<nn; j++) {

    /* shorthand for x being updated */
    x = XX[j];

    /* calculate the Bartlet quantities */
    calc_g_mui_kxy_sep(m, x, gpsep->X, n, gpsep->Ki, NULL, 0, gpsep->d, 
                       gpsep->g, gvec, &mui, kx, NULL);

    /* Gmui = g %*% t(g)/mu */
    linalg_dgemm(CblasNoTrans,CblasTrans,n,n,1,
     mui,&gvec,n,&gvec,n,0.0,Gmui,n);
    
    /* Ki = Ki + Gmui */
    linalg_daxpy(n*n, 1.0, *Gmui, 1, *(gpsep->Ki), 1);
    
    /* now augment covariance matrices */
    /* for nn > 1 might be better to make bigger matricies once
       outside the for-loop */
    gpsep->Ki = new_bigger_matrix(gpsep->Ki, n, n, n+1, n+1);
    for(i=0; i<n; i++) gpsep->Ki[n][i] = gpsep->Ki[i][n] = gvec[i];
    gpsep->Ki[n][n] = 1.0/mui;
    gpsep->K = new_bigger_matrix(gpsep->K, n, n, n+1, n+1);
    for(i=0; i<n; i++) gpsep->K[n][i] = gpsep->K[i][n] = kx[i];
    covar_sep_symm(m, &x, 1, gpsep->d, gpsep->g, temp);
    gpsep->K[n][n] = **temp;

    /* update the determinant calculation */
    gpsep->ldetK += log(**temp + mui * linalg_ddot(n, gvec, 1, kx, 1));

    /* update KiZ and phi */
    /* Ztg = t(Z) %*% gvec */
    Ztg = linalg_ddot(n, gvec, 1, gpsep->Z, 1);
    gpsep->KiZ = realloc(gpsep->KiZ, sizeof(double)*(n+1));
    /* KiZ[1:n] += (Ztg/mu + Z*g) * gvec */
    linalg_daxpy(n, Ztg*mui + ZZ[j], gvec, 1, gpsep->KiZ, 1);
    /* KiZ[n+1] = Ztg + z*mu */
    gpsep->KiZ[n] = Ztg + ZZ[j]/mui;
    /* phi += Ztg^2/mu + 2*z*Ztg + z^2*mu */
    gpsep->phi += sq(Ztg)*mui + 2.0*ZZ[j]*Ztg + sq(ZZ[j])/mui;    

    /* now augment X and Z */
    gpsep->X = new_bigger_matrix(gpsep->X, n, m, n+1, m);
    dupv(gpsep->X[n], x, m);
    gpsep->Z = (double*) realloc(gpsep->Z, sizeof(double)*(n+1));
    gpsep->Z[n] = ZZ[j];
    (gpsep->n)++;

    /* augment derivative covariance matrices */
    for(l=0; l<m; l++) gpsep->dK[l] = new_bigger_matrix(gpsep->dK[l], n, n, n+1, n+1);
    double ***dKn = (double***) malloc(sizeof(double **) * m);
    for(l=0; l<m; l++) dKn[l] = new_matrix(1, n);
    diff_covar_sep(m, &x, 1, gpsep->X, n, gpsep->d, &(gpsep->K[n]), dKn);
    for(l=0; l<m; l++) {
      for(i=0; i<n; i++) gpsep->dK[l][i][n] = gpsep->dK[l][n][i] = dKn[l][0][i];
      delete_matrix(dKn[l]);
    }
    free(dKn);
    for(l=0; l<m; l++) gpsep->dK[l][n][n] = 0.0;

    /* if more then re-allocate */
    if(j < nn-1) {
      kx = (double*) realloc(kx, sizeof(double)*(n+1));
      gvec = (double*) realloc(gvec, sizeof(double)*(n+1));
      Gmui = new_bigger_matrix(Gmui, n, n, n+1, n+1);
    }

    /* progress meter? */
    if(verb > 0)
      myprintf(mystdout, "update_sep j=%d, n=%d, ldetK=%g\n", j+1, gpsep->n, gpsep->ldetK);
    n = gpsep->n; /* increment for next interation */
  }

  /* clean up */
  delete_matrix(Gmui);
  free(kx);
  free(gvec);
  delete_matrix(temp);
}


/*
 * updateGPsep_R:
 *
 * R-interface allowing the internal/global GP representation
 * to be quickly augment (O(n^2)) based on new X-Z pairs.  
 * Uses the Bartlet partition inverse equations
 */

void updateGPsep_R(/* inputs */
    int *gpsepi_in,
    int *m_in,
    int *nn_in,
    double *XX_in,
    double *ZZ_in,
    int *verb_in)
{
  GPsep *gpsep;
  unsigned int gpsepi;
  double **XX;

  /* get the cloud */
  gpsepi = *gpsepi_in;
  if(gpseps == NULL || gpsepi >= NGPsep || gpseps[gpsepi] == NULL) 
    error("gpsep %d is not allocated\n", gpsepi);
  gpsep = gpseps[gpsepi];
  if((unsigned) *m_in != gpsep->m)  
    error("ncol(X)=%d does not match GPsep/C-side (%d)", *m_in, gpsep->m);

  /* check that this is not a degenerate GP: not implemented (yet) */
  if(gpsep->d[0] <= 0) error("updating degenerate GP (d=0) not supported");

  /* sanity check and XX representation */
  XX = new_matrix_bones(XX_in, *nn_in, gpsep->m);

  /* call real C routine */
  updateGPsep(gpsep, *nn_in, XX, ZZ_in, *verb_in);

  /* clean up */
  free(XX);
}


/*
 * predGPsep:
 *
 * return the student-t predictive equations,
 * i.e., parameters to a multivatiate t-distribution
 * for XX predictive locations of dimension (n*m)
 */

void predGPsep(GPsep* gpsep, unsigned int nn, double **XX, double *mean, 
      double **Sigma, double *df, double *llik)
{
  unsigned int i, j, m, n;
  double **k, **ktKi, **ktKik;
  double phidf;

  /* easier referencing for dims */
  n = gpsep->n;  m = gpsep->m;

  /* variance (s2) components */
  *df = (double) n; 
  phidf = gpsep->phi/(*df);

  /* calculate marginal likelihood (since we have the bits) */
  *llik = 0.0 - 0.5*((*df) * log(0.5 * gpsep->phi) + gpsep->ldetK);
  /* continuing: - ((double) n)*M_LN_SQRT_2PI;*/

  /* k <- covar(X1=X, X2=XX, d=Zt$d, g=0) */
  k = new_matrix(n, nn);
  covar_sep(m, gpsep->X, n, XX, nn, gpsep->d, 0.0, k);
  /* Sigma <- covar(X1=XX, d=Zt$d, g=Zt$g) */
  covar_sep_symm(m, XX, nn, gpsep->d, gpsep->g, Sigma);
  
  /* ktKi <- t(k) %*% util$Ki */
  ktKi = new_matrix(n, nn);
  linalg_dsymm(CblasRight,nn,n,1.0,gpsep->Ki,n,k,nn,0.0,ktKi,nn);
  /* ktKik <- ktKi %*% k */
  ktKik = new_matrix(nn, nn);
  linalg_dgemm(CblasNoTrans,CblasTrans,nn,nn,n,
               1.0,k,nn,ktKi,nn,0.0,ktKik,nn);

  /* mean <- ktKi %*% Z */
  linalg_dgemv(CblasNoTrans,nn,n,1.0,ktKi,nn,gpsep->Z,1,0.0,mean,1);

  /* Sigma <- phi*(Sigma - ktKik)/df */
  for(i=0; i<nn; i++) {
     Sigma[i][i] = phidf * (Sigma[i][i] - ktKik[i][i]);
    for(j=0; j<i; j++)
      Sigma[j][i] = Sigma[i][j] = phidf * (Sigma[i][j] - ktKik[i][j]);
  }

  /* clean up */
  delete_matrix(k);
  delete_matrix(ktKi);
  delete_matrix(ktKik);
}


/*
 * new_predutilGPsep_lite:
 *
 * utility function that allocates and calculate useful vectors 
 * and matrices for prediction; used by predGP_lite and dmus2GP
 */

void new_predutilGPsep_lite(GPsep *gpsep, unsigned int nn, double **XX, 
  double ***k, double ***ktKi, double **ktKik)
{
  unsigned int i, j, m, n;

  /* k <- covar(X1=X, X2=XX, d=Zt$d, g=0) */
  n = gpsep->n;  m = gpsep->m;
  *k = new_matrix(n, nn);
  covar_sep(m, gpsep->X, n, XX, nn, gpsep->d, 0.0, *k);
  
  /* ktKi <- t(k) %*% util$Ki */
  *ktKi = new_matrix(n, nn);
  linalg_dsymm(CblasRight,nn,n,1.0,gpsep->Ki,n,*k,nn,0.0,*ktKi,nn);
  /* ktKik <- diag(ktKi %*% k) */
  *ktKik = new_zero_vector(nn); 
  for(i=0; i<nn; i++) for(j=0; j<n; j++) (*ktKik)[i] += (*ktKi)[j][i]*(*k)[j][i];
}


/*
 * predGPsep_lite:
 *
 * return the student-t predictive equations,
 * i.e., parameters to a multivatiate t-distribution
 * for XX predictive locations of dimension (n*m);
 * lite because sigma2 not Sigma is calculated
 */

void predGPsep_lite(GPsep* gpsep, unsigned int nn, double **XX, double *mean, 
     double *sigma2, double *df, double *llik)
{
  unsigned int i;
  double **k, **ktKi;
  double *ktKik;
  double phidf;
  
  /* sanity checks */
  assert(df);
  *df = gpsep->n; 

  /* utility calculations */
  new_predutilGPsep_lite(gpsep, nn, XX, &k, &ktKi, &ktKik);

  /* mean <- ktKi %*% Z */
  if(mean) linalg_dgemv(CblasNoTrans,nn,gpsep->n,1.0,ktKi,nn,gpsep->Z,1,0.0,mean,1);

  /* Sigma <- phi*(Sigma - ktKik)/df */
  /* *df = n - m - 1.0; */  /* only if estimating beta */
  if(sigma2) {
    phidf = gpsep->phi/(*df);
    for(i=0; i<nn; i++) sigma2[i] = phidf * (1.0 + gpsep->g - ktKik[i]);
  }

  /* calculate marginal likelihood (since we have the bits) */
  /* might move to updateGP if we decide to move phi to updateGP */
  if(llik) *llik = 0.0 - 0.5*(((double) gpsep->n) * log(0.5* gpsep->phi) + gpsep->ldetK);
  /* continuing: - ((double) n)*M_LN_SQRT_2PI;*/

  /* clean up */
  delete_matrix(k);
  delete_matrix(ktKi);
  free(ktKik);
}


/*
 * predGPsep_R:
 *
 * R-interface to C-side function that 
 * returns the student-t predictive equations,
 * i.e., parameters to a multivatiate t-distribution
 * for XX predictive locations of dimension (n*m)
 * using the stored GP parameterization
 */

void predGPsep_R(/* inputs */
        int *gpsepi_in,
        int *m_in,
        int *nn_in,
        double *XX_in,
        int *lite_in,
        
        /* outputs */
        double *mean_out,
        double *Sigma_out,
        double *df_out,
        double *llik_out)
{
  GPsep* gpsep;
  unsigned int gpsepi;
  double **Sigma, **XX;

  /* get the gp */
  gpsepi = *gpsepi_in;
  if(gpseps == NULL || gpsepi >= NGPsep || gpseps[gpsepi] == NULL) 
    error("gpsep %d is not allocated\n", gpsepi);
  gpsep = gpseps[gpsepi];
  if((unsigned) *m_in != gpsep->m) 
    error("ncol(X)=%d does not match GPsep/C-side (%d)", *m_in, gpsep->m);

  /* sanity check and XX representation */
  XX = new_matrix_bones(XX_in, *nn_in, *m_in);
  if(! *lite_in) Sigma = new_matrix_bones(Sigma_out, *nn_in, *nn_in);
  else Sigma = NULL;

  /* call the C-only Predict function */
  if(*lite_in) predGPsep_lite(gpsep, *nn_in, XX, mean_out, Sigma_out, df_out, llik_out);
  else predGPsep(gpsep, *nn_in, XX, mean_out, Sigma, df_out, llik_out);
  
  /* clean up */
  free(XX);
  if(Sigma) free(Sigma);
}



/*
 * alGPsep_R:
 *
 * R interface to C-side function that returns the a Monte Carlo approximation 
 * to the expected improvement (EI) and expected y-value (EY) under an augmented 
 * Lagrangian with constraint separable GPs (cgpseps) assuming a known linear 
 * objective function with scale bscale.  The constraints can be scaled with the cnorms
 */

void alGPsep_R(/* inputs */
       int *m_in,
       double *XX_in,
       int *nn_in,
       int *fgpsepi_in,
       double *fnorm_in,
       int *ncgpseps_in,
       int *cgpsepis_in,
       double *cnorms_in,
       double *lambda_in,
       double *alpha_in,
       double *ymin_in,
       int *nomax_in,
       int *N_in,
       
       /* outputs */
       double *eys_out,
       double *eis_out)
{
  GPsep **cgpseps, *fgpsep;
  unsigned int gpsepi, ncgpseps, i, j, k;
  double **cmu, **cs, **XX;
  double *mu, *s;
  double df;

  /* get the gps */
  ncgpseps = *ncgpseps_in;
  cgpseps = (GPsep**) malloc(sizeof(GPsep*) * ncgpseps);
  for(i=0; i<ncgpseps; i++) {
    gpsepi = cgpsepis_in[i];
    if(gpseps == NULL || gpsepi >= NGPsep || gpseps[gpsepi] == NULL) 
      error("gpsep %d is not allocated\n", gpsepi);
    cgpseps[i] = gpseps[gpsepi];
    if((unsigned) *m_in != cgpseps[i]->m)  
      error("ncol(X)=%d does not match GPsep/C-side (%d)", *m_in, cgpseps[i]->m);
  }

  /* make matrix bones */
  XX = new_matrix_bones(XX_in, *nn_in, *m_in);

  /* allocate storage for the (possibly null) distribution of f */
  mu = new_vector(*nn_in);
  if(*fgpsepi_in >= 0) { /* if modeling f */
    gpsepi = *fgpsepi_in;
    if(gpseps == NULL || gpsepi >= NGPsep || gpseps[gpsepi] == NULL) 
      error("gpsep %d is not allocated\n", gpsepi);
    fgpsep = gpseps[gpsepi];
    s = new_vector(*nn_in);
    predGPsep_lite(fgpsep, *nn_in, XX, mu, s, &df, NULL);
    for(k=0; k<*nn_in; k++) s[k] = sqrt(s[k]);
  } else { /* not modeling f; using known linear mean */
    for(k=0; k<*nn_in; k++) mu[k] = sumv(XX[k], cgpseps[0]->m);
    s = NULL;
  }

  /* allocate storage for means and variances under normal approx */
  cmu = new_matrix(ncgpseps, *nn_in);
  cs = new_matrix(ncgpseps, *nn_in);
  for(j=0; j<ncgpseps; j++) {
    predGPsep_lite(cgpseps[j], *nn_in, XX, cmu[j], cs[j], &df, NULL);
    for(k=0; k<*nn_in; k++) cs[j][k] = sqrt(cs[j][k]);
  }

  /* clean up */
  free(XX);
  free(cgpseps);

  GetRNGstate();

  /* use mu and s to calculate EI and EY */
  calc_al_eiey(ncgpseps, *nn_in, mu, s, *fnorm_in, cmu, cs, cnorms_in, 
    lambda_in, alpha_in, *ymin_in, *nomax_in, *N_in, eys_out, eis_out);

  PutRNGstate();

  /* clean up */
  delete_matrix(cmu);
  delete_matrix(cs);
  free(mu);
  if(s) free(s);
}
