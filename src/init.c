#include <stdlib.h> // for NULL
#include <R_ext/Rdynload.h>

/* .C calls */
extern void aGP_R(void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *);
extern void aGPsep_R(void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *);
#ifdef _GPU
extern void alcGP_gpu_R(void *, void *, void *, void *, void *, void *, void *, void *);
#endif
extern void alcGP_omp_R(void *, void *, void *, void *, void *, void *, void *, void *);
extern void alcGP_R(void *, void *, void *, void *, void *, void *, void *, void *);
extern void alcGPsep_omp_R(void *, void *, void *, void *, void *, void *, void *, void *);
extern void alcGPsep_R(void *, void *, void *, void *, void *, void *, void *, void *);
extern void alcrayGP_R(void *, void *, void *, void *, void *, void *, void *, void *, void *);
extern void alcrayGPsep_R(void *, void *, void *, void *, void *, void *, void *, void *, void *);
extern void alGP_R(void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *);
extern void alGPsep_R(void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *);
extern void buildKGP_R(void *);
extern void buildKGPsep_R(void *);
extern void copyGP_R(void *, void *);
extern void deletedKGP_R(void *);
extern void deletedKGPsep_R(void *);
extern void deleteGP_R(void *);
extern void deleteGPs_R();
extern void deleteGPsep_R(void *);
extern void deleteGPseps_R();
extern void distance_R(void *, void *, void *, void *, void *, void *);
extern void distance_symm_R(void *, void *, void *, void *);
extern void dllikGP_nug_R(void *, void *, void *, void *);
extern void dllikGP_R(void *, void *, void *, void *);
extern void dllikGPsep_nug_R(void *, void *, void *, void *);
extern void dllikGPsep_R(void *, void *, void *);
extern void dmus2GP_R(void *, void *, void *, void *, void *, void *, void *, void *, void *, void *);
extern void efiGP_R(void *, void *, void *, void *, void *);
extern void getdGPsep_R(void *, void *);
extern void getgGPsep_R(void *, void *);
extern void getmGPsep_R(void *, void *);
extern void ieciGP_R(void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *);
extern void ieciGPsep_R(void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *);
extern void Igamma_inv_R(void *, void *, void *, void *, void *);
extern void jmleGP_R(void *, void *, void *, void *, void *, void *, void *, void *, void *, void *);
extern void jmleGPsep_R(void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *);
extern void laGP_R(void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *);
extern void laGPsep_R(void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *, void *);
extern void lalcrayGP_R(void *, void *, void *, void *, void *, void *, void *, void *, void *, void *);
extern void lalcrayGPsep_R(void *, void *, void *, void *, void *, void *, void *, void *, void *, void *);
extern void llikGP_R(void *, void *, void *, void *);
extern void llikGPsep_R(void *, void *, void *, void *);
extern void mleGP_R(void *, void *, void *, void *, void *, void *, void *, void *);
extern void mleGPsep_both_R(void *, void *, void *, void *, void *, void *, void *, void *, void *, void *);
extern void mleGPsep_nug_R(void *, void *, void *, void *, void *, void *, void *);
extern void mleGPsep_R(void *, void *, void *, void *, void *, void *, void *, void *, void *, void *);
extern void mspeGP_R(void *, void *, void *, void *, void *, void *, void *, void *, void *);
extern void newGP_R(void *, void *, void *, void *, void *, void *, void *, void *);
extern void newGPsep_R(void *, void *, void *, void *, void *, void *, void *, void *);
extern void newparamsGP_R(void *, void *, void *);
extern void newparamsGPsep_R(void *, void *, void *);
extern void predGP_R(void *, void *, void *, void *, void *, void *, void *, void *, void *);
extern void predGPsep_R(void *, void *, void *, void *, void *, void *, void *, void *, void *);
extern void rbetter_R(void *, void *, void *, void *, void *);
extern void updateGP_R(void *, void *, void *, void *, void *, void *);
extern void updateGPsep_R(void *, void *, void *, void *, void *, void *);

static const R_CMethodDef CEntries[] = {
    {"aGP_R",            (DL_FUNC) &aGP_R,            30},
    {"aGPsep_R",         (DL_FUNC) &aGPsep_R,         27},
#ifdef _GPU
    {"alcGP_gpu_R",      (DL_FUNC) &alcGP_gpu_R,       8},
#endif
    {"alcGP_omp_R",      (DL_FUNC) &alcGP_omp_R,       8},
    {"alcGP_R",          (DL_FUNC) &alcGP_R,           8},
    {"alcGPsep_omp_R",   (DL_FUNC) &alcGPsep_omp_R,    8},
    {"alcGPsep_R",       (DL_FUNC) &alcGPsep_R,        8},
    {"alcrayGP_R",       (DL_FUNC) &alcrayGP_R,        9},
    {"alcrayGPsep_R",    (DL_FUNC) &alcrayGPsep_R,     9},
    {"alGP_R",           (DL_FUNC) &alGP_R,           16},
    {"alGPsep_R",        (DL_FUNC) &alGPsep_R,        18},
    {"buildKGP_R",       (DL_FUNC) &buildKGP_R,        1},
    {"buildKGPsep_R",    (DL_FUNC) &buildKGPsep_R,     1},
    {"copyGP_R",         (DL_FUNC) &copyGP_R,          2},
    {"deletedKGP_R",     (DL_FUNC) &deletedKGP_R,      1},
    {"deletedKGPsep_R",  (DL_FUNC) &deletedKGPsep_R,   1},
    {"deleteGP_R",       (DL_FUNC) &deleteGP_R,        1},
    {"deleteGPs_R",      (DL_FUNC) &deleteGPs_R,       0},
    {"deleteGPsep_R",    (DL_FUNC) &deleteGPsep_R,     1},
    {"deleteGPseps_R",   (DL_FUNC) &deleteGPseps_R,    0},
    {"distance_R",       (DL_FUNC) &distance_R,        6},
    {"distance_symm_R",  (DL_FUNC) &distance_symm_R,   4},
    {"dllikGP_nug_R",    (DL_FUNC) &dllikGP_nug_R,     4},
    {"dllikGP_R",        (DL_FUNC) &dllikGP_R,         4},
    {"dllikGPsep_nug_R", (DL_FUNC) &dllikGPsep_nug_R,  4},
    {"dllikGPsep_R",     (DL_FUNC) &dllikGPsep_R,      3},
    {"dmus2GP_R",        (DL_FUNC) &dmus2GP_R,        10},
    {"efiGP_R",          (DL_FUNC) &efiGP_R,           5},
    {"getdGPsep_R",      (DL_FUNC) &getdGPsep_R,       2},
    {"getgGPsep_R",      (DL_FUNC) &getgGPsep_R,       2},
    {"getmGPsep_R",      (DL_FUNC) &getmGPsep_R,       2},
    {"ieciGP_R",         (DL_FUNC) &ieciGP_R,         11},
    {"ieciGPsep_R",      (DL_FUNC) &ieciGPsep_R,      11},
    {"Igamma_inv_R",     (DL_FUNC) &Igamma_inv_R,      5},
    {"jmleGP_R",         (DL_FUNC) &jmleGP_R,         10},
    {"jmleGPsep_R",      (DL_FUNC) &jmleGPsep_R,      13},
    {"laGP_R",           (DL_FUNC) &laGP_R,           26},
    {"laGPsep_R",        (DL_FUNC) &laGPsep_R,        25},
    {"lalcrayGP_R",      (DL_FUNC) &lalcrayGP_R,      10},
    {"lalcrayGPsep_R",   (DL_FUNC) &lalcrayGPsep_R,   10},
    {"llikGP_R",         (DL_FUNC) &llikGP_R,          4},
    {"llikGPsep_R",      (DL_FUNC) &llikGPsep_R,       4},
    {"mleGP_R",          (DL_FUNC) &mleGP_R,           8},
    {"mleGPsep_both_R",  (DL_FUNC) &mleGPsep_both_R,  10},
    {"mleGPsep_nug_R",   (DL_FUNC) &mleGPsep_nug_R,    7},
    {"mleGPsep_R",       (DL_FUNC) &mleGPsep_R,       10},
    {"mspeGP_R",         (DL_FUNC) &mspeGP_R,          9},
    {"newGP_R",          (DL_FUNC) &newGP_R,           8},
    {"newGPsep_R",       (DL_FUNC) &newGPsep_R,        8},
    {"newparamsGP_R",    (DL_FUNC) &newparamsGP_R,     3},
    {"newparamsGPsep_R", (DL_FUNC) &newparamsGPsep_R,  3},
    {"predGP_R",         (DL_FUNC) &predGP_R,          9},
    {"predGPsep_R",      (DL_FUNC) &predGPsep_R,       9},
    {"rbetter_R",        (DL_FUNC) &rbetter_R,         5},
    {"updateGP_R",       (DL_FUNC) &updateGP_R,        6},
    {"updateGPsep_R",    (DL_FUNC) &updateGPsep_R,     6},
    {NULL, NULL, 0}
};

void R_init_laGP(DllInfo *dll)
{
    R_registerRoutines(dll, CEntries, NULL, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}

