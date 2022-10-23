#ifndef __RHELP_H__
#define __RHELP_H__

#include <stdio.h>
#include <time.h>

/* this is now covered by -D RPRINT flags in Makevars */
/*#define RPRINT*/

#ifndef RPRINT
void warning(const char *str, ...);
void error(const char *str, ...);
/* #define DOUBLE_EPS 2.220446e-16 */
#define M_LN_SQRT_2PI   0.918938533204672741780329736406  
#include <stdio.h>
#define MYstdout stdout
#define MYstderr stderr
#else
// #include <R_ext/Utils.h>
// #include <R.h>
// #include <Rmath.h>
// #include <Rinterface.h>
#include <R_ext/Error.h>
extern FILE *MYstdout, *MYstderr;
#endif

// void R_FlushConsole(void); /* R < 2.3 does not have this in R.h (in Rinterface.h) */
void MYprintf(FILE *outfile, const char *str, ...);
void MYflush(FILE *outfile);
time_t MY_r_process_events(time_t itime);

#endif
