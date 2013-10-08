#ifndef __RHELP_H__
#define __RHELP_H__

#include <stdio.h>
#include <time.h>

/* this is now covered by -D RPRINT flags in Makevars */
/*#define RPRINT*/

#ifndef RPRINT
void warning(const char *str, ...);
void error(const char *str, ...);
#define DOUBLE_EPS 2.220446e-16
#define M_LN_SQRT_2PI   0.918938533204672741780329736406  
#include <stdio.h>
#define mystdout stdout
#define mystderr stderr
#else
#include <R_ext/Utils.h>
#include <R.h>
#include <Rmath.h>
extern FILE *mystdout, *mystderr;
#endif

void R_FlushConsole(void); /* R < 2.3 does not have this in R.h (in Rinterface.h) */
void myprintf(FILE *outfile, const char *str, ...);
void myflush(FILE *outfile);
time_t my_r_process_events(time_t itime);

#endif
