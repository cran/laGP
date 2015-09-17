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

#include "rhelp.h"
#ifdef RPRINT
#include <R_ext/Print.h>
#include <R.h>
FILE *MYstdout = (FILE*) 0;
FILE *MYstderr = (FILE*) 1;
#endif
#include <stdarg.h>
#include <time.h>
#include <assert.h>

/* 
 * MYprintf:
 *
 * a function many different types of printing-- in particular, using 
 * the Rprintf if the code happens to be compiled with RPRINT, 
 * othersie fprintf (takes the same arguments as fprintf)
 */

void MYprintf(FILE *outfile, const char *str, ...)
{
  va_list argp;
  va_start(argp, str);
  
  #ifdef RPRINT
  if(outfile == MYstdout) Rvprintf(str, argp);
  else if(outfile == MYstderr) REvprintf(str, argp);
  else vfprintf(outfile, str, argp);
  #else
  vfprintf(outfile, str, argp);
  #endif

  va_end(argp);
}


#ifndef RPRINT
/*
 * error:
 *
 * printf style function that reports errors to stderr
 */

void error(const char *str, ...)
{
  va_list argp;
  va_start(argp, str);
  
  MYprintf(stderr, "ERROR: ");
  vfprintf(stderr, str, argp);
  
  va_end(argp);
  MYflush(stderr);
  
  /* add a final newline */
  MYprintf(stderr, "\n");

  /* kill the code */
  assert(0);
}


/*
 * warning:
 *
 * printf style function that reports warnings to stderr
 */

void warning(const char *str, ...)
{
  va_list argp;
  va_start(argp, str);
  
  MYprintf(stderr, "WARNING: ");
  vfprintf(stderr, str, argp);

  va_end(argp);
  MYflush(stderr);

  /* add a final newline */
  MYprintf(stderr, "\n");
}
#endif


/* 
 * MYflush:
 *
 * a function for many different types of flushing--  in particular, 
 * using * the R_FlushConsole the code happens to be compiled with 
 * RPRINT, otherwise fflush
 */

void MYflush(FILE *outfile)
{
#ifdef RPRINT
  R_FlushConsole();
#else
  fflush(outfile);
#endif
}


/*
 * MY_r_process_events:
 *
 * at least every 1 second(s) pass control back to
 * R so that it can check for interrupts and/or 
 * process other R-gui events
 */

time_t MY_r_process_events(time_t itime)
{
#ifdef RPRINT  
  time_t ntime = time(NULL);

  if(ntime - itime > 1) {
    R_FlushConsole();
    R_CheckUserInterrupt();
#if  (defined(HAVE_AQUA) || defined(Win32) || defined(Win64))
    R_ProcessEvents();
#endif
    itime = ntime;
  }
#endif
  return itime;
}
