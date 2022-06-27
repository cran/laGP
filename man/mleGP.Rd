\name{mleGP}
\alias{mleGP}
\alias{mleGPsep.R}
\alias{mleGPsep}
\alias{jmleGP}
\alias{jmleGP.R}
\alias{jmleGPsep}
\alias{jmleGPsep.R}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
  Inference for GP correlation parameters
}
\description{
  Maximum likelihood/a posteriori inference for (isotropic and separable)
   Gaussian lengthscale and nugget parameters, marginally or jointly, for 
  Gaussian process regression
}
\usage{
mleGP(gpi, param = c("d", "g"), tmin=sqrt(.Machine$double.eps), 
  tmax = -1, ab = c(0, 0), verb = 0)
mleGPsep(gpsepi, param=c("d", "g", "both"), tmin=rep(sqrt(.Machine$double.eps), 2),
  tmax=c(-1,1), ab=rep(0,4), maxit=100, verb=0)
mleGPsep.R(gpsepi, param=c("d", "g"), tmin=sqrt(.Machine$double.eps), 
  tmax=-1, ab=c(0,0), maxit=100, verb=0)
jmleGP(gpi, drange=c(sqrt(.Machine$double.eps),10), 
  grange=c(sqrt(.Machine$double.eps), 1), dab=c(0,0), gab=c(0,0), verb=0)
jmleGP.R(gpi, N=100, drange=c(sqrt(.Machine$double.eps),10), 
  grange=c(sqrt(.Machine$double.eps), 1), dab=c(0,0), gab=c(0,0), verb=0)
jmleGPsep(gpsepi, drange=c(sqrt(.Machine$double.eps),10), 
  grange=c(sqrt(.Machine$double.eps), 1), dab=c(0,0), gab=c(0,0), 
  maxit=100, verb=0)
jmleGPsep.R(gpsepi, N=100, drange=c(sqrt(.Machine$double.eps),10), 
  grange=c(sqrt(.Machine$double.eps), 1), dab=c(0,0), gab=c(0,0), 
  maxit=100, mleGPsep=mleGPsep.R, verb=0)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{gpi}{
    a C-side GP object identifier (positive integer);
    e.g., as returned by \code{\link{newGP}}
  }
  \item{gpsepi}{ similar to \code{gpi} but indicating a separable GP object,
  as returned by \code{\link{newGPsep}}}
  \item{N}{ for \code{jmleGP.R}, the maximum number of times the pair of margins
    should be iterated over before determining failed convergence; note
    that (at this time) \code{jmleGP} uses a hard-coded \code{N=100} in
    its C implementation}
  \item{param}{
    for \code{mleGP}, indicating whether to work on the lengthscale
    (\code{d}) or nugget (\code{g}) margin
}
  \item{tmin}{
    for \code{mleGP}, smallest value considered for the parameter (\code{param})
  }
\item{tmax}{
      for \code{mleGP}, largest value considered for the parameter (\code{param}); a setting of \code{-1} for lengthscales, the default, causes
      \code{ncol(X)^2} to be used
    }
\item{drange}{ for \code{jmleGP}, these are \code{c(tmin, tmax)}
    values for the lengthscale parameter; the default values are
    reasonable for 1-d inputs in the unit interval }
\item{grange}{ for \code{jmleGP}, these are \code{c(tmin, tmax)}
    values for the nugget parameter; the default values are reasonable
 for responses with a range of one }
\item{ab}{
  for \code{mleGP}, a non-negative 2-vector describing shape and rate parameters to a
  Gamma prior for the parameter (\code{param}); a zero-setting for
  either value results in no-prior being used (MLE inference); otherwise
  MAP inference is performed
}
\item{maxit}{ for \code{mleGPsep} this is passed as \code{control=list(trace=maxit)}
  to \code{\link{optim}}'s L-BFGS-B method for optimizing
  the likelihood/posterior of a separable GP representation; this argument is
  not used for isotropic GP versions, nor for optimizing the nugget }
\item{dab}{ for \code{jmleGP}, this is \code{ab} for the lengthscale
  parameter }
\item{gab}{ for \code{jmleGP}, this is \code{ab} for the nugget
  parameter }
\item{mleGPsep}{ function for internal MLE calculation of the separable
  lengthscale parameter; one of either \code{mleGPsep.R} based on 
  \code{method="L-BFGS-B"} using \code{\link{optim}}; 
  or \code{mleGPsep} using the \code{C} entry point \code{lbfgsb}.  
  Both options use a \code{C} backend for the nugget }
\item{verb}{
    a verbosity argument indicating how much information about the optimization
    steps should be printed to the screen; \code{verb <= 0} is quiet; for
    \code{jmleGP}, a \code{verb - 1} value is passed to the \code{mleGP} or 
    \code{mleGPsep} subroutine(s)
}
}
\details{
  \code{mleGP} and \code{mleGPsep} perform marginal (or profile) inference
  for the specified \code{param}, either the lengthscale or the nugget.  
  \code{mleGPsep} can perform simultaneous lengthscale and nugget inference via
  a common gradient with \code{param = "both"}.  More details are provided below.

  For the lengthscale, \code{mleGP} uses a Newton-like scheme with analytic first 
  and second derivatives (more below) to find the scalar parameter for the isotropic
  Gaussian correlation function, with hard-coded 100-max iterations threshold and a
  \code{sqrt(.Machine$double.eps)} tolerance for determining convergence;
  \code{mleGPsep.R} uses L-BFGS-B via \code{\link{optim}} for the 
  vectorized parameter of the separable Gaussian correlation, with a user-supplied
  maximum number of iterations (\code{maxit}) passed to \code{\link{optim}}.  
  When \code{maxit} is reached the output \code{conv = 1} is returned, 
  subsequent identical calls to \code{mleGPsep.R} can be used to continue with
  further iterations.  \code{mleGPsep} is similar, but uses the \code{C} 
  entry point \code{lbfgsb}.

  For the nugget, both \code{mleGP} and \code{mleGPsep} utilize a (nearly 
  identical) Newton-like scheme leveraging first and second derivatives.

  \code{jmleGP} and \code{jmleGPsep} provide joint inference by iterating
  over the marginals of lengthscale and nugget.  The \code{jmleGP.R} function
  is an \R-only wrapper around
  \code{mleGP} (which is primarily in C), whereas \code{jmleGP} is
  primarily in C but with reduced output and 
  with hard-coded \code{N=100}.  The same is true for \code{jmleGPsep}.

  \code{mleGPsep} provides a \code{param = "both"} alternative to \code{jmleGPsep} 
  leveraging a common gradient. It can be helpful to supply a larger \code{maxit} 
  argument compared to \code{jmleGPsep} as the latter may do up to 100 outer
  iterations, cycling over lengthscale and nugget.  \code{mleGPsep} usually requires
  many fewer total iterations, unless one of the lengthscale or nugget is 
  already converged. In anticipation of \code{param = "both"} the 
  \code{mleGPsep} function has longer default values for its bounds and prior 
  arguments.  These longer arguments are 
  ignored when \code{param != "both"}.  At this time \code{mleGP} does not have a 
  \code{param = "both"} option.

  All methods are initialized at the value of the parameter(s) currently
  stored by the C-side object referenced by \code{gpi} or \code{gpsepi}.  It is
  \emph{highly recommended} that sensible range values (\code{tmin}, \code{tmax}
  or \code{drange}, \code{grange}) be provided.  The defaults provided are
  too loose for most applications.  As illustrated in the examples below,
  the \code{\link{darg}} and \code{\link{garg}} functions can be used
  to set appropriate ranges from the distributions of inputs and output
  data respectively.

  The Newton-like method implemented for the isotropic lengthscale and for the
  nugget offers very fast convergence to local maxima, but sometimes it fails
  to converge (for any of the usual reasons).  The implementation
  detects this, and in such cases it invokes a \code{Brent_fmin} call instead -
  this is the method behind the \code{\link{optimize}} function.

  Note that the \code{gpi} or \code{gpsepi} object(s) must have been allocated with
  \code{dK=TRUE}; alternatively, one can call \code{buildKGP} or \code{buildKGPsep}
  - however, this is not in the NAMESPACE at this time
}
\value{
  A self-explanatory \code{\link{data.frame}} is returned containing the
  values inferred and the number of iterations used.  The
  \code{jmleGP.R} and \code{jmleGPsep.R} functions will also show progress details (the values
  obtained after each iteration over the marginals).

  However, the most important \dQuote{output} is the modified GP object
  which retains the setting of the parameters reported on output as a 
  side effect.

  \code{mleGPsep} and \code{jmleGPsep} provide an output field/column
  called \code{conv} indicating convergence (when 0), or alternately 
  a value agreeing with a non-convergence code provided on 
  output by \code{\link{optim}}
}
\references{
  For standard GP inference, refer to any graduate text, e.g., Rasmussen
  & Williams \emph{Gaussian Processes for Machine Learning}, or

  Gramacy, R. B. (2020) \emph{Surrogates: Gaussian Process Modeling,
  Design and Optimization for the Applied Sciences}. Boca Raton,
  Florida: Chapman Hall/CRC.  (See Chapter 5.)
  \url{https://bobby.gramacy.com/surrogates/}
}
\author{
    Robert B. Gramacy \email{rbg@vt.edu}
}
%% ~Make other sections like Warning with \section{Warning }{....} ~

\seealso{
\code{vignette("laGP")}, 
  \code{\link{newGP}}, \code{\link{laGP}}, \code{\link{llikGP}}, \code{\link{optimize}}
}
\examples{
## a simple example with estimated nugget
if(require("MASS")) {

  ## motorcycle data and predictive locations
  X <- matrix(mcycle[,1], ncol=1)
  Z <- mcycle[,2]

  ## get sensible ranges
  d <- darg(NULL, X)
  g <- garg(list(mle=TRUE), Z)

  ## initialize the model
  gpi <- newGP(X, Z, d=d$start, g=g$start, dK=TRUE)

  ## separate marginal inference (not necessary - for illustration only)
  print(mleGP(gpi, "d", d$min, d$max))
  print(mleGP(gpi, "g", g$min, g$max))

  ## joint inference (could skip straight to here)
  print(jmleGP(gpi, drange=c(d$min, d$max), grange=c(g$min, g$max)))

  ## plot the resulting predictive surface
  N <- 100
  XX <- matrix(seq(min(X), max(X), length=N), ncol=1)
  p <- predGP(gpi, XX, lite=TRUE)
  plot(X, Z, main="stationary GP fit to motorcycle data")
  lines(XX, p$mean, lwd=2)
  lines(XX, p$mean+1.96*sqrt(p$s2*p$df/(p$df-2)), col=2, lty=2)
  lines(XX, p$mean-1.96*sqrt(p$s2*p$df/(p$df-2)), col=2, lty=2)

  ## clean up
  deleteGP(gpi)
}

## 
## with a separable correlation function 
##

## 2D Example: GoldPrice Function, mimicking GP_fit package
f <- function(x) 
{
  x1 <- 4*x[,1] - 2
  x2 <- 4*x[,2] - 2;
  t1 <- 1 + (x1 + x2 + 1)^2*(19 - 14*x1 + 3*x1^2 - 14*x2 + 6*x1*x2 + 3*x2^2);
  t2 <- 30 + (2*x1 -3*x2)^2*(18 - 32*x1 + 12*x1^2 + 48*x2 - 36*x1*x2 + 27*x2^2);
  y <- t1*t2;
  return(y)
}

## build design
library(tgp)
n <- 50 ## change to 100 or 1000 for more interesting demo
B <- rbind(c(0,1), c(0,1))
X <- dopt.gp(n, Xcand=lhs(10*n, B))$XX
## this differs from GP_fit in that we use the log response
Y <- log(f(X))

## get sensible ranges
d <- darg(NULL, X)
g <- garg(list(mle=TRUE), Y)

## build GP and jointly optimize via profile mehtods
gpisep <- newGPsep(X, Y, d=rep(d$start, 2), g=g$start, dK=TRUE)
jmleGPsep(gpisep, drange=c(d$min, d$max), grange=c(g$min, g$max))

## clean up
deleteGPsep(gpisep)

## alternatively, we can optimize via a combined gradient
gpisep <- newGPsep(X, Y, d=rep(d$start, 2), g=g$start, dK=TRUE)
mleGPsep(gpisep, param="both", tmin=c(d$min, g$min), tmax=c(d$max, g$max))
deleteGPsep(gpisep)
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.

\keyword{ nonparametric }
\keyword{ nonlinear }
\keyword{ smooth }
\keyword{ models }
\keyword{ regression }
\keyword{ spatial }
\keyword{ optimize }
