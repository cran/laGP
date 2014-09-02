\name{mleGP}
\alias{mleGP}
\alias{jmleGP}
\alias{jmleGP.R}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
  Inference for GP correlation parameters
}
\description{
  Maximum likelihood/a posteriori inference for (isotropic Gaussian) 
  lengthscale and nugget parameters, marginally or jointly, for 
  Gaussian process regression
}
\usage{
mleGP(gpi, param = c("d", "g"), tmin = 0, tmax = -1, ab = c(0, 0),
      verb = 0)
jmleGP(gpi, drange=c(0,10), grange=c(sqrt(.Machine$double.eps), 1),
      dab=c(0,0), gab=c(0,0), verb=0)
jmleGP.R(gpi, N=100, drange=c(0,10), grange=c(sqrt(.Machine$double.eps), 1),
      dab=c(0,0), gab=c(0,0), verb=0)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{gpi}{
    a C-side GP object identifier (positive integer);
    e.g., as returned by \code{\link{newGP}}
  }
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
      for \code{mleGP}, largest value considered for the parameter (\code{param})
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
\item{dab}{ for \code{jmleGP}, this is \code{ab} for the lengthscale
  parameter }
\item{gab}{ for \code{jmleGP}, this is \code{ab} for the nugget
  parameter }
  \item{verb}{
    a verbosity argument indicating how much information about the optimization
    steps should be printed to the screen; \code{verb <= 0} is quiet; for
    \code{jmleGP}, a \code{verb - 1} value is passed to the \code{mleGP} subroutine(s).
}
}
\details{
  \code{mleGP} performs marginal (or profile) inference for the
  specified \code{param}, either the lengthscale or the nugget;  it has
  a hard-coded 100-max iterations threshold and a
  \code{sqrt(.Machine$double.eps)} tolerance for determining convergence.

  \code{jmleGP} provides joint inference by iterating over the
  marginals.  The \code{jmleGP.R} function is an \R-only wrapper around
  \code{mleGP} (which is primarily in C), whereas \code{jmleGP} is
  primarily in C but should give the same output. 

  By default, the algorithm is a modified Newton method initialized at
  the previous setting of the parameter(s).  Convergence is usually very
  fast.  Sometimes the Newton method fails however (for any of the usual
  reasons). The \code{mleGP} method can detect this, and in such cases
  it invokes a \code{Brent_fmin} call instead - this is the method behind
  the \code{\link{optimize}} function.

  Note that the \code{gpi} object must have been allocated with
  \code{dK=TRUE}; alternatively, one can call \code{buildKGP} - however,
  this is not in the namespace at this time
}
\value{
  A self-explanatory \code{\link{data.frame}} is returned containing the
  values inferred and the number of iterations used.  The
  \code{jmleGP.R} function will also show progress details (the values
  obtained after each iteration over the marginals).

  However, the most important \dQuote{output} is the modified GP object
  which retains the setting of the parameters reported on output as a 
  side effect
}
\references{
  For standard GP inference, refer to any graduate text, e.g., Rasmussen
  & Williams \emph{Gaussian Processes for Machine Learning};
}
\author{
    Robert B. Gramacy \email{rbgramacy@chicagobooth.edu}
}
%% ~Make other sections like Warning with \section{Warning }{....} ~

\seealso{
  \code{\link{newGP}}, \code{\link{laGP}}, \code{\link{llikGP}}, \code{\link{optimize}}
}
\examples{
## a simple example with estimated nugget
library(MASS)

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
print(jmleGP(gpi, c(d$min, d$max), c(g$min, g$max)))

## plot the resulting predictive surface
N <- 100
XX <- matrix(seq(min(X), max(X), length=N), ncol=1)
p <- predGP(gpi, XX, lite=TRUE)
plot(X, Z, main="stationary GP fit to motorcycle data")
lines(XX, p$mean, lwd=2)
lines(XX, p$mean+1.96*sqrt(p$s2*p$df/(p$df-2)), col=2, lty=2)
lines(XX, p$mean-1.96*sqrt(p$s2*p$df/(p$df-2)), col=2, lty=2)
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
