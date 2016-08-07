\name{newGP}
\alias{newGP}
\alias{updateGP}
\alias{newGPsep}
\alias{updateGPsep}

\title{
Create A New GP Object
}
\description{
  Build a Gaussian process C-side object based on the \code{X}-\code{Z}
  data and parameters provided, and augment those objects with new
  data
}
\usage{
newGP(X, Z, d, g, dK = FALSE)
newGPsep(X, Z, d, g, dK = FALSE)
updateGP(gpi, X, Z, verb = 0)
updateGPsep(gpsepi, X, Z, verb = 0)
}

\arguments{
  \item{X}{
     a \code{matrix} or \code{data.frame} containing
    the full (large) design matrix of input locations
  }
\item{Z}{
      a vector of responses/dependent values with \code{length(Z) = ncol(X)}
}
  \item{d}{
    a positive scalar lengthscale parameter for an isotropic Gaussian
    correlation function (\code{newGP}); or a vector for a separable
    version (\code{newGPsep}) 
}
  \item{g}{
    a positive scalar nugget parameter
}
  \item{dK}{
    a scalar logical indicating whether or not derivative information
    (for the lengthscale) should be maintained by the GP object; 
    this is required for calculating MLEs/MAPs of the lengthscale 
    parameter(s) via \code{\link{mleGP}} and \code{\link{jmleGP}}
}
\item{gpi}{
  a C-side GP object identifier (positive integer); e.g., as returned by \code{newGP}
}
\item{gpsepi}{ similar to \code{gpi} but indicating a separable GP object,
  as returned by \code{\link{newGPsep}}}
\item{verb}{
  a scalar logical indicating the verbosity level.  A positive value
  causes progress statements to be printed to the screen for each
  update of \code{i in 1:nrow(X)}
}
}
\details{
  \code{newGP} allocates a new GP object on the C-side and returns its
  unique integer identifier (\code{gpi}), taking time which is cubic on
  \code{nrow(X)};  allocated GP objects must (eventually) be destroyed
  with \code{\link{deleteGP}} or \code{\link{deleteGPs}} or memory will leak.
  The same applies for \code{newGPsep}, except deploying a separable 
  correlation with limited feature set; see \code{\link{deleteGPsep}} and
  \code{\link{deleteGPseps}}

  \code{updateGP} takes \code{gpi} identifier as
  input and augments that GP with new data.  A sequence of updates is
  performed, for each \code{i in 1:nrow(X)}, each taking time which is
  quadratic in the number of data points. 
  \code{updateGP} also updates any statistics needed in order to quickly
  search for new local design candidates via \code{\link{laGP}}.
  The same applies to \code{updateGPsep} on \code{gpsepi} objects
}
\value{
 \code{newGP} and \code{newGPsep} create 
 a unique GP indicator (\code{gpi} or \code{gpsepi}) referencing a C-side object;
  \code{updateGP} and \code{updateGPsep} do not return anything, but yields a
  modified C-side object as a side effect }
\references{
  For standard GP inference, refer to any graduate text, e.g., Rasmussen
  & Williams \emph{Gaussian Processes for Machine Learning}. \cr

  For efficient updates of GPs, see: \cr
  
  R.B. Gramacy and D.W. Apley (2015).
  \emph{Local Gaussian process approximation for large computer
    experiments.} Journal of Computational and Graphical Statistics, 
  24(2), pp. 561-678; preprint on arXiv:1303.0383;
  \url{http://arxiv.org/abs/1303.0383}
}
\author{
  Robert B. Gramacy \email{rbg@vt.edu}
}

\seealso{
\code{vignette("laGP")}, 
\code{\link{deleteGP}}, \code{\link{mleGP}}, \code{\link{predGP}}, \code{\link{laGP}} 
}

\examples{
## for more examples, see predGP and mleGP docs

## simple sine data
X <- matrix(seq(0,2*pi,length=7), ncol=1)
Z <- sin(X)

## new GP fit
gpi <- newGP(X, Z, 2, 0.000001)

## make predictions
XX <- matrix(seq(-1,2*pi+1, length=499), ncol=ncol(X))
p <- predGP(gpi, XX)

## sample from the predictive distribution
library(mvtnorm)
N <- 100
ZZ <- rmvt(N, p$Sigma, p$df) 
ZZ <- ZZ + t(matrix(rep(p$mean, N), ncol=N))
matplot(XX, t(ZZ), col="gray", lwd=0.5, lty=1, type="l", 
       xlab="x", ylab="f-hat(x)", bty="n")
points(X, Z, pch=19, cex=2)

## update with four more points
X2 <- matrix(c(pi/2, 3*pi/2, -0.5, 2*pi+0.5), ncol=1)
Z2 <- sin(X2)
updateGP(gpi, X2, Z2)

## make a new set of predictions
p2 <- predGP(gpi, XX)
ZZ <- rmvt(N, p2$Sigma, p2$df) 
ZZ <- ZZ + t(matrix(rep(p2$mean, N), ncol=N))
matplot(XX, t(ZZ), col="gray", lwd=0.5, lty=1, type="l", 
       xlab="x", ylab="f-hat(x)", bty="n")
points(X, Z, pch=19, cex=2)
points(X2, Z2, pch=19, cex=2, col=2)

## clean up
deleteGP(gpi)
}

\keyword{ nonparametric }
\keyword{ nonlinear }
\keyword{ smooth }
\keyword{ models }
\keyword{ regression }
\keyword{ spatial }
