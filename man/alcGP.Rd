\name{alcGP}
\alias{alcGP}
\alias{alcrayGP}
\alias{mspeGP}
\alias{efiGP}

\title{
  Improvement statistics for sequential or local design
}
\description{
  Calculate the active learning Cohn (ALC) statistic, mean-squared predictive
  error (MSPE) or expected Fisher information (EFI) for a Gaussian
  process (GP) predictor relative to a set of reference locations, towards
  sequential design or local search for Gaussian process regression
}
\usage{
alcGP(gpi, Xcand, Xref = Xcand, parallel = c("none", "omp", "gpu"), 
      verb = 0)
alcrayGP(gpi, Xref, Xstart, Xend, verb=0)
mspeGP(gpi, Xcand, Xref = Xcand, fi = TRUE, verb = 0)
efiGP(gpi, Xcand)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
\item{gpi}{
   a C-side GP object identifier (positive integer);
    e.g., as returned by \code{\link{newGP}}; note that separable GP objects
    (created by \code{\link{newGPsep}} are not supported at this time)
}
\item{Xcand}{
   a \code{matrix} or \code{data.frame} containing
   a design of candidate predictive locations at which the ALC
   (or other) criteria is (are) evaluated.  In the context of
   \code{\link{laGP}}, these are the possible locations for adding
   into the current local design
}
\item{Xref}{
   a \code{matrix} or \code{data.frame} containing a design of
   reference locations for ALC or MSPE.  I.e., these are the locations
   at which the reduction in variance, or mean squared predictive error,
   are calculated.  In the context of \code{\link{laGP}}, this is
   the single location \code{Xref = x} around which a local design 
   is sought.  For \code{alcray} the \code{matrix} may only have one
   row, i.e., one reference location
}
\item{parallel}{
   a switch indicating if any parallel calculation of 
  the criteria (\code{method}) is desired.    
  For \code{parallel = "omp"}, the package be compiled with OpenMP flags; 
  for \code{parallel = "gpu"}, the package must be compiled with CUDA
  flags (only the ALC criteria is supported on the GPU); see README/INSTALL
  in the package source for more details;  at this time this option is
  only supported for \code{method = "alc"} criteria
}
\item{Xstart}{
  a \code{1}-by-\code{ncol(Xref)} starting location for a search along
  a ray between \code{Xstart} and \code{Xend}
}
\item{Xend}{
  a \code{1}-by-\code{ncol(Xref)} starting location for a search along
  a ray between \code{Xstart} and \code{Xend}
}
\item{fi}{
  a scalar logical indicating if the expected Fisher information portion
  of the expression (MSPE is essentially \code{ALC + c(x)*EFI}) should be
  calculated (\code{TRUE}) or set to zero (\code{FALSE}).  This flag is mostly
  for error checking against the other functions, \code{alcGP} and \code{efiGP},
  since the constituent parts are separately available via those functions
}
\item{verb}{
  a positive integer specifying the verbosity level; \code{verb = 0}
  is quiet, and larger values cause more progress information to be
  printed to the screen
}
}
\details{
  The best way to see how these functions are used in the context of local
  approximation is to inspect the code in the \code{\link{laGP.R}} function.

  Otherwise they are pretty self explanatory.  They evaluate the ALC, MSPE,
  and EFI quantities outlined in Gramacy & Apley (2014).  The ALC is originally
  due to Seo, et al. (2000).  The ray-based search is described by a forthcoming
  paper by Gramacy & Haaland (2014).

  \code{alcrayGP} allows only one reference location (\code{nrow(Xref) = 1}).  
}
\value{
  A vector of length \code{nrow(Xcand)} is returned filled with values
  corresponding to the desired statistic
}
\references{
  R.B. Gramacy and D.W. Apley (2014).
  \emph{Local Gaussian process approximation for large computer
    experiments.} Journal of Computational and Graphical Statistics, \emph{to appear};
  preprint available on arXiv:1303.0383; 
  \url{http://arxiv.org/abs/1303.0383}

  R.B. Gramacy, J. Niemi, R.M. Weiss (2014). 
  \emph{Massively parallel approximate Gaussian process regression.}
  SIAM/ASA Journal on Uncertainty Quantification, \emph{to appear};
  preprint on arXiv:1310.5182
  \url{http://arxiv.org/abs/1310.5182}  

  R.B. Gramacy and B. Haaland (2014).
  \emph{Speeding up neighborhood search in local Gaussian process prediction.}
  Preprint on arXiv:1409.0074 
  \url{http://arxiv.org/abs/1409.0074}

  Seo, S., Wallat, M., Graepel, T., and Obermayer, K. (2000). 
  \emph{Gaussian Process Regression: Active Data Selection and Test Point Rejection.}
  In Proceedings of the International Joint Conference on Neural Networks, 
  vol. III, 241-246. IEEE.
}
\author{
  Robert B. Gramacy \email{rbgramacy@chicagobooth.edu}
}

\seealso{
  \code{\link{laGP}}, \code{\link{aGP}}, \code{\link{predGP}}
}
\note{ Note that separable GP objects (i.e., created by \code{\link{newGPsep}} 
are not supported at this time}
\examples{
## this follows the example in predGP, but only evaluates 
## information statistics documented here

## Simple 2-d test function used in Gramacy & Apley (2014);
## thanks to Lee, Gramacy, Taddy, and others who have used it before
f2d <- function(x, y=NULL)
  {
    if(is.null(y)) {
      if(!is.matrix(x)) x <- matrix(x, ncol=2)
      y <- x[,2]; x <- x[,1]
    }
    g <- function(z)
      return(exp(-(z-1)^2) + exp(-0.8*(z+1)^2) - 0.05*sin(8*(z+0.1)))
    z <- -g(x)*g(y)
  }

## design with N=441
x <- seq(-2, 2, length=11)
X <- as.matrix(expand.grid(x, x))
Z <- f2d(X)

## fit a GP
gpi <- newGP(X, Z, d=0.35, g=1/1000, dK=TRUE)

## predictive grid with NN=400
xx <- seq(-1.9, 1.9, length=20)
XX <- as.matrix(expand.grid(xx, xx))

## predict
alc <- alcGP(gpi, XX)
mspe <- mspeGP(gpi, XX)
efi <- efiGP(gpi, XX)

## visualize the result
par(mfrow=c(1,3))
image(xx, xx, matrix(sqrt(alc), nrow=length(xx)), col=heat.colors(128),
      xlab="x1", ylab="x2", main="sqrt ALC")
image(xx, xx, matrix(sqrt(mspe), nrow=length(xx)), col=heat.colors(128),
      xlab="x1", ylab="x2", main="sqrt MSPE")
image(xx, xx, matrix(log(efi), nrow=length(xx)), col=heat.colors(128),
      xlab="x1", ylab="x2", main="log EFI")

## clean up
deleteGP(gpi)
}

\keyword{ nonparametric }
\keyword{ nonlinear }
\keyword{ smooth }
\keyword{ models }
\keyword{ regression }
\keyword{ spatial }