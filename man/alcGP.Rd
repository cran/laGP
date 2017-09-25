\name{alcGP}
\alias{alcGP}
\alias{alcGPsep}
\alias{alcrayGP}
\alias{alcrayGPsep}
\alias{mspeGP}
\alias{fishGP}
\alias{ieciGP}
\alias{ieciGPsep}
\alias{alcoptGP}
\alias{dalcGP}
\alias{alcoptGPsep}
\alias{dalcGPsep}

\title{
  Improvement statistics for sequential or local design
}

\description{
  Calculate the active learning Cohn (ALC) statistic, mean-squared predictive
  error (MSPE) or expected Fisher information (fish) for a Gaussian
  process (GP) predictor relative to a set of reference locations, towards
  sequential design or local search for Gaussian process regression
}

\usage{
alcGP(gpi, Xcand, Xref = Xcand, parallel = c("none", "omp", "gpu"), 
      verb = 0)
alcGPsep(gpsepi, Xcand, Xref = Xcand, parallel = c("none", "omp", "gpu"), 
      verb = 0)
alcrayGP(gpi, Xref, Xstart, Xend, verb = 0)
alcrayGPsep(gpsepi, Xref, Xstart, Xend, verb = 0)
ieciGP(gpi, Xcand, fmin, Xref = Xcand, w = NULL, verb = 0)
ieciGPsep(gpsepi, Xcand, fmin, Xref = Xcand, w = NULL, verb = 0)
mspeGP(gpi, Xcand, Xref = Xcand, fi = TRUE, verb = 0)
fishGP(gpi, Xcand)
alcoptGP(gpi, Xref, start, lower, upper, maxit = 100, verb = 0)
alcoptGPsep(gpsepi, Xref, start, lower, upper, maxit = 100, verb = 0)
dalcGP(gpi, Xcand, Xref = Xcand, verb = 0)
dalcGPsep(gpsepi, Xcand, Xref = Xcand, verb = 0)
}

%- maybe also 'usage' for other objects documented here.
\arguments{
\item{gpi}{
   a C-side GP object identifier (positive integer);
    e.g., as returned by \code{\link{newGP}}
    }
\item{gpsepi}{
  a C-side separable GP object identifier (positive integer);
    e.g., as returned by \code{\link{newGPsep}}
}
\item{Xcand}{
   a \code{matrix} or \code{data.frame} containing
   a design of candidate predictive locations at which the ALC
   (or other) criteria is (are) evaluated.  In the context of
   \code{\link{laGP}}, these are the possible locations for adding
   into the current local design
}
\item{fmin}{ for \code{ieci*} only: a scalar value indicating the 
  value of the best minimum found so far.  This is usually set to 
  the minimum of the \code{Z}-values stored in the \code{gpi} or 
  \code{gpsepi} reference (for deterministic/low nugget settings),
  or otherwise the predicted mean value at the \code{X} locations  
}
\item{Xref}{
   a \code{matrix} or \code{data.frame} containing a design of
   reference locations for ALC or MSPE.  I.e., these are the locations
   at which the reduction in variance, or mean squared predictive error,
   are calculated.  In the context of \code{\link{laGP}}, this is
   the single location \code{Xref = x} around which a local design 
   is sought.  For \code{alcrayGP} and \code{alcrayGPsep} the 
   \code{matrix} may only have one row, i.e., one reference location
}
\item{parallel}{
  a switch indicating if any parallel calculation of 
  the criteria (\code{method}) is desired.    
  For \code{parallel = "omp"}, the package be compiled with OpenMP flags; 
  for \code{parallel = "gpu"}, the package must be compiled with CUDA
  flags (only the ALC criteria is supported on the GPU); see README/INSTALL
  in the package source for more details
}
\item{Xstart}{
  a \code{1}-by-\code{ncol(Xref)} starting location for a search along
  a ray between \code{Xstart} and \code{Xend}
}
\item{Xend}{
  a \code{1}-by-\code{ncol(Xref)} ending location for a search along
  a ray between \code{Xstart} and \code{Xend}
}
\item{fi}{
  a scalar logical indicating if the expected Fisher information portion
  of the expression (MSPE is essentially \code{ALC + c(x)*EFI}) should be
  calculated (\code{TRUE}) or set to zero (\code{FALSE}).  This flag is mostly
  for error checking against the other functions, \code{alcGP} and \code{fishGP},
  since the constituent parts are separately available via those functions
}
\item{w}{ weights on the reference locations \code{Xref} for IECI calculations;
  IECI is not fully documented at this time}
\item{verb}{
  a non-negative integer specifying the verbosity level; \code{verb = 0}
  is quiet, and larger values cause more progress information to be
  printed to the screen
}
\item{start}{ initial values to the derivative-based search via \code{"L-BFGS-B"} 
  within \code{alcoptGP} and \code{alcoptGPsep}; a nearest neighbor often
  represents a sensible initialization }
\item{lower, upper}{ bounds on the derivative-based search via \code{"L-BFGS-B"}
  within \code{alcoptGP} and \code{alcoptGPsep}}
\item{maxit}{ the maximum number of iterations (default \code{maxit=100}) in \code{"L-BFGS-B"}
  search within \code{alcoptGP} and \code{alcoptGPsep}}
}

\details{
  The best way to see how these functions are used in the context of local
  approximation is to inspect the code in the \code{\link{laGP.R}} function.

  Otherwise they are pretty self-explanatory.  They evaluate the ALC, MSPE,
  and EFI quantities outlined in Gramacy & Apley (2015).  The ALC is originally
  due to Seo, et al. (2000).  The ray-based search is described by Gramacy & Haaland (2015).

  MSPE and EFI calculations are not supported for separable GP models, i.e.,
  there are no \code{mspeGPsep} or \code{fishGPsep} functions.

  \code{alcrayGP} and \code{alcrayGPsep} allows only one reference location 
  (\code{nrow(Xref) = 1}).  \code{alcoptGP} and \code{alcoptGPsep} allows multiple 
  reference locations. It optimizes a continuous ALC analog in its natural logarithm form 
  via log(derivatives), using the starting locations, bounding boxes and (stored) gp provided, 
  and finally snaps the solution back to the candidate grid.  The derivative-based search is
  described by Sun, et al. (2017).
  
  Note that \code{ieciGP} and \code{ieciGPsep}, which are for optimization via 
  integrated expected conditional improvement (Gramacy & Lee, 2011) are 
  alpha functionality and are not fully documented at this time.
  
  \code{dalcGP} and \code{dalcGPsep} calculate the derivative, with respect to \code{Xcand}, of the
  \eqn{s2'}{s2'} component of the ALC calculation of the expected reduction in variance calculation 
  at locations \code{Xcand} averaging over reference locations \code{Xref}: \eqn{ds2 = s2 - s2'}{ds2 = s2 - s2'}, where 
  the \eqn{s2}{s2} are at \code{Xref} and the \eqn{s2'}{s2'} incorporates \code{Xcand}, and everything is averaged over \code{Xref}.
}

\value{
Except for \code{alcoptGP}, \code{alcoptGPsep}, \code{dalcGP}, and \code{dalcGPsep}, a vector of length \code{nrow(Xcand)} is returned 
filled with values corresponding to the desired statistic
\item{par}{the best set of parameters found}
\item{its}{a two-element integer vector giving the number of calls to the object function and the gradient respectively.}
\item{msg}{a character string giving any additional information returned by the optimizer, or NULL}
\item{convergence}{An integer code. \code{0} indicates successful completion. For the other error codes,
                   see the documentation for \code{\link{optim}}}
\item{alcs}{the predictive variance with the augmented local design by the new location averaging over the reference locations}
\item{dalcs}{the derivative of \code{alcs} with respect to the new location}
}

\references{ 

  F. Sun, R.B. Gramacy, B. Haaland, E. Lawrence, and A. Walker (2017)
  \emph{Emulating satellite drag from large simulation experiments.};
  preprint on arXiv coming soon. 

  R.B. Gramacy (2016). \emph{\pkg{laGP}: Large-Scale Spatial Modeling via 
  Local Approximate Gaussian Processes in \R.}, Journal of Statistical 
  Software, 72(1), 1-46; or see \code{vignette("laGP")}
  
  R.B. Gramacy and B. Haaland (2016).
  \emph{Speeding up neighborhood search in local Gaussian process prediction.}
  Technometrics, 58(3), pp. 294-303;
  preprint on arXiv:1409.0074 
  \url{http://arxiv.org/abs/1409.0074}

  R.B. Gramacy and D.W. Apley (2015).
  \emph{Local Gaussian process approximation for large computer
    experiments.} Journal of Computational and Graphical Statistics, 
  24(2), pp. 561-678; preprint on arXiv:1303.0383;
  \url{http://arxiv.org/abs/1303.0383}

  R.B. Gramacy, J. Niemi, R.M. Weiss (2014).
  \emph{Massively parallel approximate Gaussian process regression.}
  SIAM/ASA Journal on Uncertainty Quantification, 2(1), pp. 568-584;
  preprint on arXiv:1310.5182;
  \url{http://arxiv.org/abs/1310.5182}

  R.B. Gramacy, H.K.H. Lee (2011). 
  \emph{Optimization under unknown constraints.}, Valencia discussion paper, in Bayesian Statistics 9. 
  Oxford University Press; preprint on arXiv:1004.4027; \url{http://arxiv.org/abs/1004.4027}

  Seo, S., Wallat, M., Graepel, T., Obermayer, K. (2000). 
  \emph{Gaussian Process Regression: Active Data Selection and Test Point Rejection.}
  In Proceedings of the International Joint Conference on Neural Networks, 
  vol. III, 241-246. IEEE.
}

\author{
  Robert B. Gramacy \email{rbg@vt.edu} and Furong Sun \email{furongs@vt.edu}
}

\seealso{
  \code{\link{laGP}}, \code{\link{aGP}}, \code{\link{predGP}}
}

\examples{
## this follows the example in predGP, but only evaluates 
## information statistics documented here

## Simple 2-d test function used in Gramacy & Apley (2015);
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
fish <- fishGP(gpi, XX)

## visualize the result
par(mfrow=c(1,3))
image(xx, xx, matrix(sqrt(alc), nrow=length(xx)), col=heat.colors(128),
      xlab="x1", ylab="x2", main="sqrt ALC")
image(xx, xx, matrix(sqrt(mspe), nrow=length(xx)), col=heat.colors(128),
      xlab="x1", ylab="x2", main="sqrt MSPE")
image(xx, xx, matrix(log(fish), nrow=length(xx)), col=heat.colors(128),
      xlab="x1", ylab="x2", main="log fish")

## clean up
deleteGP(gpi)

## search the new location to augment the current local design
xx <- seq(-1.9, 1.9, length=10)
XX <- as.matrix(expand.grid(xx, xx))
e.alc <- laGP.R(XX, start=6, end=100, X, Z, method="alc", close=10000, parallel="omp")
XC <- X[e.alc$Xi[1:6],] # the initial local design containing 6 NNs
ZC <- Z[e.alc$Xi[1:6]]  # the corresponding responses
Xcan <- data.frame(X[-e.alc$Xi[1:6],]) # the candidate input space
gpi <- newGP(XC, ZC, d=e.alc$mle$d, g=e.alc$g$start)
out1 <- alcoptGP(gpi=gpi, Xref=XX, start=c(0,0), lower=range(x)[1], upper=range(x)[2])
## out1$par is the "new" location
D <- t(distance(matrix(out1$par, ncol=ncol(XC)), as.matrix(Xcan))) 
D.frame <- data.frame(cbind(Xcan, D, as.numeric(row.names(Xcan))))
names(D.frame) <- c(paste("x", 1:ncol(XC), sep=""), "distance", "lab")
D.label <- which.min(D.frame$distance) # snap the "new" location back to the global input space
lab <- as.numeric(D.frame[D.label, (ncol(XC)+2)])
xnew <- data.frame(X)[lab,] # the new location to augment the current local design
out2 <- dalcGP(gpi=gpi, Xcand=as.matrix(xnew), Xref=XX)

## Below is the predictive variance with the local design augmented 
## by the new location averaging over the reference locations
ALC <- out2$alcs 

## clean up
deleteGP(gpi)
}

\keyword{ nonparametric }
\keyword{ nonlinear }
\keyword{ smooth }
\keyword{ models }
\keyword{ regression }
\keyword{ spatial }
