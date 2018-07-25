\name{predGP}
\alias{predGP}
\alias{predGPsep}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
GP Prediction/Kriging
}
\description{
  Perform Gaussian processes prediction (under isotropic or separable formulation)
  at new \code{XX} locations using a GP object stored on the C-side
}
\usage{
predGP(gpi, XX, lite = FALSE, nonug = FALSE)
predGPsep(gpsepi, XX, lite = FALSE, nonug = FALSE)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{gpi}{
    a C-side GP object identifier (positive integer);
    e.g., as returned by \code{\link{newGP}}
}
\item{gpsepi}{ similar to \code{gpi} but indicating a separable GP object,
as returned by \code{\link{newGPsep}}}
\item{XX}{
   a \code{matrix} or \code{data.frame} containing
   a design of predictive locations
}
  \item{lite}{
    a scalar logical indicating whether (\code{lite = FALSE}, default) or not
    (\code{lite = TRUE}) a full predictive covariance matrix should be
    returned, as would be required for plotting random sample paths,
    but substantially increasing computation time if only point-prediction
    is required
}
\item{nonug}{ a scalar logical indicating if a (nonzero) nugget should be used in the predictive
equations; this allows the user to toggle between visualizations of uncertainty due just
to the mean, and a full quantification of predictive uncertainty. The latter (default \code{nonug = FALSE}) 
is the standard approach, but the former may work better in some sequential design contexts.  See, e.g.,
\code{\link{ieciGP}}}
}
\details{
  Returns the parameters of Student-t predictive equations.  By
  default, these include a full predictive covariance matrix between all
  \code{XX} locations.  However, this can be slow when \code{nrow(XX)}
  is large, so a \code{lite} options is provided, which only returns the
  diagonal of that matrix.

  GP prediction is sometimes called \dQuote{kriging}, especially in
  the spatial statistics literature.  So this function could
  also be described as returning evaluations of the \dQuote{kriging equations}
%%  ~~ If necessary, more details than the description above ~~
}
\value{
  The output is a \code{list} with the following components.
  
  \item{mean }{a vector of predictive means of length \code{nrow(Xref)}}
  \item{Sigma}{ covariance matrix of
    for a multivariate Student-t distribution; alternately
     if \code{lite = TRUE},
    then a field \code{s2} contains the diagonal of this matrix}
  \item{df }{a Student-t degrees of freedom scalar (applies to all
  \code{XX})}
}
\references{
  For standard GP prediction, refer to any graduate text, e.g., Rasmussen
  & Williams \emph{Gaussian Processes for Machine Learning}
}
\author{
    Robert B. Gramacy \email{rbg@vt.edu}
}

\seealso{
\code{vignette("laGP")}, 
  \code{\link{newGP}}, \code{\link{mleGP}}, \code{\link{jmleGP}}, 
}
\examples{
## a "computer experiment" -- a much smaller version than the one shown
## in the aGP docs

## Simple 2-d test function used in Gramacy & Apley (2015);
## thanks to Lee, Gramacy, Taddy, and others who have used it before
f2d <- function(x, y=NULL)
  {
    if(is.null(y)) {
      if(!is.matrix(x) && !is.data.frame(x)) x <- matrix(x, ncol=2)
      y <- x[,2]; x <- x[,1]
    }
    g <- function(z)
      return(exp(-(z-1)^2) + exp(-0.8*(z+1)^2) - 0.05*sin(8*(z+0.1)))
    z <- -g(x)*g(y)
  }

## design with N=441
x <- seq(-2, 2, length=11)
X <- expand.grid(x, x)
Z <- f2d(X)

## fit a GP
gpi <- newGP(X, Z, d=0.35, g=1/1000)

## predictive grid with NN=400
xx <- seq(-1.9, 1.9, length=20)
XX <- expand.grid(xx, xx)
ZZ <- f2d(XX)

## predict
p <- predGP(gpi, XX, lite=TRUE)
## RMSE: compare to similar experiment in aGP docs
sqrt(mean((p$mean - ZZ)^2))

## visualize the result
par(mfrow=c(1,2))
image(xx, xx, matrix(p$mean, nrow=length(xx)), col=heat.colors(128),
      xlab="x1", ylab="x2", main="predictive mean")
image(xx, xx, matrix(p$mean-ZZ, nrow=length(xx)), col=heat.colors(128),
      xlab="x1", ylab="x2", main="bas")

## clean up
deleteGP(gpi)

## see the newGP and mleGP docs for examples using lite = FALSE for
## sampling from the joint predictive distribution
}

\keyword{ nonparametric }
\keyword{ nonlinear }
\keyword{ smooth }
\keyword{ models }
\keyword{ regression }
\keyword{ spatial }
