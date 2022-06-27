\name{llikGP}
\alias{llikGP}
\alias{llikGPsep}

\title{
	Calculate a GP log likelihood
}
\description{
	Calculate a Gaussian process (GP) log likelihood or posterior
	probability with reference to a C-side GP object
}
\usage{
llikGP(gpi, dab = c(0, 0), gab = c(0, 0))
llikGPsep(gpsepi, dab = c(0, 0), gab = c(0, 0))
}

\arguments{
\item{gpi}{
   a C-side GP object identifier (positive integer);
    e.g., as returned by \code{\link{newGP}}}
\item{gpsepi}{ similar to \code{gpi} but indicating a separable GP object}

\item{dab}{
  \code{ab} for the lengthscale parameter, see Details
}
\item{gab}{
  \code{ab} for the nugget parameter, see Details
}
}
\details{

  An \dQuote{\code{ab}} parameter is a non-negative 2-vector describing 
  shape and rate parameters to a Gamma prior; a zero-setting for
  either value results in no-prior being used in which case a log likelihood
  is returned.  If both \code{ab} parameters are specified, then the value
  returned can be interpreted as a log posterior density.  See \code{\link{darg}}
  for more information about \code{ab}

}
\value{
	A real-valued scalar is returned.
}

\author{
  Robert B. Gramacy \email{rbg@vt.edu}
}

\seealso{
\code{\link{mleGP}}, \code{\link{darg}}
}
\examples{
## partly following the example in mleGP
if(require("MASS")) {

  ## motorcycle data and predictive locations
  X <- matrix(mcycle[,1], ncol=1)
  Z <- mcycle[,2]

  ## get sensible ranges
  d <- darg(NULL, X)
  g <- garg(list(mle=TRUE), Z)
  
  ## initialize the model
  gpi <- newGP(X, Z, d=d$start, g=g$start)

  ## calculate log likelihood
  llikGP(gpi)
  ## calculate posterior probability
  llikGP(gpi, d$ab, g$ab)

  ## clean up
  deleteGP(gpi)
}
}

\keyword{ nonparametric }
\keyword{ nonlinear }
\keyword{ smooth }
\keyword{ models }
\keyword{ regression }
\keyword{ spatial }
