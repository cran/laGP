\name{deleteGP}
\alias{deleteGP}
\alias{deleteGPs}
\alias{deleteGPsep}
\alias{deleteGPseps}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Delete C-side Gaussian Process Objects
}
\description{
  Frees memory allocated by a particular C-side Gaussian process
  object, or all GP objects currently allocated
}
\usage{
deleteGP(gpi)
deleteGPsep(gpsepi)
deleteGPs()
deleteGPseps()
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{gpi}{
    a scalar positive integer specifying an allocated isotropic GP object
  }
  \item{gpsepi}{ similar to \code{gpi} but indicating a separable GP object}
}
\details{
  Any function calling \code{\link{newGP}} or \code{\link{newGPsep}}
  will require destruction
  via these functions or there will be a memory leak.
}
\value{
  Nothing is returned.
}

\author{
  Robert B. Gramacy \email{rbg@vt.edu}
}

\seealso{
\code{vignette("laGP")}, 
  \code{\link{newGP}}, \code{\link{predGP}}, \code{\link{mleGP}} 
}
\examples{
## see examples for newGP, predGP, or mleGP
}

\keyword{ utilities }
