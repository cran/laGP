\name{deleteGP}
\alias{deleteGP}
\alias{deleteGPs}
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
deleteGPs()
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{gpi}{
    a scalar positive integer specifying an allocated GP object
}
}
\details{
  Any function calling \code{\link{newGP}} will require destruction
  via these functions or there will be a memory leak.
}
\value{
  Nothing is returned.
}

\author{
  Robert B. Gramacy \email{rbgramacy@chicagobooth.edu}
}

\seealso{
  \code{\link{newGP}}, \code{\link{predGP}}, \code{\link{mleGP}}
}
\examples{
## see examples for newGP, predGP, or mleGP
}

\keyword{ utilities }
