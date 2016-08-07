\name{distance}
\alias{distance}

\title{
    Calculate Euclidean distance between pairs of points
}
\description{
    Calculate the Euclidean distance between pairs of points
    and return a distance matrix
}
\usage{
distance(X1, X2 = NULL)
}

\arguments{
  \item{X1}{
    a \code{matrix} or \code{data.frame} containing real-valued
    numbers
}
  \item{X2}{
  an optional \code{matrix} or \code{data.frame} containing real-valued
  numbers; must have \code{ncol(X2) = ncol(X1)}
}
}
\details{
    If \code{X2 = NULL} distances between \code{X1} and itself are
    calculated, resulting in an \code{nrow(X1)}-by-\code{nrow(X1)} distance 
    matrix.  Otherwise the result is \code{nrow(X1)}-by-\code{nrow(X2)} and
    contains distances between \code{X1} and \code{X2}.

    Calling \code{distance(X)} is the same as \code{distance(X,X)}
}
\value{
    The output is a \code{matrix}, whose dimensions are described in the Details
    section above
}

\author{
  Robert B. Gramacy \email{rbg@vt.edu}
}

%% ~Make other sections like Warning with \section{Warning }{....} ~

\seealso{
    \code{\link{darg}}
}
\examples{
x <- seq(-2, 2, length=11)
X <- as.matrix(expand.grid(x, x))
## predictive grid with NN=400
xx <- seq(-1.9, 1.9, length=20)
XX <- as.matrix(expand.grid(xx, xx))

D <- distance(X)
DD <- distance(X, XX)
}

\keyword{ utilities }
