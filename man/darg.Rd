\name{darg}
\alias{darg}
\alias{garg}

\title{
  Generate Priors for GP correlation
}
\description{
  Generate empirical Bayes regularization (priors) and choose initial values
   and ranges for (isotropic) lengthscale and nugget parameters to a Gaussian
   correlation function for a GP regression model }
\usage{
darg(d, X, samp.size = 1000)
garg(g, y)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
\item{d}{
    can be \code{NULL}, or a scalar indicating an initial value
    or a partial \code{list} whose format matches the one described
    in the Value section below
  }
\item{g}{
    can be \code{NULL}, or a scalar indicating an initial value
    or a partial \code{list} whose format matches the one described
    in the Value section below
}
\item{X}{
    a \code{matrix} or \code{data.frame} containing
    the full (large) design matrix of input locations
  }
\item{y}{
    a vector of responses/dependent values
}
  \item{samp.size}{
    a scalar integer indicating a subset size of \code{X} to use
    for calculations; this is important for very large \code{X} matrices
    since the calculations are quadratic in \code{nrow(X)}
}
}
\details{
  These functions use aspects of the data, either \code{X} or \code{y},
  to form weakly informative default priors, and choose initial values,
  for a lengthscale and nugget parameter.  This is useful since the
  likelihood can sometimes be very flat, and even with proper priors
  inference can be very sensitive to the specification of those priors
  and any initial search values.  The focus is on avoiding pathologies
  while otherwise remaining true to the spirit of MLE calculation.  

  \code{darg} output specifies MLE inference (\code{out$mle = TRUE})
  by default, whereas \code{garg} instead fixes the nugget at the starting value,
  which may be sensible for emulating deterministic computer simulation data; 
  when \code{out$mle =  FALSE} the calculated range outputs \code{c(out$min, out$max)} 
  are set to dummy values that are ignored in other parts of the \pkg{laGP} package.

  \code{darg} calculates a Gaussian distance matrix between all pairs of
  \code{X} rows, or a subsample of rows of size \code{samp.size}.  From
  those distances it chooses the range and start values from the range
  of (non-zero) distances and the \code{0.1} quantile, respectively.
  The Gamma prior values have a shape of \code{out$a = 3/2} and a rate
  \code{out$b} chosen by the incomplete Gamma inverse function to put
  \code{0.95} probability below \code{out$max}. 

  \code{garg} is similar except that it works with \code{(y- mean(y))^2}
  instead of the pairwise distances of \code{darg}.  The only difference
  is that the starting value is chosen as the 2.5\% quantile.
}
\value{
  Both functions return a list containing the following entries.  If the
  input object (\code{d} or \code{g}) specifies one of the values then
  that value is copied to the same list entry on output.  See the
  Details section for how these values are calculated
  
 \item{mle }{ by default, \code{TRUE} for \code{darg} and \code{FALSE}
   for \code{garg} }
 \item{start }{ starting value chosen from the quantiles of
   \code{distance(X)} or \code{(y - mean(y))^2)}}
 \item{min }{ minimum value in allowable range for the parameter - for
   future inference purposes }
 \item{max }{ maximum value in allowable range for the parameter - for
   future inference purposes }
 \item{ab }{ shape and rate parameters specifying a Gamma prior for the parameter}

}

\author{
    Robert B. Gramacy \email{rbgramacy@chicagobooth.edu}
}


\seealso{
\code{vignette("laGP")}, 
  \code{\link{laGP}}, \code{\link{aGP}}, \code{\link{laGP}},
  \code{\link{mleGP}}, \code{\link{distance}}, \code{\link{llikGP}}
}
\examples{
## motorcycle data
library(MASS)
X <- matrix(mcycle[,1], ncol=1)
Z <- mcycle[,2]

## get darg and garg
darg(NULL, X)
garg(list(mle=TRUE), Z)
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
