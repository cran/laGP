\name{laGP}
\alias{laGP}
\alias{laGP.R}

\title{
Localized Approximate GP Regression At a Single Predictive Location
}
\description{
  Build a sub-design of \code{X} of size \code{end} for approximate
  Gaussian process prediction at reference location(s) \code{Xref}, and then
  return the moments of those predictive equations
}
\usage{
laGP(Xref, start, end, X, Z, d = NULL, g = 1/1000,
     method = c("alc", "alcray", "mspe", "nn", "efi"), Xi.ret = TRUE,
        close = min(1000, nrow(X)), alc.gpu = FALSE, rect = NULL, 
        verb = 0)
laGP.R(Xref, start, end, X, Z, d = NULL, g = 1/1000,
       method = c("alc", "alcray", "mspe", "nn", "efi"), Xi.ret = TRUE,
       pall = FALSE, close = min(1000, nrow(X)),  
       parallel = c("none", "omp", "gpu"), rect = NULL, verb = 0)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{Xref}{
    a vector of length \code{ncol(X)} containing a single reference
    location; or a \code{matrix} with \code{ncol(Xref) = ncol(X)} containing
    multiple reference locations for simultaneous sub-design and
    prediction
}
  \item{start}{
    the number of Nearest Neighbor (NN) locations for initialization; must
    specify \code{start >= 6}
}
  \item{end}{
    the total size of the local designs; must have \code{start < end}
}
  \item{X}{
    a \code{matrix} or \code{data.frame} containing
    the full (large) design matrix of input locations
}
  \item{Z}{
    a vector of responses/dependent values with \code{length(Z) = ncol(X)}
}
  \item{d}{
    a prior or initial setting for the (single/isotropic) lengthscale
    parameter for a Gaussian correlation function; a (default)
    \code{NULL} value causes a sensible regularization (prior) and
    initial setting to be generated via \code{\link{darg}};
    a scalar specifies an initial value, causing \code{\link{darg}}
    to only generate the prior; otherwise, a list or partial list matching the output
    of \code{\link{darg}} can be used to specify a custom prior.  In the
    case of a partial list, the only the missing entries will be
    generated. Note that a default/generated list specifies MLE/MAP
    inference for this parameter
}
\item{g}{
    a prior or initial setting for the nugget parameter; a 
    \code{NULL} value causes a sensible regularization (prior) and
    initial setting to be generated via \code{\link{garg}}; a scalar
    (default \code{g = 1/1000}) specifies an initial value, causing \code{\link{garg}}
    to only generate the prior; otherwise, a
    list or partial list matching the output of \code{\link{garg}} can be used to
    specify a custom prior.  In the case of a partial list, the only the
    missing entries will be generated. Note that a default/generated list
    specifies \emph{no} inference for this parameter; i.e., it is fixed
    at its starting value
}
\item{method}{
    Specifies the method by which \code{end-start} candidates from
    \code{X} are chosen in order to predict at \code{Xref}. In brief, ALC
    (\code{"alc"}, default) minimizes predictive variance; ALCRAY 
    (\code{"alcray")}) executes a thrify search focused on rays eminating
    out of the reference location(s); MSPE (\code{"mspe"}) augments that 
    with extra derivative information to minimize mean-squared prediction 
    error (requires extra computation);
    NN (\code{"nn"}) uses nearest neighbor; and EFI (\code{"efi"}) uses
    the expected Fisher information - essentially \code{1/G} from
    Gramacy & Apley (2013) - which is global heuristic, i.e., not
    localized to \code{Xref}
}
  \item{Xi.ret}{
    A scalar logical indicating whether or not a vector of indices
    into \code{X}, specifying the chosen sub-design, should be returned on
    output
  }
  \item{pall}{a scalar logical (for \code{laGP.R} only) offering the
    ability to obtain predictions after every update (for progress
    indication and debugging), rather than after just the last update}
  \item{close}{
    a non-negative integer specifying the number of NNs
    (to \code{Xref}) in \code{X} to consider when
    searching for the sub-design; \code{close = 0} specifies all
  }
  \item{alc.gpu}{
    a scalar \code{logical} indicating if a GPU should be used to
    parallelize the evaluation of the ALC criteria (\code{method = "alc"}).
    Requires the package be compiled with CUDA flags; see README/INSTALL 
    in the package source for more details; currently only available 
    via \code{laGP}, and only supports off-loading ALC calculation to the GPU
  }
  \item{parallel}{
    a switch indicating if any parallel calculation of 
    the criteria (\code{method}) is desired.    
    For \code{parallel = "omp"}, the package be compiled with OMP flags; 
    for \code{parallel = "gpu"}, the package must be compiled with CUDA
    flags (only the ALC criteria is supported on the GPU); see README/INSTALL
    in the package source for more details; currently only available 
    via \code{laGP.R}
}
\item{rect}{
  an optional \code{2}-by-\code{ncol(X)} \code{matrix} describing a bounding
  rectangle for \code{X} that is used by the \code{"alcray"} method.  
  If not specified, the rectangle is calculated from the \code{range} applied
  to the columns of \code{X}
}
  \item{verb}{
    a positive integer specifying the verbosity level; \code{verb = 0}
    is quiet, and larger values cause more progress information to be
    printed to the screen
}
}
\details{

  A sub-design of \code{X} of size \code{end} is built-up according to
  the criteria prescribed by the \code{method} used to predict at
  \code{Xref}.  The first \code{start} locations are NNs in order to
  initialize the first GP, via \code{\link{newGP}}, and thereby initialize the
  search.  Each subsequent addition is found via the chosen criterion
  (\code{method}), and the GP fit is updated via \code{\link{updateGP}}.

  The runtime is cubic in \code{end}, although
  the multiplicative \dQuote{constant} out front depends on the
  \code{method} chosen, the size of the design \code{X}, and
  \code{close}.  The \code{"alcray"} method has a smaller constant
  since it does not search over all candidate exhaustively.

  After building the sub-design, local MLE/MAP lengthscale (and/or
  nugget) parameters are estimated, depending on the \code{d} and
  \code{g} arguments supplied.  This is facilitated by calls to
  \code{\link{mleGP}} or \code{\link{jmleGP}}.

  Finally \code{\link{predGP}} is called on the resulting local GP
  model, and the parameters of the resulting Student-t distribution(s)
  are returned.
  
  The function \code{laGP.R} is a prototype \R-only version for
  debugging and transparency purposes.  It is slower than
  \code{laGP}, which is primarily in C.  However it may be
  useful for developing new programs that involve similar subroutines.
  The current version allows OpenMP and/or GPU parallelization 
  of the criteria (\code{method}) if the package is compiled with 
  the appropriate flags.  See README/INSTALL in the package source for 
  more information.  For algorithmic details, see 
  Gramacy, Niemi, \& Weiss (2013)
}
\value{
The output is a \code{list} with the following components.

\item{mean }{a vector of predictive means of length \code{nrow(Xref)}}
\item{s2 }{a vector of Student-t scale parameters of length
  \code{nrow(Xref)}}
\item{df }{a Student-t degrees of freedom scalar (applies to all
  \code{Xref})}
\item{llik }{a scalar indicating the maximized log likelihood or log posterior
  probability of the data/parameter(s) under the chosen sub-design;
  provided up to an additive constant}
\item{time }{a scalar giving the passage of wall-clock time elapsed
  for (substantive parts of) the calculation}
\item{method }{a copy of the \code{method} argument}
\item{d }{a full-list version of the \code{d} argument, possibly completed by \code{darg}}
\item{g }{a full-list version of the \code{g} argument, possibly
  completed by \code{garg}}
\item{mle }{if \code{d$mle} and/or \code{g$mle} are \code{TRUE}, then
  \code{mle} is a \code{data.frame} containing the values found for
  these parameters, and the number of required iterations}
\item{Xi }{when \code{Xi.ret = TRUE}, this field contains a vector of
  indices of length \code{end} into \code{X} indicating the sub-design chosen}
}
\references{
  R.B. Gramacy and D.W. Apley (2013).
  \emph{Local Gaussian process approximation for large computer
    experiments.}
  Preprint available on arXiv:1303.0383; 
  \url{http://arxiv.org/abs/1303.0383}

  R.B. Gramacy, J. Niemi, R.M. Weiss (2013). 
  \emph{Massively parallel approximate Gaussian process regression.}
  Preprint on arXiv:1310.5182
  \url{http://arxiv.org/abs/1310.5182}
}
\author{
  Robert B. Gramacy \email{rbgramacy@chicagobooth.edu}
}
\note{
  Note that using \code{method="NN"} gives the same result as specifying
  \code{start=end}, however at some extra computational expense.

  Handling multiple reference locations
  (\code{nrow(Xref) > 1}) is \dQuote{beta} functionality.  In this case
  the initial \code{start} locations are chosen by applying NN to the
  average distances to all \code{Xref} locations.  This is not
  supported for \code{method="alcray"}.

  Differences between the C \code{qsort} function and \R's
  \code{\link{order}} function may cause chosen designs returned from
  \code{laGP} and \code{laGP.R} to differ when multiple \code{X}
  values are equidistant to \code{Xref}
}

%% ~Make other sections like Warning with \section{Warning }{....} ~

\seealso{
  \code{\link{aGP}}, \code{\link{newGP}}, \code{\link{updateGP}},
    \code{\link{predGP}}, \code{\link{mleGP}}, \code{\link{jmleGP}},
    \code{\link{alcGP}}, \code{\link{mspeGP}}
}
\examples{
## examining a particular laGP call from the larger analysis provided
## in the aGP documentation

## Simple 2-d test function used in Gramacy & Apley (2013);
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

## build up a design with N=~40K locations
x <- seq(-2, 2, by=0.02)
X <- as.matrix(expand.grid(x, x))
Z <- f2d(X)

## predictive grid with NN=400 locations,
## change NN to 10K to mimic setup in Gramacy & Apley (2013)
xx <- seq(-1.975, 1.975, length=20)
XX <- as.matrix(expand.grid(xx, xx))
ZZ <- f2d(XX)

## local analysis, first pass
Xref <- XX[250,]
out <- laGP(Xref, 6, 50, X, Z, method="nn")

## second and pass via ALC and MSPE, respectively
out2 <- laGP(Xref, 6, 50, X, Z, d=out$mle$d)
out2.mspe <- laGP(Xref, 6, 50, X, Z, d=out2$mle$d)

## look at the different designs
plot(rbind(X[out2$Xi,], X[out2.mspe$Xi,]), type="n",
     xlab="x1", ylab="x2", main="comparing local designs")
points(Xref[1], Xref[2], col=2, cex=0.5)
text(X[out2$Xi,], labels=1:50, cex=0.6)
text(X[out2.mspe$Xi,], labels=1:50, cex=0.6, col="green")
legend("topright", c("pass 2 ALC", "pass 3 MSPE"),
       text.col=c("black", "green"), bty="n")
}

\keyword{ nonparametric }
\keyword{ nonlinear }
\keyword{ smooth }
\keyword{ models }
\keyword{ regression }
\keyword{ spatial }
