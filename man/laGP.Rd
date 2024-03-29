\name{laGP}
\alias{laGP}
\alias{laGP.R}
\alias{laGPsep.R}
\alias{laGPsep}

\title{
Localized Approximate GP Prediction At a Single Input Location
}
\description{
  Build a sub-design of \code{X} of size \code{end}, and infer parameters,
  for approximate Gaussian process prediction at reference location(s) 
  \code{Xref}. Return the moments of those predictive equations, and indices
  into the local design
}
\usage{
laGP(Xref, start, end, X, Z, d = NULL, g = 1/10000,
     method = c("alc", "alcopt", "alcray", "mspe", "nn", "fish"), Xi.ret = TRUE,
     close = min((1000+end)*if(method[1] \%in\% c("alcray", "alcopt")) 10 else 1, nrow(X)), 
     alc.gpu = FALSE, numstart = if(method[1] == "alcray") ncol(X) else 1, 
     rect = NULL, lite = TRUE, verb = 0)
laGP.R(Xref, start, end, X, Z, d = NULL, g = 1/10000,
     method = c("alc", "alcopt", "alcray", "mspe", "nn", "fish"), 
     Xi.ret = TRUE, pall = FALSE, 
     close = min((1000+end)*if(method[1] \%in\% c("alcray", "alcopt")) 10 else 1, nrow(X)),
     parallel = c("none", "omp", "gpu"), 
     numstart = if(method[1] == "alcray") ncol(X) else 1, 
     rect = NULL, lite = TRUE, verb = 0)
laGPsep(Xref, start, end, X, Z, d = NULL, g = 1/10000,
     method = c("alc", "alcopt", "alcray", "nn"), Xi.ret = TRUE, 
     close = min((1000+end)*if(method[1] \%in\% c("alcray", "alcopt")) 10 else 1, nrow(X)), 
     alc.gpu = FALSE, numstart = if(method[1] == "alcray") ncol(X) else 1, 
     rect = NULL, lite = TRUE, verb=0)
laGPsep.R(Xref, start, end, X, Z, d = NULL, g = 1/10000,
     method = c("alc", "alcopt", "alcray", "nn"), 
     Xi.ret = TRUE, pall = FALSE, 
     close = min((1000+end)*if(method[1] \%in\% c("alcray", "alcopt")) 10 else 1, nrow(X)),
     parallel = c("none", "omp", "gpu"), 
     numstart = if(method[1] == "alcray") ncol(X) else 1, 
     rect = NULL, lite = TRUE, verb = 0)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{Xref}{
    a vector of length \code{ncol(X)} containing a single reference
    location; or a \code{matrix} with \code{ncol(Xref) = ncol(X)} containing
    multiple reference locations (unless \code{method = "alcray"}) 
    for simultaneous sub-design and prediction
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
    a vector of responses/dependent values with \code{length(Z) = nrow(X)}
   }
  \item{d}{
    a prior or initial setting for the lengthscale
    parameter for a Gaussian correlation function; a (default)
    \code{NULL} value causes a sensible regularization (prior) and
    initial setting to be generated via \code{\link{darg}};
    a scalar specifies an initial value, causing \code{\link{darg}}
    to only generate the prior; otherwise, a list or partial list matching the output
    of \code{\link{darg}} can be used to specify a custom prior.  In the
    case of a partial list, the only the missing entries will be
    generated. Note that a default/generated list specifies MLE/MAP
    inference for this parameter. With 
    \code{laGPsep}, the starting values can be an
    \code{ncol(X)}-by-\code{nrow(XX)} \code{matrix} or \code{ncol(X)} vector
   }
  \item{g}{
    a prior or initial setting for the nugget parameter; a 
    \code{NULL} value causes a sensible regularization (prior) and
    initial setting to be generated via \code{\link{garg}}; a scalar
    (default \code{g = 1/10000}) specifies an initial value, 
    causing \code{\link{garg}} to only generate the prior; otherwise, a
    list or partial list matching the output of \code{\link{garg}} 
    can be used to specify a custom prior.  In the case of a partial list,
    only the missing entries will be generated. Note that a 
    default/generated list specifies \emph{no} inference for this
    parameter; i.e., it is fixed at its starting or default value, 
    which may be appropriate for emulating 
    deterministic computer code output.  In such situations a 
    value much smaller than the default may work even better (i.e., 
    yield better out-of-sample predictive performance).  
    The default was chosen conservatively
   }
  \item{method}{
    Specifies the method by which \code{end-start} candidates from
    \code{X} are chosen in order to predict at \code{Xref}. In brief, ALC
    (\code{"alc"}, default) minimizes predictive variance; ALCRAY 
    (\code{"alcray")}) executes a thrifty ALC-based search focused on 
    rays emanating from the reference location [must have \code{nrow(Xref) = 1}]; 
    ALCOPT (\code{"alcopt"}) optimizes a continuous ALC analog via derivatives to
    and snaps the solution back to the candidate grid; 
    MSPE (\code{"mspe"}) augments ALC 
    with extra derivative information to minimize mean-squared prediction 
    error (requires extra computation);
    NN (\code{"nn"}) uses nearest neighbor; and EFI (\code{"fish"}) uses
    the expected Fisher information - essentially \code{1/G} from
    Gramacy & Apley (2015) - which is global heuristic, i.e., not
    localized to \code{Xref}.  
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
   a non-negative integer \code{end < close <= nrow(X)} 
   specifying the number of NNs (to \code{Xref}) in 
   \code{X} to consider when searching for elements of the sub-design; 
   \code{close = 0} specifies all.  For \code{method="alcray"} and 
   \code{method="alcopt"}, this argument specifies the scope used to snap 
   solutions obtained via analog continuous searches back to elements of \code{X}, 
   otherwise there are no restrictions on those searches.  Since these
   approximate searches are cheaper, they can afford a larger 
   \dQuote{snapping scope} hence the larger default}
  \item{alc.gpu}{
   a scalar \code{logical} indicating if a GPU should be used to
   parallelize the evaluation of the ALC criteria (\code{method = "alc"}).
   Requires the package be compiled with CUDA flags; see README/INSTALL 
   in the package source for more details; currently only available 
   for \code{nrow(Xref) == 1} 
   via \code{laGP}, not \code{laGPsep} or the \code{.R} variants,
   and only supports off-loading ALC calculation to the GPU
   }
  \item{parallel}{
   a switch indicating if any parallel calculation of 
   the criteria is desired.  Currently parallelization at this level is only 
   provided for option \code{method = "alc"}). 
   For \code{parallel = "omp"}, the package must be compiled with OMP flags;
   for \code{parallel = "gpu"}, the package must be compiled with CUDA
   flags see README/INSTALL
   in the package source for more details; currently only available 
   via \code{laGP.R}
   }
  \item{numstart}{ a scalar integer indicating the number of rays for each
   greedy search when \code{method="alcray"} or the number of restarts
   when \code{method="alcopt"}.  More rays or restarts
   leads to a more thorough, but more computational intensive search.
   This argument is not involved in other methods }
  \item{rect}{
   an optional \code{2}-by-\code{ncol(X)} \code{matrix} describing a bounding
   rectangle for \code{X} that is used by the \code{"alcray"} method.  
   If not specified, the rectangle is calculated from \code{range} applied
   to the columns of \code{X}
   }
  \item{lite}{ Similar to the \code{\link{predGP}} option of the same name,
  this argument specifies whether (\code{TRUE}, the default) or not (\code{FALSE}) to return
  a full covariance structure is returned, as opposed the diagonal only.  A full covariance
  structure requires more computation and more storage.  This option is
  only relevant when \code{nrow(Xref) > 1} }
  \item{verb}{
   a non-negative integer specifying the verbosity level; \code{verb = 0}
   is quiet, and larger values cause more progress information to be
   printed to the screen
   }
}
\details{

  A sub-design of \code{X} of size \code{end} is built-up according to
  the criteria prescribed by the \code{method} and then used to predict at
  \code{Xref}.  The first \code{start} locations are NNs in order to
  initialize the first GP, via \code{\link{newGP}} or \code{\link{newGPsep}}, 
  and thereby initialize the
  search.  Each subsequent addition is found via the chosen criterion
  (\code{method}), and the GP fit is updated via \code{\link{updateGP}}
  or \code{\link{updateGPsep}}

  The runtime is cubic in \code{end}, although
  the multiplicative \dQuote{constant} out front depends on the
  \code{method} chosen, the size of the design \code{X}, and
  \code{close}.  The \code{"alcray"} method has a smaller constant
  since it does not search over all candidates exhaustively.

  After building the sub-design, local MLE/MAP lengthscale (and/or
  nugget) parameters are estimated, depending on the \code{d} and
  \code{g} arguments supplied.  This is facilitated by calls to
  \code{\link{mleGP}} or \code{\link{jmleGP}}.

  Finally \code{\link{predGP}} is called on the resulting local GP
  model, and the parameters of the resulting Student-t distribution(s)
  are returned.  Unless \code{Xi.ret = FALSE}, the indices of the
  local design are also returned.
  
  \code{laGP.R} and \code{laGPsep.R} are a prototype \R-only version for
  debugging and transparency purposes.  They are slower than
  \code{laGP} and \code{laGPsep}, which are primarily in C, and may not
  provide identical output in all cases due to differing library implementations
  used as subroutines; see note below for an example.  \code{laGP.R} and other
  \code{.R} functions in the package may be useful for developing new programs
  that involve similar subroutines. The current version of \code{laGP.R}
  allows OpenMP and/or GPU parallelization of the criteria (\code{method}) if
  the package is compiled with the appropriate flags.  See README/INSTALL in
  the package source for more information.  For algorithmic details, see
  Gramacy, Niemi, & Weiss (2014) }
  
\value{
The output is a \code{list} with the following components.

\item{mean}{a vector of predictive means of length \code{nrow(Xref)}}
\item{s2}{a vector of Student-t scale parameters of length
  \code{nrow(Xref)}}
\item{df}{a Student-t degrees of freedom scalar (applies to all
  \code{Xref})}
\item{llik}{a scalar indicating the maximized log likelihood or log posterior
  probability of the data/parameter(s) under the chosen sub-design;
  provided up to an additive constant}
\item{time}{a scalar giving the passage of wall-clock time elapsed
  for (substantive parts of) the calculation}
\item{method}{a copy of the \code{method} argument}
\item{d}{a full-list version of the \code{d} argument, possibly completed by \code{darg}}
\item{g}{a full-list version of the \code{g} argument, possibly
  completed by \code{garg}}
\item{mle }{if \code{d$mle} and/or \code{g$mle} are \code{TRUE}, then
  \code{mle} is a \code{data.frame} containing the values found for
  these parameters, and the number of required iterations}
\item{Xi}{when \code{Xi.ret = TRUE}, this field contains a vector of
  indices of length \code{end} into \code{X} indicating the sub-design chosen}
\item{close}{a copy of the input argument}
}

\references{
  Gramacy, R. B. (2020) \emph{Surrogates: Gaussian Process Modeling,
  Design and Optimization for the Applied Sciences}. Boca Raton,
  Florida: Chapman Hall/CRC.  (See Chapter 9.)
  \url{https://bobby.gramacy.com/surrogates/}

  F. Sun, R.B. Gramacy, B. Haaland, E. Lawrence, and A. Walker (2019).
  \emph{Emulating satellite drag from large simulation experiments},
  SIAM/ASA Journal on Uncertainty Quantification, 7(2), pp. 720-759;
  preprint on arXiv:1712.00182;
  \url{https://arxiv.org/abs/1712.00182}

  R.B. Gramacy (2016). \emph{\pkg{laGP}: Large-Scale Spatial Modeling via 
  Local Approximate Gaussian Processes in \R.}, Journal of Statistical 
  Software, 72(1), 1-46; \doi{10.18637/jss.v072.i01} 
  or see \code{vignette("laGP")}
  
  R.B. Gramacy and B. Haaland (2016).
  \emph{Speeding up neighborhood search in local Gaussian process prediction.}
  Technometrics, 58(3), pp. 294-303;
  preprint on arXiv:1409.0074 
  \url{https://arxiv.org/abs/1409.0074}

  R.B. Gramacy and D.W. Apley (2015).
  \emph{Local Gaussian process approximation for large computer
    experiments.} Journal of Computational and Graphical Statistics, 
  24(2), pp. 561-678; preprint on arXiv:1303.0383;
  \url{https://arxiv.org/abs/1303.0383}

  R.B. Gramacy, J. Niemi, R.M. Weiss (2014).
  \emph{Massively parallel approximate Gaussian process regression.}
  SIAM/ASA Journal on Uncertainty Quantification, 2(1), pp. 568-584;
  preprint on arXiv:1310.5182;
  \url{https://arxiv.org/abs/1310.5182}

}

\author{
  Robert B. Gramacy \email{rbg@vt.edu} and Furong Sun \email{furongs@vt.edu}
}

\note{
  \code{laGPsep} provides the same functionality as \code{laGP} but deploys
  a separable covariance function.  However criteria (\code{method}s) EFI and
  MSPE are not supported.  This is considered \dQuote{beta} functionality
  at this time.

  Note that using \code{method="NN"} gives the same result as specifying
  \code{start=end}, however at some extra computational expense.

  Handling multiple reference locations
  (\code{nrow(Xref) > 1}) is \dQuote{beta} functionality.  In this case
  the initial \code{start} locations are chosen by applying NN to the
  average distances to all \code{Xref} locations.  Using 
  \code{method="alcopt"} causes exhaustive search to be approximated by
  a continuous analog via closed form derivatives.  
  See \code{\link{alcoptGP}} for more details.   Although the approximation
  provided has a spirit similar to \code{method="alcray"}, in that 
  both methods are intended to offer a thrifty alternative, 
  \code{method="alcray"} is not applicable when \code{nrow(Xref) > 1}.

  Differences between the C \code{qsort} function and \R's
  \code{\link{order}} function may cause chosen designs returned from
  \code{laGP} and \code{laGP.R} (code and \code{laGPsep} and \code{laGPsep.R})
  to differ when multiple \code{X} values are equidistant to \code{Xref}
}

%% ~Make other sections like Warning with \section{Warning }{....} ~

\seealso{
\code{vignette("laGP")}, 
  \code{\link{aGP}}, \code{\link{newGP}}, \code{\link{updateGP}},
    \code{\link{predGP}}, \code{\link{mleGP}}, \code{\link{jmleGP}},
    \code{\link{alcGP}}, \code{\link{mspeGP}}, \code{\link{alcrayGP}},
    \code{\link{randLine}} ## path-based local prediction via \code{laGP}
}

\examples{
## examining a particular laGP call from the larger analysis provided
## in the aGP documentation

## A simple 2-d test function used in Gramacy & Apley (2014);
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

## build up a design with N=~40K locations
x <- seq(-2, 2, by=0.02)
X <- as.matrix(expand.grid(x, x))
Z <- f2d(X)

## optional first pass of nearest neighbor
Xref <- matrix(c(-1.725, 1.725), nrow=TRUE)
out <- laGP(Xref, 6, 50, X, Z, method="nn")

## second pass via ALC, ALCOPT, MSPE, and ALC-ray respectively,
## conditioned on MLE d-values found above
out2 <- laGP(Xref, 6, 50, X, Z, d=out$mle$d)
# out2.alcopt <- laGP(Xref, 6, 50, X, Z, d=out2$mle$d, method="alcopt")
out2.mspe <- laGP(Xref, 6, 50, X, Z, d=out2$mle$d, method="mspe")
out2.alcray <- laGP(Xref, 6, 50, X, Z, d=out2$mle$d, method="alcray")

## look at the different designs
plot(rbind(X[out2$Xi,], X[out2.mspe$Xi,]), type="n",
     xlab="x1", ylab="x2", main="comparing local designs")
points(Xref[1], Xref[2], col=2, cex=0.5)
text(X[out2$Xi,], labels=1:50, cex=0.6)
# text(X[out2.alcopt$Xi,], labels=1:50, cex=0.6, col="forestgreen")
text(X[out2.mspe$Xi,], labels=1:50, cex=0.6, col="blue")
text(X[out2.alcray$Xi,], labels=1:50, cex=0.6, col="red")
legend("right", c("ALC", "ALCOPT", "MSPE", "ALCRAY"),
       text.col=c("black", "forestgreen", "blue", "red"), bty="n")

## compare computational time
c(nn=out$time, alc=out2$time, # alcopt=out2.alcopt$time, 
  mspe=out2.mspe$time, alcray=out2.alcray$time)

\dontrun{
  ## Joint path sampling: a comparison between ALC-ex, ALC-opt and NN

  ## defining a predictive path
  wx <- seq(-0.85, 0.45, length=100)
  W <- cbind(wx-0.75, wx^3+0.51)

  ## three comparators from Sun, et al. (2017)
  ## larger-than-default "close" argument to capture locations nearby path
  p.alc <- laGPsep(W, 6, 100, X, Z, close=10000, lite=FALSE)
  p.alcopt <- laGPsep(W, 6, 100, X, Z, method="alcopt", lite=FALSE)
  ## note that close=10*(1000+end) would be the default for method = "alcopt"
  p.nn <- laGPsep(W, 6, 100, X, Z, method="nn", close=10000, lite=FALSE)

  ## time comparison
  c(alc=p.alc$time, alcopt=p.alcopt$time, nn=p.nn$time)

  ## visualization
  plot(W, type="l", xlab="x1", ylab="x2", xlim=c(-2.25,0), ylim=c(-0.75,1.25), lwd=2)
  points(X[p.alc$Xi,], col=2, cex=0.6)
  lines(W[,1]+0.25, W[,2]-0.25, lwd=2)
  points(X[p.nn$Xi,1]+0.25, X[p.nn$Xi,2]-0.25, pch=22, col=3, cex=0.6)
  lines(W[,1]-0.25, W[,2]+0.25, lwd=2) 
  points(X[p.alcopt$Xi,1]-0.25, X[p.alcopt$Xi,2]+0.25, pch=23, col=4, cex=0.6)
  legend("bottomright", c("ALC-opt", "ALC-ex", "NN"), pch=c(22, 21, 23), col=c(4,2,3))
}
}

\keyword{ nonparametric }
\keyword{ nonlinear }
\keyword{ smooth }
\keyword{ models }
\keyword{ regression }
\keyword{ spatial }
