\name{aGP}
\alias{aGP}
\alias{aGP.R}
\alias{aGPsep.R}
\alias{aGPsep}
\alias{aGP.parallel}
\alias{aGP.seq}

\title{
Localized Approximate GP Regression For Many Predictive Locations
}
\description{
  Facilitates localized Gaussian process inference and prediction at a large 
  set of predictive locations, by (essentially) calling \code{\link{laGP}}
  at each location, and returning the moments of the predictive
  equations, and indices into the design, thus obtained
}
\usage{
aGP(X, Z, XX, start = 6, end = 50, d = NULL, g = 1/10000,
    method = c("alc", "alcray", "mspe", "nn", "fish"), Xi.ret = TRUE,
    close = min((1000+end)*if(method[1] == "alcray") 10 else 1, nrow(X)), 
    numrays = ncol(X), num.gpus = 0, gpu.threads = num.gpus, 
    omp.threads = if (num.gpus > 0) 0 else 1, 
    nn.gpu = if (num.gpus > 0) nrow(XX) else 0, verb = 1)
aGP.parallel(cls, XX, chunks = length(cls), X, Z, start = 6, end = 50, 
    d = NULL, g = 1/10000, method = c("alc", "alcray", "mspe", "nn", "fish"), 
    Xi.ret = TRUE, 
    close = min((1000+end)*if(method[1] == "alcray") 10 else 1, nrow(X)),
    numrays = ncol(X), num.gpus = 0, gpu.threads = num.gpus, 
    omp.threads = if (num.gpus > 0) 0 else 1,
    nn.gpu = if (num.gpus > 0) nrow(XX) else 0, verb = 1)
aGP.R(X, Z, XX, start = 6, end = 50, d = NULL, g = 1/10000,
    method = c("alc", "alcray", "mspe", "nn", "fish"), Xi.ret = TRUE,
    close = min((1000+end) *if(method[1] == "alcray") 10 else 1, nrow(X)),
    numrays = ncol(X), laGP=laGP.R, verb = 1)
aGPsep(X, Z, XX, start = 6, end = 50, d = NULL, g = 1/10000,
    method = c("alc", "alcray", "nn"), Xi.ret = TRUE,
    close = min((1000+end)*if(method[1] == "alcray") 10 else 1, nrow(X)),
    numrays = ncol(X),  omp.threads = 1, verb = 1)
aGPsep.R(X, Z, XX, start = 6, end = 50, d = NULL, g = 1/10000,
    method = c("alc", "alcray", "nn"), Xi.ret = TRUE,
    close = min((1000+end)*if(method[1] == "alcray") 10 else 1, nrow(X)),
    numrays = ncol(X), laGPsep=laGPsep.R, verb = 1)
aGP.seq(X, Z, XX, d, methods=rep("alc", 2), M=NULL, ncalib=0, ...)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{X}{a \code{matrix} or \code{data.frame} containing
    the full (large) design matrix of input locations
}
\item{Z}{
  a vector of responses/dependent values with \code{length(Z) = nrow(X)}
}
  \item{XX}{
    a \code{matrix} or \code{data.frame} of out-of-sample
    predictive locations with \code{ncol(XX) = ncol(X)}; \code{aGP} calls \code{laGP} for
    each row of \code{XX} as a value of \code{Xref}, independently
}
  \item{start}{
        the number of Nearest Neighbor (NN) locations to start each
	independent call to \code{laGP} with; must have \code{start >= 6}
}
\item{end}{
      the total size of the local designs; \code{start < end}
}
\item{d}{
   a prior or initial setting for the lengthscale
   parameter in a Gaussian correlation function; a (default)
    \code{NULL} value triggers a sensible regularization (prior) and
    initial setting to be generated via \code{\link{darg}};
    a scalar specifies an initial value, causing \code{\link{darg}}
    to only generate the prior; otherwise,
    a list or partial list matching the output
    of \code{\link{darg}} can be used to specify a custom prior.  In the
    case of a partial list, only the missing entries will be
    generated. Note that a default/generated list specifies MLE/MAP
    inference for this parameter.  When specifying initial values, a
    vector of length \code{nrow(XX)} can be provided, giving a
    different initial value for each predictive location. With 
    \code{aGPsep}, the starting values can be an 
    \code{ncol(X)}-by-\code{nrow(XX)} \code{matrix} or an \code{ncol(X)} vector
}
\item{g}{
  a prior or initial setting for the nugget parameter; a 
    \code{NULL} value causes a sensible regularization (prior) and
    initial setting to be generated via \code{\link{garg}}; a scalar
    (default \code{g = 1/10000}) specifies an initial value, causing \code{\link{garg}}
    to only generate the prior; otherwise, a
    list or partial list matching the output of \code{\link{garg}} can be used to
    specify a custom prior.  In the case of a partial list, only the
    missing entries will be generated. Note that a default/generated list
    specifies \emph{no} inference for this parameter; i.e., it is fixed
    at its starting or default value, which may be appropriate for
    emulating deterministic computer code output. In such situations a 
    value much smaller than the default may work even better (i.e., 
    yield better out-of-sample predictive performance).  
    The default was chosen conservatively.  When specifying non-default initial
    values, a vector of length \code{nrow(XX)} can be provided, giving a different
    initial value for each predictive location
}
  \item{method}{
    specifies the method by which \code{end-start} candidates from
    \code{X} are chosen in order to predict at each row \code{XX} independently. 
    In brief, ALC (\code{"alc"}, default) minimizes predictive variance; 
    ALCRAY (\code{"alcray")} executes a thrifty search focused on rays emanating
    from the reference location(s); MSPE
    (\code{"mspe"}) augments ALC with extra derivative information to
    minimize mean-squared prediction error (requires extra computation);
    NN (\code{"nn"}) uses nearest neighbor; and (\code{"fish"}) uses
    the expected Fisher information - essentially \code{1/G} from
    Gramacy & Apley (2015) - which is global heuristic, i.e., not
    localized to each row of \code{XX}}
  \item{methods}{ for \code{aGP.seq} this is a vectorized \code{method} argument,
    containing a list of valid methods to perform in sequence.  When \code{methods = FALSE}
    a call to \code{M} is invoked instead; see below for
    more details }
  \item{Xi.ret}{
    a scalar logical indicating whether or not a \code{matrix} of indices
    into \code{X}, describing the chosen sub-design for each of the
    predictive locations in \code{XX}, should be returned on
    output}
  \item{close}{
    a non-negative integer  \code{end < close <= nrow(X)}
    specifying the number of NNs
    (to each row \code{XX}) in \code{X} to consider when
    searching for the sub-design; \code{close = 0} specifies all. 
    For \code{method="alcray"} this
    specifies the scope used to snap ray-based solutions back to
    elements of \code{X}, otherwise there are no restrictions on that 
    search
  }
  \item{numrays}{ a scalar integer indicating the number of rays for each
  greedy search; only relevant when \code{method="alcray"}.  More rays
  leads to a more thorough, but more computationally intensive search }
  \item{laGP}{
    applicable only to the \R-version \code{aGP.R}, this is a
    function providing the local design implementation to be used.
    Either \code{\link{laGP}} or \code{\link{laGP.R}} can be
    provided, or a bespoke routine providing similar outputs
  }
  \item{laGPsep}{
    applicable only to the \R-version \code{aGPsep.R}, this is a
    function providing the local design implementation to be used.
    Either \code{\link{laGPsep}} or \code{\link{laGPsep.R}} can be
    provided, or a bespoke routine providing similar outputs
  }
\item{num.gpus}{
  applicable only to the C-version \code{aGP}, this is a scalar 
  positive integer indicating the number of GPUs available for calculating 
  ALC (see \code{\link{alcGP}}); the package must be compiled for CUDA support; 
  see README/INSTALL in the package source for more details
}
\item{gpu.threads}{ applicable only to the C-version \code{aGP}; this 
  is a scalar positive integer indicating the number of SMP (i.e., CPU)
  threads queuing ALC jobs on a GPU; the package must be compiled for CUDA support.
   If \code{gpu.threads >= 2} then the package must \emph{also}
  be compiled for OpenMP support; see README/INSTALL in the package source for
  more details.  We recommend setting \code{gpu.threads} to up to two-times
  the sum of the number of GPU devices and CPU cores.
  Only \code{method = "alc"} is supported when using CUDA.   If the
   sum of \code{omp.threads} and \code{gpu.threads} is bigger than the
   max allowed by your system, then that max is used instead (giving 
   \code{gpu.threads} preference) 
}
\item{omp.threads}{ applicable only to the C-version \code{aGP};
   this is a scalar positive integer indicating the number
   of threads to use for SMP parallel processing; the package must be 
   compiled for OpenMP support; see README/INSTALL in the package source for
   more details.  For most Intel-based machines, we recommend setting
   \code{omp.threads} to up to two-times the number of hyperthreaded cores.  When
   using GPUs (\code{num.gpu > 0}), a good default is \code{omp.threads=0},
   otherwise load balancing could be required; see \code{nn.gpu} below. If the
   sum of \code{omp.threads} and \code{gpu.threads} is bigger than the
   max allowed by your system, then that max is used instead (giving 
   \code{gpu.threads} preference)  }
\item{nn.gpu}{ a scalar non-negative integer between \code{0} and 
    \code{nrow(XX)} indicating the number of predictive locations 
    utilizing GPU ALC calculations.  Note this argument is only useful when
    both \code{gpu.threads} and \code{omp.threads} are non-zero, whereby
    it acts as a load balancing mechanism }
\item{verb}{
    a non-negative integer specifying the verbosity level; \code{verb = 0}
    is quiet, and larger values cause more progress information to be
    printed to the screen.  The value \code{min(0,verb-1)} is provided
    to each \code{laGP} call}
\item{cls}{ 
    a cluster object created by \code{\link[parallel]{makeCluster}} 
    from the \pkg{parallel} or \pkg{snow} packages
}
\item{chunks}{
    a scalar integer indicating the number of chunks to break \code{XX} into
    for \pkg{parallel} evaluation on cluster \code{cls}.  
    Usually \code{chunks = length(cl)} is appropriate.  
    However specifying more chunks can be useful when the nodes of
    the cluster are not homogeneous}
\item{M}{ an optional function taking two matrix inputs, of \code{ncol(X)-ncalib}
  and \code{ncalib} columns respectively, which is applied in lieu of 
  \code{aGP}.  This can be useful for calibration where the computer model
  is cheap, i.e., does not require emulation; more details below
  }
\item{ncalib}{ an integer between 1 and \code{ncol(X)} indicating how to
partition \code{X} and \code{XX} inputs into the two matrices required for \code{M}
}
\item{...}{ other arguments passed from \code{aGP.sep} to \code{aGP} }
}
\details{

  This function invokes \code{\link{laGP}} with argument \code{Xref
  = XX[i,]} for each \code{i=1:nrow(XX)}, building up local designs,
  inferring local correlation parameters, and
  obtaining predictive locations independently for each location.  For
  more details see \code{\link{laGP}}.

  The function \code{aGP.R} is a prototype \R-only version for
  debugging and transparency purposes.  It is slower than
  \code{aGP}, which is primarily in C.  However it may be
  useful for developing new programs that involve similar subroutines.
  Note that \code{aGP.R} may provide different output than \code{aGP}
  due to differing library subroutines deployed in \R and C.

  The function \code{aGP.parallel} allows \code{aGP} to be called on segments
  of the \code{XX} matrix distributed to a cluster created by \pkg{parallel}.
  It breaks \code{XX} into \code{chunks} which are sent to \code{aGP} 
  workers pointed to by the entries of \code{cls}.  The \code{aGP.parallel} function
  collects the outputs from each chunk before returning an object
  almost identical to what would have been returned from a single \code{aGP}
  call.  On a single (SMP) node, this represents is a poor-man's version of
  the OpenMP version described below.  On multiple nodes both can be used.

  If compiled with OpenMP flags, the independent calls to 
  \code{\link{laGP}} will be
  farmed out to threads allowing them to proceed in parallel - obtaining
  nearly linear speed-ups.  At this time \code{aGP.R} does not
  facilitate parallel computation, although a future version may exploit
  the \pkg{parallel} functionality for clustered parallel execution.

  If \code{num.gpus > 0} then the ALC part of the independent 
  calculations performed by each thread will be offloaded to a GPU.  
  If both \code{gpu.threads >= 1} and \code{omp.threads >= 1}, 
  some of the ALC calculations will be done on the GPUs, and some 
  on the CPUs.  In our own experimentation we have not found this
  to lead to large speedups relative to \code{omp.threads = 0} when
  using GPUs.  For more details, see Gramacy, Niemi, & Weiss (2014).

  The \code{aGP.sep} function is provided primarily for use in calibration
  exercises, see Gramacy, et al. (2015).  It automates a sequence of
  \code{aGP} calls, each with a potentially different method, 
  successively feeding the previous estimate of local lengthscale (\code{d})
  in as an initial set of values for the next call.  It also allows the
  use of \code{aGP} to be bypassed, feeding the inputs into a user-supplied
  \code{M} function instead.  This feature is enabled when 
  \code{methods = FALSE}.  The \code{M} function takes two matrices
  (same number of rows) as inputs, where the first \code{ncol(X) - ncalib} 
  columns represent
  \dQuote{field data} inputs shared by the physical and computer model
  (in the calibration context), and the remaining \code{ncalib} are 
  the extra tuning or calibration parameters required to evalue the computer
  model.  For examples illustrating \code{aGP.seq} please see the
  documentation file for \code{\link{discrep.est}} and \code{demo("calib")}
}
\value{
  The output is a \code{list} with the following components.

\item{mean }{a vector of predictive means of length \code{nrow(XX)}}
\item{var }{a vector of predictive variances of length
  \code{nrow(Xref)}}
\item{llik }{a vector indicating the log likelihood/posterior
  probability of the data/parameter(s) under the chosen sub-design for
  each predictive location in \code{XX}; provided up to an additive constant}
\item{time }{a scalar giving the passage of wall-clock time elapsed
  for (substantive parts of) the calculation}
\item{method }{a copy of the \code{method} argument}
\item{d }{a full-list version of the \code{d} argument, possibly completed by \code{darg}}
\item{g }{a full-list version of the \code{g} argument, possibly
  completed by \code{garg}}
\item{mle }{if \code{d$mle} and/or \code{g$mle} are \code{TRUE}, then
  \code{mle} is a \code{data.frame} containing the values found for
  these parameters, and the number of required iterations, for each
  predictive location in \code{XX} }
\item{Xi }{when \code{Xi.ret = TRUE}, this field contains a \code{matrix} of
  indices of length \code{end} into \code{X} indicating the sub-design
  chosen for each predictive location in \code{XX}}
\item{close }{ a copy of the input argument }

The \code{aGP.seq} function only returns the output from the final \code{aGP} call.
When \code{methods = FALSE} and \code{M} is supplied, the returned object
is a data frame with a \code{mean} column indicating the output of the computer
model run, and a \code{var} column, which at this time is zero
}
\references{
  Gramacy, R. B. (2020) \emph{Surrogates: Gaussian Process Modeling,
  Design and Optimization for the Applied Sciences}. Boca Raton,
  Florida: Chapman Hall/CRC.  (See Chapter 9.)
  \url{https://bobby.gramacy.com/surrogates/}

  R.B. Gramacy (2016). \emph{\pkg{laGP}: Large-Scale Spatial Modeling via 
  Local Approximate Gaussian Processes in \R.}, Journal of Statistical 
  Software, 72(1), 1-46; \doi{10.18637/jss.v072.i01} 
  or see \code{vignette("laGP")}

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

  R.B. Gramacy and B. Haaland (2016).
  \emph{Speeding up neighborhood search in local Gaussian process prediction.}
  Technometrics, 58(3), pp. 294-303;
  preprint on arXiv:1409.0074 
  \url{https://arxiv.org/abs/1409.0074}
}
\author{
  Robert B. Gramacy \email{rbg@vt.edu}
}
\note{
  \code{aGPsep} provides the same functionality as \code{aGP} but deploys
  a separable covariance function.  Criteria (\code{method}s) EFI and
  MSPE are not supported by \code{aGPsep} at this time.

 Note that using \code{method="NN"} gives the same result as specifying
 \code{start=end}, however at some extra computational expense.

 At this time, this function provides no facility to find local designs
 for the subset of predictive locations \code{XX} jointly, i.e.,
 providing a matrix \code{Xref} to \code{\link{laGP}}.  See \code{\link{laGP}}
 for more details/support for this alternative.

 The use of OpenMP threads with \code{aGPsep} is not as efficient as with
 \code{aGP} when calculating MLEs with respect to the lengthscale (i.e.,
 \code{d=list(mle=TRUE, ...)}).  The reason is that the \code{lbfgsb} C
 entry point uses static variables, and is therefore not thread safe.  
 To circumvent this problem, an OpenMP \code{critical} pragma is used,
 which can create a small bottle neck
}

\seealso{
  \code{vignette("laGP")}, 
  \code{\link{laGP}}, \code{\link{alcGP}}, \code{\link{mspeGP}},  \code{\link{alcrayGP}},
  \code{\link[parallel]{makeCluster}}, \code{\link[parallel]{clusterApply}}
}
\examples{
## first, a "computer experiment"

## Simple 2-d test function used in Gramacy & Apley (2014);
## thanks to Lee, Gramacy, Taddy, and others who have used it before
f2d <- function(x, y=NULL)
  {
    if(is.null(y)){
      if(!is.matrix(x) && !is.data.frame(x)) x <- matrix(x, ncol=2)
      y <- x[,2]; x <- x[,1]
    }
    g <- function(z)
      return(exp(-(z-1)^2) + exp(-0.8*(z+1)^2) - 0.05*sin(8*(z+0.1)))
    z <- -g(x)*g(y)
  }

## build up a design with N=~40K locations
x <- seq(-2, 2, by=0.02)
X <- expand.grid(x, x)
Z <- f2d(X)

## predictive grid with NN=400 locations,
## change NN to 10K (length=100) to mimic setup in Gramacy & Apley (2014)
## the low NN set here is for fast CRAN checks
xx <- seq(-1.975, 1.975, length=10)
XX <- expand.grid(xx, xx)
ZZ <- f2d(XX)

## get the predictive equations, first based on Nearest Neighbor
out <- aGP(X, Z, XX, method="nn", verb=0)
## RMSE
sqrt(mean((out$mean - ZZ)^2))

\dontrun{
## refine with ALC
out2 <- aGP(X, Z, XX, method="alc", d=out$mle$d)
## RMSE
sqrt(mean((out2$mean - ZZ)^2))

## visualize the results
par(mfrow=c(1,3))
image(xx, xx, matrix(out2$mean, nrow=length(xx)), col=heat.colors(128),
      xlab="x1", ylab="x2", main="predictive mean")
image(xx, xx, matrix(out2$mean-ZZ, nrow=length(xx)), col=heat.colors(128),
      xlab="x1", ylab="x2", main="bias")
image(xx, xx, matrix(sqrt(out2$var), nrow=length(xx)), col=heat.colors(128),
      xlab="x1", ylab="x2", main="sd")

## refine with MSPE
out3 <- aGP(X, Z, XX, method="mspe", d=out2$mle$d)
## RMSE
sqrt(mean((out3$mean - ZZ)^2))
}

## version with ALC-ray which is much faster than the ones not
## run above
out2r <- aGP(X, Z, XX, method="alcray", d=out$mle$d, verb=0)
sqrt(mean((out2r$mean - ZZ)^2))

## a simple example with estimated nugget
if(require("MASS")) {

  ## motorcycle data and predictive locations
  X <- matrix(mcycle[,1], ncol=1)
  Z <- mcycle[,2]
  XX <- matrix(seq(min(X), max(X), length=100), ncol=1)

  ## first stage
  out <- aGP(X=X, Z=Z, XX=XX, end=30, g=list(mle=TRUE), verb=0) 
  
  ## plot smoothed versions of the estimated parameters
  par(mfrow=c(2,1))
  df <- data.frame(y=log(out$mle$d), XX)
  lo <- loess(y~., data=df, span=0.25)
  plot(XX, log(out$mle$d), type="l")
  lines(XX, lo$fitted, col=2)
  dfnug <- data.frame(y=log(out$mle$g), XX)
  lonug <- loess(y~., data=dfnug, span=0.25)
  plot(XX, log(out$mle$g), type="l")
  lines(XX, lonug$fitted, col=2)

  ## second stage design
  out2 <- aGP(X=X, Z=Z, XX=XX, end=30, verb=0,
		  d=list(start=exp(lo$fitted), mle=FALSE),
		  g=list(start=exp(lonug$fitted)))
  
  ## plot the estimated surface
  par(mfrow=c(1,1))
  plot(X,Z)
  df <- 20
  s2 <- out2$var*(df-2)/df
  q1 <- qt(0.05, df)*sqrt(s2) + out2$mean
  q2 <- qt(0.95, df)*sqrt(s2) + out2$mean
  lines(XX, out2$mean)
  lines(XX, q1, col=1, lty=2)
  lines(XX, q2, col=1, lty=2)
}

## compare to the single-GP result provided in the mleGP documentation
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ nonparametric }
\keyword{ nonlinear }
\keyword{ smooth }
\keyword{ models }
\keyword{ regression }
\keyword{ spatial }
