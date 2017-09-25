\name{fcalib}
\alias{fcalib}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Objective function for performing large scale computer model
calibration via optimization
}
\description{
Defines an objective function for performing blackbox optimization towards
solving a modularized calibration of large computer model simulation to field
data }
\usage{
fcalib(u, XU, Z, X, Y, da, d, g, uprior = NULL, 
  methods = rep("alc", 2), M = NULL, bias = TRUE, 
  omp.threads = 1, save.global = FALSE, verb=1)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{u}{
  a vector of length \code{ncol(XU) - ncol(X)} containing a setting
  of the calibration parameter
}
  \item{XU}{
   a \code{matrix} or \code{data.frame} containing
    the full (large) design matrix of input locations to a computer
    simulator whose final \code{ncol(XU) - ncol(X)} columns
    contain settings of a calibration or tuning parameter like
    \code{u}
}
  \item{Z}{
  a vector of responses/dependent values with \code{length(Z) = ncol(XU)}
  of computer model outputs at \code{XU}
}
  \item{X}{
  a \code{matrix} or \code{data.frame} containing
    the full (large) design matrix of input locations
}
  \item{Y}{
  a vector of values with \code{length(Y) = ncol(X)}
  containing the response from field data observations
  at \code{X}.  A \code{Y}-vector with \code{length(Y) = k*ncol(X)},
  for positive integer \code{k}, can be supplied in which case
  the multiple \code{Y}-values will be treated as replicates
  at the \code{X}-values
}
  \item{da}{
  for emulating \code{Z} at \code{XU}: 
  a prior or initial setting for the (single/isotropic) lengthscale
   parameter in a Gaussian correlation function; a (default)
    \code{NULL} value triggers a sensible regularization (prior) and
    initial setting to be generated via \code{\link{darg}};
    a scalar specifies an initial value, causing \code{\link{darg}}
    to only generate the prior; otherwise,
    a list or partial list matching the output
    of \code{\link{darg}} can be used to specify a custom prior.  In the
    case of a partial list, the only the missing entries will be
    generated. Note that a default/generated list specifies MLE/MAP
    inference for this parameter.  When specifying initial values, a
    vector of length \code{nrow(XX)} can be provided, giving a
    different initial value for each predictive location
}
  \item{d}{
  for the discrepancy between emulations \code{Yhat} at \code{X}, based
  on \code{Z} at \code{XU}, and the oputs \code{Y} observed at \code{X}.
  Otherwise, same description as \code{da} above 
}
  \item{g}{
  for the nugget in the GP model for the discrepency between emulation
  \code{Yhat} at \code{X}, based
  on \code{Z} at \code{XU}, and the outputs \code{Y} observed at \code{X}:
   a prior or initial setting for the nugget parameter; a 
    \code{NULL} value causes a sensible regularization (prior) and
    initial setting to be generated via \code{\link{garg}}; a scalar (default
    \code{g = 1/1000}) specifies an initial value, causing \code{\link{garg}}
    to only generate the prior; otherwise, a list or partial list matching the
    output of \code{\link{garg}} can be used to specify a custom prior.  In
    the case of a partial list, only the missing entries will be
    generated. Note that a default/generated list specifies \emph{no}
    inference for this parameter; i.e., it is fixed at its starting value,
    which may be appropriate for emulating deterministic computer code output.
    When specifying initial values, a vector of length \code{nrow(XX)} can be
    provided, giving a different initial value for each predictive location.
    At this time, estimating a nugget for the computer model emulator is not
    supported by \code{fcalib} }
  \item{uprior}{
  an optional function taking \code{u} arguments which returns a log
  prior density value for the calibration parameter.
}
  \item{methods}{
  a sequence of local search methods to be deployed when emulating 
  \code{Z} at \code{XU} via \code{\link{aGP}};  see \code{\link{aGP.seq}} for more
  details; provide \code{methods = FALSE} to use the computer model \code{M}
  directly
}
  \item{M}{
  a computer model \dQuote{simulation} function taking two matrices as
  inputs, to be used in lieu of emulation; see \code{\link{aGP.seq}} for mode details
}
  \item{bias}{
  a scalar logical indicating whether a GP discrepancy or bias term should
  be estimated via \code{\link{discrep.est}}, as opposed to 
  only a Gaussian (zero-mean) variance;
  see \code{\link{discrep.est}} for more details
  }
  \item{omp.threads}{
  a scalar positive integer indicating the number
   of threads to use for SMP parallel processing; see \code{\link{aGP}} for more details
}
  \item{save.global}{
  an environment, e.g., \code{.GlobalEnv} if each evaluation of \code{fcalib}, say as
  called by a wrapper or optimization routine, should be saved.  The variable
  used in that environment will be \code{fcalib.save}.  Otherwise \code{save.global = FALSE}
  will skip saving the information}
  \item{verb}{
    a non-negative integer specifying the verbosity level; \code{verb = 0}
    is quiet, whereas a larger value causes each evaluation to be printed
    to the screen}
}
\details{

  Gramacy, et al. (2015) defined an objective function which, when optimized,
  returns a setting of calibration parameters under a setup akin to
  the modularized calibration method of Liu, et al., (2009).  The \code{fcalib}
  function returns a log density (likelihood or posterior probability) value 
  obtained by performing emulation at a set of inputs \code{X} augmented
  with a value of the calibration parameter, \code{u}.  The emulator
  is trained on \code{XU} and \code{Z}, presumed to be very large 
  relative to the size of the field data set \code{X} and \code{Y},
  necesitating the use of approximate methods like \code{\link{aGP}}, 
  via \code{\link{aGP.seq}}.  The
  emulated values, call them \code{Yhat} are fed along with \code{X} and 
  \code{Y} into the \code{\link{discrep.est}} function, whose whose
  likelihood or posterior calculation serves as a measure of merit for
  the value \code{u}.

  The \code{fcalib} function is deterministic but, as Gramacy, et al. (2015)
  described, can result is a rugged objective surface for optimizing,
  meaning that conventional methods, like those in \code{\link{optim}}
  are unlikely to work well.  They instead recommend using a blackbox
  derivative-free method, like NOMAD (Le Digabel, 2011).  In our example
  below we use the implementation in the \pkg{crs} package, which provides
  an \R wrapper around the underlying C library.

  Note that while \code{fcalib} automates a call first to \code{\link{aGP.seq}}
  and then to \code{\link{discrep.est}}, it does not return enough
  information to complete, say, an out-of-sample prediction exercise like
  the one demonstrated in the \code{\link{discrep.est}} documentation.
  Therefore, after \code{\link{fcalib}} is used in an optimization to
  find the best setting of the calibration parameter, \code{u}, 
  those functions must then be used in post-processing to complete a
  prediction exercise.  See \code{demo("calib")} or \code{vignette("laGP")}
  for more details
}
\value{
Returns a scalar measuring the negative log likelihood or posterior density
of the calibration parameter \code{u} given the other inputs, for
the purpose of optimization over \code{u}
}

\note{
  Note that in principle a separable correlation function could be used 
  (e.g, via \code{newGPsep}), however this is not implemented at this time
}

\references{

  R.B. Gramacy (2016). \emph{\pkg{laGP}: Large-Scale Spatial Modeling via 
  Local Approximate Gaussian Processes in \R.}, Journal of Statistical 
  Software, 72(1), 1-46; or see \code{vignette("laGP")}

  R.B. Gramacy, D. Bingham, JP. Holloway, M.J. Grosskopf, C.C. Kuranz, E. Rutter, 
  M. Trantham, and P.R. Drake (2015). \emph{Calibrating a large computer 
  experiment simulating radiative shock hydrodynamics.} 
  Annals of Applied Statistics, 9(3) 1141-1168; preprint on arXiv:1410.3293 
  \url{http://arxiv.org/abs/1410.3293}

  F. Liu, M. Bayarri, and J. Berger (2009). 
  \emph{Modularization in Bayesian analysis, with emphasis on analysis of computer models.} 
  Bayesian Analysis, 4(1) 119-150.

  S. Le Digabel (2011). 
  \emph{Algorithm 909: NOMAD: Nonlinear Optimization with the MADS algorithm}.
  ACM Transactions on Mathematical Software, 37, 4, 44:1-44:15.

  J.S. Racine, Z. and Nie (2012). \pkg{crs}: 
  \emph{Categorical regression splines}. \R package version 0.15-18.
}
\author{
  Robert B. Gramacy \email{rbg@vt.edu}
}

%% ~Make other sections like Warning with \section{Warning }{....} ~

\seealso{
\code{vignette("laGP")}, 
\code{\link{jmleGP}}, \code{\link{newGP}}, \code{\link{aGP.seq}}, \code{\link{discrep.est}},
\code{\link[crs]{snomadr}} 
}
\examples{
## the example here illustrates how fcalib combines aGP.seq and 
## discrep.est functions, duplicating the example in the discrep.est
## documentation file.  it is comprised of snippets from demo("calib"), 
## which contains code from the Calibration Section of vignette("laGP")

## Here we generate calibration data using a true calibration
## parameter, u, and then evaluate log posterior probabities; 
## the discrep.est documentation repeats this with by first calling
## aGP.seq and then discrep.est.  The answers should be idential, however
## note that a call first to example("fcalib") and then 
## example("discrep.est") will generate two random data sets, causing
## the results not to match

## begin data-generation code identical to aGP.seq, discrep.est, fcalib
## example sections and demo("calib")

## M: computer model test functon used in Goh et al, 2013 (Technometrics)
## an elaboration of one from Bastos and O'Hagan, 2009 (Technometrics) 
M <- function(x,u) 
  {
    x <- as.matrix(x)
    u <- as.matrix(u)
    out <- (1-exp(-1/(2*x[,2]))) 
    out <- out * (1000*u[,1]*x[,1]^3+1900*x[,1]^2+2092*x[,1]+60) 
    out <- out / (100*u[,2]*x[,1]^3+500*x[,1]^2+4*x[,1]+20)  
    return(out)
  }
  
## bias: discrepancy function from Goh et al, 2013 
bias <- function(x) 
  {
    x<-as.matrix(x)   
    out<- 2*(10*x[,1]^2+4*x[,2]^2) / (50*x[,1]*x[,2]+10)
    return(out)
  }

## beta.prior: marginal beta prior for u, 
## defaults to a mode at 1/2 in hypercube
beta.prior <- function(u, a=2, b=2, log=TRUE)
{
  if(length(a) == 1) a <- rep(a, length(u))
  else if(length(a) != length(u)) stop("length(a) must be 1 or length(u)")
  if(length(b) == 1) b <- rep(b, length(u))
  else if(length(b) != length(u)) stop("length(b) must be 1 or length(u)")
  if(log) return(sum(dbeta(u, a, b, log=TRUE)))
  else return(prod(dbeta(u, a, b, log=FALSE)))
}

## tgp for LHS sampling
library(tgp)
rect <- matrix(rep(0:1, 4), ncol=2, byrow=2)

## training inputs
ny <- 50; 
X <- lhs(ny, rect[1:2,])    ## computer model train

## true (but unknown) setting of the calibration parameter
## for the computer model
u <- c(0.2, 0.1)
Zu <- M(X, matrix(u, nrow=1)) 

## field data response, biased and replicated
sd <- 0.5
## Y <- computer output + bias + noise
reps <- 2 ## example from paper uses reps <- 10
Y <- rep(Zu,reps) + rep(bias(X),reps) + rnorm(reps*length(Zu), sd=sd) 
## variations: remove the bias or change 2 to 1 to remove replicates

## computer model design
nz <- 10000
XU <- lhs(nz, rect)
nth <- 1 ## number of threads to use in emulation, demo uses 8

## augment with physical model design points 
## with various u settings
XU2 <- matrix(NA, nrow=10*ny, ncol=4)
for(i in 1:10) {
  I <- ((i-1)*ny+1):(ny*i)
  XU2[I,1:2] <- X
}
XU2[,3:4] <- lhs(10*ny, rect[3:4,])
XU <- rbind(XU, XU2)

## evaluate the computer model
Z <- M(XU[,1:2], XU[,3:4])

## flag indicating if estimating bias/discrepency or not
bias.est <- TRUE
## two passes of ALC with MLE calculations for aGP.seq
methods <- rep("alcray", 2) ## demo uses rep("alc", 2)

## set up priors
da <- d <- darg(NULL, XU)
g <- garg(list(mle=TRUE), Y) 

## end identical data generation code

## now calculate log posterior for true calibration parameter 
## value (u).  You could repeat this for an estimate value 
## from demo("calib"), for example u.hat <- c(0.8236673, 0.1406989)

fcalib(u, XU, Z, X, Y, da, d, g, beta.prior, methods, M, bias.est, nth)
}


\keyword{ nonparametric }
\keyword{ nonlinear }
\keyword{ smooth }
\keyword{ models }
\keyword{ regression }
\keyword{ spatial }
\keyword{ optimize }
\keyword{ design }
