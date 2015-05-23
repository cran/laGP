\name{discrep.est}
\alias{discrep.est}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Estimate Discrepency in Calibration Model
}
\description{
Estimates the Gaussian process discrepency/bias and/or noise term in
a modularized calibration of a computer model (emulator) to field data,
and returns the log likelihood or posterior probability
}
\usage{
discrep.est(X, Y, Yhat, d, g, bias = TRUE, clean = TRUE)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{X}{
  a \code{matrix} or \code{data.frame} containing
  a design matrix of input locations for field data sites.  Any columns
  of \code{X} without at least three unique input settings are dropped
  in a pre-processing step
}
  \item{Y}{
	a vector values with \code{length(Y) = ncol(X)}
	containing the response from field data observations
	at \code{X}.  A \code{Y}-vector with \code{length(Y) = k*ncol(X)},
	for positive integer \code{k}, can be supplied in which case
	the multiple code \code{Y}-values will be treated as replicates
	at the \code{X}-values
}
  \item{Yhat}{
  a vector with \code{length(Yhat) = length(Y)} containing
  predictions at \code{X} from an emulator of a computer simulation
}
  \item{d}{
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
    at its starting value, which may be appropriate for emulating 
    deterministic computer code output. When specifying initial values, a
    vector of length \code{nrow(XX)} can be provided, giving a
    different initial value for each predictive location
}
  \item{bias}{
  a scalar logical indicating if a (isotropic) 
  GP discrepency should be estimated (\code{TRUE})
  or a Gaussian noise term only (\code{FALSE})
}
  \item{clean}{
	a scalar logical indicating if the C-side GP object should be freed before
	returning.
}
}
\details{
Estimates an isotropic Guassian correlation Gaussian process (GP) discrepency
term for the difference between a computer model output (\code{Yhat}) and
field data obsevations (\code{Y}) at locations \code{X}. The computer model
predictions would typically come from a GP emulation from simulation data,
possibly via \code{\link{aGP}} if the computer experiment is large.

This function is used primarly as a subroutine by \code{\link{fcalib}} which
defines an objective function for optimization in order to solve the
calibration problem via the method described by Gramacy, et al. (2014),
designed for large computer experiments.  However, once calibration is
performed this function can be useful for making comparisons to other methods.
Examples are provided in the \code{\link{fcalib}} documentation.

When \code{bias=FALSE} no discrepency is estimated; only a zero-mean 
Gaussian error distribution is assumed
}
\value{
The output object is comprised of the output of \code{jmleGP}, applied to a GP
object built with responses \code{Y - Yhat}.  That object is augmented with
a log likelihood, in \code{$ll}, and with a GP index \code{$gpu} when
\code{clean=FALSE}.  When \code{bias = FALSE} the output object retains the
same form as above, except with dummy zero-values since calling \code{jmleGP} is not
required.
}
\references{
R.B. Gramacy, D. Bingham, JP. Holloway, M.J. Grosskopf, C.C. Kuranz, E. Rutter, 
M. Trantham, and P.R. Drake (2014). \emph{Calibrating a large computer 
experiment simulating radiative shock hydrodynamics.}
Preprint on arXiv:1410.3293 
\url{http://arxiv.org/abs/1410.3293}

F. Liu, M. Bayarri and J. Berger (2009). 
\emph{Modularization in Bayesian analysis, with emphasis on analysis of computer 
models.} Bayesian Analysis, 4, 1, 119-150.
}
\author{
  Robert B. Gramacy \email{rbgramacy@chicagobooth.edu}
}

\seealso{
\code{vignette("laGP")},
\code{\link{jmleGP}}, \code{\link{newGP}}, \code{\link{aGP.seq}}, \code{\link{fcalib}} 
}
\examples{
## the example here combines aGP.seq and discrep.est functions; 
## it is comprised of snippets from demo("calib"), which contains
## code from the Calibration Section of vignette("laGP")

## Here we generate calibration data using a true calibration
## parameter, u, and then evaluatew log posterior probabities
## and out-of-sample RMSEs for that u value;  the fcalib 
## documentation repeats this with a single call to fcalib rather 
## than first aGP.seq and then discrep.est

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

## training and testing inputs
ny <- 50; nny <- 1000  
X <- lhs(ny, rect[1:2,])    ## computer model train
XX <- lhs(nny, rect[1:2,],) ## test

## true (but unknown) setting of the calibration parameter
## for the computer model
u <- c(0.2, 0.1)
Zu <- M(X, matrix(u, nrow=1)) 
ZZu <- M(XX, matrix(u, nrow=1)) 

## field data response, biased and replicated
sd <- 0.5
## Y <- computer output + bias + noise
reps <- 2 ## example from paper uses reps <- 10
Y <- rep(Zu,reps) + rep(bias(X),reps) + rnorm(reps*length(Zu), sd=sd) 
YYtrue <- ZZu + bias(XX) 
## variations: remove the bias or change 2 to 1 to remove replicates

## computer model design
nz <- 10000
XT <- lhs(nz, rect)
nth <- 1 ## number of threads to use in emulation, demo uses 8

## augment with physical model design points 
## with various u settings
XT2 <- matrix(NA, nrow=10*ny, ncol=4)
for(i in 1:10) {
  I <- ((i-1)*ny+1):(ny*i)
  XT2[I,1:2] <- X
}
XT2[,3:4] <- lhs(10*ny, rect[3:4,])
XT <- rbind(XT, XT2)

## evaluate the computer model
Z <- M(XT[,1:2], XT[,3:4])

## flag indicating if estimating bias/discrepency or not
bias.est <- TRUE
## two passes of ALC with MLE calculations for aGP.seq
methods <- rep("alcray", 2) ## demo uses rep("alc", 2)

## set up priors
da <- d <- darg(NULL, XT)
g <- garg(list(mle=TRUE), Y) 

## end identical data generation code

## now calculate log posterior and do out-of-sample RMSE calculation
## for true calibration parameter value (u).  You could repeat
## this for an estimate value from demo("calib"), for example
## u.hat <- c(0.8236673, 0.1406989)

## first log posterior

## emulate at field-data locations Xu
Xu <- cbind(X, matrix(rep(u, ny), ncol=2, byrow=TRUE))
ehat.u <- aGP.seq(XT, Z, Xu, da, methods, ncalib=2, omp.threads=nth, verb=0)

## estimate discrepency from the residual
cmle.u <- discrep.est(X, Y, ehat.u$mean, d, g, bias.est, FALSE)
cmle.u$ll <- cmle.u$ll + beta.prior(u)
print(cmle.u$ll)
## compare to same caluclation with u.hat above

## now RMSE
\dontrun{
## predictive design with true calibration parameter
XXu <- cbind(XX, matrix(rep(u, nny), ncol=2, byrow=TRUE))

## emulate at predictive design
ehat.oos.u <- aGP.seq(XT, Z, XXu, da, methods, ncalib=2, 
  omp.threads=nth, verb=0)

## predict via discrepency
YYm.pred.u <- predGP(cmle.u$gp, XX)

## add in emulation
YY.pred.u <- YYm.pred.u$mean + ehat.oos.u$mean

## calculate RMSE
rmse.u <- sqrt(mean((YY.pred.u - YYtrue)^2))
print(rmse.u)
## compare to same caluclation with u.hat above

## clean up
deleteGP(cmle.u$gp)
}
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
\keyword{ design }
