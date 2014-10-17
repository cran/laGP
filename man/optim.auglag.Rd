\name{optim.auglag}
\alias{optim.auglag}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
    Optimize a linear objective function under
    multiple blackbox constraints
}
\description{
    Uses a surrogate modeled augmented Lagrangian (AL) system to optimize
    a known linear function under unknown multiple (blackbox) constraints via
    expected improvement (EI) and variations
}
\usage{
optim.auglag(fn, B, start = 10, end = 100, Xstart = NULL, 
    ab = c(3/2, 4), lambda = rep(1, ncol(B)), rho = 1/2, 
    urate = 10, ncandf = function(t) { t }, dg.start = c(0.1, 1e-06), 
    dlim = sqrt(ncol(B)) * c(1/100, 10), obj.norm = 1, 
    tol = list(ei = 1e-05, ey = 0.05, its = 10), nomax = FALSE, 
    N = 1000, plotprog = FALSE, verb = 2, ...)
}
\arguments{
  \item{fn}{
    function of a single, vector-valued, input (\code{x}) returning a \code{list} 
    with elements \code{"obj"} containing the (scalar) objective value and \code{"c"} 
    returning a vector of evaluations of the (multiple) constraint function at \code{x}
}
  \item{B}{
    2-column \code{matrix} describing the bounding box.  The number of rows
    of the \code{matrix} determines the input dimension 
    (\code{length(x)} in \code{fn(x)}); the first column gives
    lower bounds and the second gives upper bounds
}
  \item{start}{
    positive integer giving the number of random starting locations before 
    sequential design (for optimization) is performed; \code{start >= 6} is
    recommended unless \code{Xstart} is non-\code{NULL}; in the current version
    the starting locations come from a space-filling design via \code{\link[tgp]{dopt.gp}}
}
  \item{end}{
    positive integer giving the total number of evaluations/trials in the 
    optimization; must have \code{end > start}
}
  \item{Xstart}{
    optional matrix of starting design locations in lieu of, or in addition to,
     \code{start} random ones;  recommend \code{nrow(Xstart) + start >= 6}; also must
     have \code{ncol(Xstart) = nrow(B)}
}
  \item{ab}{
    prior parameters; see \code{\link{darg}} describing the prior used on the
    lengthscale parameter during emulation(s) for the constraints
}
  \item{lambda}{
    \code{m}-dimensional initial Lagrangian parameter for \code{m}-constraints
}
  \item{rho}{
    positive scalar initial quadratic penalty parameter in the augmented Lagrangian
}
  \item{urate}{
    positive integer indicating  how many optimization trials should pass before
    each MLE/MAP update is performed for estimators for GP correlation lengthscale 
    parameter(s) 
}
  \item{ncandf}{
    function taking a single integer indicating the optimization trial number \code{t}, where 
    \code{start < t <= end}, and returning the number of search candidates (e.g., for
    expected improvement calculations) at round \code{t}; the default setting
    allows the number of candidates to grow linearly with \code{t}
}
  \item{dg.start}{
    2-vector giving starting values for the lengthscale and nugget parameters
    of the GP surrogate model(s) for constraints
}
  \item{dlim}{
    2-vector giving bounds for the lengthscale parameter(s) under MLE/MAP inference,
    thereby augmenting the prior specification in \code{ab}
}
  \item{obj.norm}{
    scalar indicating the relationship between the sum of the inputs \code{sum(x)} 
    to \code{fn} and the output \code{fn(x)$obj$}; note that at this time only linear
    objectives are supported by the code - more details below
}
  \item{tol}{
    \code{list} containing entries \code{"ei"}, \code{"ey"}, and \code{"its"} together
    describing the search method and the criteria used to determine approximate convergence
    of the inner loop of the augmented Lagrangian search.  \code{"ei"} describes 
    the tolerance determining approximate convergence when sampling under expected
    improvement (EI), i.e., whether the largest EI value obtained in the current search
    is too low to continue; \code{"ey"} is a proportion indicating how many of the EIs
    at \code{ncandf(t)} candidate locations must be non-zero to trust that metric to 
    guide search, reducing to an EY-based 
    search instead [choosing that proportion to be zero forces EY-based search]; 
    \code{"its"} is an integer specifying the maximum number of inner-loop iterations 
    that can pass without improvement in the sub-problem before approximate convergence 
    is declared
}
  \item{nomax}{
    one of \code{c{0,1,2}} indicating if the max should be removed from the 
    augmented lagrangian (AL): not at all (0),
    in the evaluation of EI or EY (1), or also in the update of \code{lambda} (2); 
    see the description for details
}
  \item{N}{
    positive scalar integer indicating the number of Monte Carlo samples to be
    used for calculating EI and EY
}
  \item{plotprog}{
    \code{logical} indicating if progress plots should be made after each inner iteration;
    the plots show three panels tracking the best valid objective, the EI or EY surface
    over the first two input variables (requires \code{\link[akima]{interp}}, 
    and the parameters of the lengthscale(s) of the GP(s) respectively.  When 
    \code{plotprog = TRUE} the \code{\link[tgp]{interp.loess}} function is used to
    aid in creating surface plots, however this does not work well with fewer than 
    fifteen points.  You may also provide a function as an argument, having similar
    arguments/formals as \code{\link[tgp]{interp.loess}}.  For example, we use
    \code{\link[akima]{interp}} below, which would have been the default if not 
    for licensing incompatibilities
}
  \item{verb}{
    positive scalar integer indicating the verbosity level; the larger the value the
    more that is printed to the screen
}
  \item{...}{ additional arguments passed to \code{fn}}
}
\details{
In its current form, this is an \dQuote{alpha} code illustrating the suite of
methods used to optimize two challenging constrained optimization problems
from Gramacy, et al. (2014); see references below.  

That scheme hybridizes Gaussian process based surrogate modeling and expected
improvement (EI; Jones, et., al, 2008) with the additive penalty method (APM)
implemented by the augmented Lagrangian (AL, e.g., Kannan and Wild, 2012).
The goal is to minimize a a known linear objective function \code{f(x)} under
multiple, unknown (blackbox) constraint functions satisfying \code{c(x) <= 0},
which is vector-valued.  The solution here emulates the components of \code{c}
with Gaussian process surrogates, and guides optimization by searching the
posterior mean surface, or the EI of, the following composite objective
\deqn{
    Y(x) = f(x) + \lambda^\top Y_c(x) + \frac{1}{2\rho} \sum_{i=1}^m 
\max(0, Y_{c_i}(x))^2,
}{ Y(x) = f(x) + lambda \%*\% Yc(x) + 1/(2rho) sum(max(0, Yc(x))^2)}  
where \eqn{\lambda}{lambda} and \eqn{\rho}{rho} follow updating equations that
guarantee convergence to a valid solution minimizing the objective.  For more
details, see Gramacy, et al. (2014).

The \code{nomax} argument indicates whether or not the \code{max} is present
in the AL formula above.  Setting \code{nomax > 0} can lead to a more
aggressive search nearby the boundary between feasible and infeasible regions.
See Gramacy, et al. (2014) for more details. 

The example below illustrates a variation on the toy problem considered in that paper,
which bases sampling on EI.
}
\value{
The output is a \code{list} summarizing the progress of the evaluations of the
blackbox under optimization

\item{prog }{ vector giving the best valid (\code{c(x) < 0}) value 
  of the objective over the trials }
\item{obj }{ vector giving the value of the objective for the input under consideration 
  at each trial }
\item{X }{ \code{matrix} giving the input calues at which the blackbox function was
evaluated }
\item{C }{ \code{matrix} giving the value of the constraint function for the input 
  under consideration at each trial (corresponding to \code{X} above)}
\item{d}{ \code{matrix} of lengthscale values obtained at each update of the GP 
  estimator(s), i.e., every \code{urate} iterations}
}
\references{

Gramacy, R.B, Gray, G.A, Lee, H.K.H, Le Digabel, S., Ranjan P., Wells, G., Wild, S.M. (2014)
\dQuote{Modeling an Augmented Lagrangian for Improved
  Blackbox Constrained Optimization}, Preprint available on arXiv:1403.4890;
  \url{http://arxiv.org/abs/1403.4890}

Jones, D., Schonlau, M., and Welch, W. J. (1998). 
\dQuote{Efficient Global Optimization of Expensive Black Box Functions.} 
\emph{Journal of Global Optimization}, 13, 455-492.

Kannan, A. and Wild, S. (2012). 
\dQuote{Benefits of Deeper Analysis in Simulation-based Groundwater 
Optimization Problems.} In \emph{Proceedings of the XIX International Conference on
Computational Methods in Water Resources} (CMWR 2012).

}
\author{
  Robert B. Gramacy \email{rbgramacy@chicagobooth.edu}
}
\note{
The current version of this function only supports linear objectives; 
future versions will deal with known monotone objectives.  The is \dQuote{alpha}
software.
}


\seealso{
\code{\link[tgp]{optim.step.tgp}} for unconstrained optimization;\cr
\code{optim} with \code{method="L-BFGS-B"} for box constraints, or
\code{optim} with \code{method="SANN"} for simulated annealing
}
\examples{
library(tgp) ## uses dopt.gp, will change later
library(akima) ## for plotprog=interp below

## a test function returning objective evaluations and constraints
aimprob <- function(x)
{
  f <- sum(x)
  c1 <- 1.5-x[1]-2*x[2]-0.5*sin(2*pi*(x[1]^2-2*x[2]))
  c2 <- x[1]^2+x[2]^2-1.5
  return(list(obj=f, c=c(c1,c2)))
}

## set bounding rectangle for adaptive sampling
B <- matrix(c(rep(0,2),rep(1,2)),ncol=2)

## optimization (primarily) by EI, change 25 to 100 for
## 99% chance of finding the global optimum with value 0.6
library(akima)
out <- optim.auglag(aimprob, B, ab=c(3/2,8), end=25, plotprog=interp)


## for comparison, here is a version that uses simulated annealing
## where the Additive Penalty Method (APM) is used to handle
## the constraints
\dontrun{
aimprob.apm <- function(x, B=matrix(c(rep(0,2),rep(1,2)),ncol=2))
{ 
  ## check bounding box
  for(i in 1:length(x)) {
    if(x[i] < B[i,1] || x[i] > B[i,2]) return(Inf)
  }

  ## evaluate objective and constraints
  f <- sum(x)
  c1 <- 1.5-x[1]-2*x[2]-0.5*sin(2*pi*(x[1]^2-2*x[2]))
  c2 <- x[1]^2+x[2]^2-1.5

  ## return APM composite
  return(f + abs(c1) + abs(c2))
}

## use SA; specify control=list(maxit=100), say, to control max 
## number of iterations; does not easily facilitate plotting progress
out <- optim(runif(2), aimprob.apm, method="SANN") 
## check the final value, which typically does not satisfy both
## constraints
aimprob(out$par)
}


}
%  one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ optimize }
\keyword{ design }% __ONLY ONE__ keyword per line
