\name{optim.auglag}
\alias{optim.auglag}
\alias{optim.efi}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
    Optimize an objective function under
    multiple blackbox constraints
}
\description{
    Uses a surrogate modeled augmented Lagrangian (AL) system to optimize
    an objective function (blackbox or known and linear) 
    under unknown multiple (blackbox) constraints via
    expected improvement (EI) and variations; a comparator based
    on EI with constraints is also provided
}
\usage{
optim.auglag(fn, B, fhat = FALSE, equal = FALSE, ethresh = 1e-2, 
  slack = FALSE, cknown = NULL, start = 10, end = 100, 
  Xstart = NULL, sep = TRUE, ab = c(3/2, 8), lambda = 1, rho = NULL, 
  urate = 10, ncandf = function(t) { t }, dg.start = c(0.1, 1e-06), 
  dlim = sqrt(ncol(B)) * c(1/100, 10), Bscale = 1, ey.tol = 1e-2, 
  N = 1000, plotprog = FALSE, verb = 2, ...)
optim.efi(fn, B, fhat = FALSE, cknown = NULL, start = 10, end = 100, 
  Xstart = NULL, sep = TRUE, ab = c(3/2,8), urate = 10, 
  ncandf = function(t) { t }, dg.start = c(0.1, 1e-6), 
  dlim = sqrt(ncol(B))*c(1/100,10), Bscale = 1, plotprog = FALSE, 
  verb = 2, ...)
}
\arguments{
  \item{fn}{
    function of an input (\code{x}), facilitating vectorization on a 
    \code{matrix} \code{X} thereof,  returning a \code{list} 
    with elements \code{$obj} containing the (scalar) objective value and \code{$c} 
    containing a vector of evaluations of the (multiple) constraint function at \code{x}.
    The \code{fn} function must take a \code{known.only} argument which is explained
    in the note below; it need not act on that argument
}
\item{B}{
    2-column \code{matrix} describing the bounding box.  The number of rows
    of the \code{matrix} determines the input dimension 
    (\code{length(x)} in \code{fn(x)}); the first column gives
    lower bounds and the second gives upper bounds
}
\item{fhat}{ a scalar logical indicating if the objective function should
  be modeled with a GP surrogate.  The default of \code{FALSE} assumes a known
  linear objective scaled by \code{Bscale}.  Using \code{TRUE} is an \dQuote{alpha}
  feature at this time}
\item{equal}{ an optional vector containing zeros and ones, whose length equals the number of
constraints, specifying which should be treated as equality constraints (\code{0}) and 
which as inequality (\code{1}) }
\item{ethresh}{ a threshold used for equality constraints to determine validity for 
progress measures; ignored if there are no equality constraints }
\item{slack}{ A scalar logical indicating if slack variables, and thus exact EI 
  calculations should be used.  The default of \code{slack = FALSE} results in Monte 
  Carlo EI approximation.  
  One can optionally specify \code{slack = 2} to get the \code{slack = TRUE} behavior,
  with a second-stage L-BFGS-B optimization of the EI acquisition applied at the end,
  starting from the best value found on the random search grid }
\item{cknown}{ A optional positive integer vector specifying which of the constraint
  values returned by \code{fn} should be treated as \dQuote{known}, i.e., not modeled
  with Gaussian processes}
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
     \code{start} random ones;  we recommend \code{nrow(Xstart) + start >= 6}; also must
     have \code{ncol(Xstart) = nrow(B)}
}
  \item{sep}{
    The default \code{sep = TRUE} uses separable GPs (i.e., via \code{\link{newGPsep}}, etc.)
    to model the constraints and objective; otherwise the isotropic GPs are used
  }
  \item{ab}{
    prior parameters; see \code{\link{darg}} describing the prior used on the
    lengthscale parameter during emulation(s) for the constraints
}
  \item{lambda}{
    \code{m}-dimensional initial Lagrange multiplier parameter for \code{m}-constraints
}
  \item{rho}{
    positive scalar initial quadratic penalty parameter in the augmented Lagrangian; the default setting of \code{rho = NULL} causes an automatic starting value to be chosen; see rejoinder to Gramacy, et al. (2016) or supplementary material to Picheny, et al. (2016)
}
  \item{urate}{
    positive integer indicating  how many optimization trials should pass before
    each MLE/MAP update is performed for GP correlation lengthscale 
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
  \item{Bscale}{
    scalar indicating the relationship between the sum of the inputs, \code{sum(x)}, 
    to \code{fn} and the output \code{fn(x)$obj}; note that at this time only linear
    objectives are fully supported by the code - more details below
}
  \item{ey.tol}{
    a scalar proportion indicating how many of the EIs
    at \code{ncandf(t)} candidate locations must be non-zero to \dQuote{trust}
     that metric to guide search, reducing to an EY-based search instead 
     [choosing that proportion to be one forces EY-based search]
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
    a non-negative integer indicating the verbosity level; the larger the value the
    more that is printed to the screen
}
  \item{...}{ additional arguments passed to \code{fn}}
}
\details{
These subroutines support a suite of
methods used to optimize challenging constrained problems
from Gramacy, et al. (2016); and from Picheny, et al., (2016) see references below.  

Those schemes hybridize Gaussian process based surrogate modeling and expected
improvement (EI; Jones, et., al, 2008) with the additive penalty method (APM)
implemented by the augmented Lagrangian (AL, e.g., Nocedal & Wright, 2006).
The goal is to minimize a (possibly known) linear objective function \code{f(x)} under
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
details, see Gramacy, et al. (2016).

A slack variable implementation that allows for exact EI calculations and can 
accommodate equality constraints, and mixed (equality and inequality) constraints,
 is also provided.  For further details, see Picheny, et al. (2016).

The example below illustrates a variation on the toy problem considered in both papers,
which bases sampling on EI.  For examples making used of equality constraints, 
following the Picheny, et al (2016) papers; see the demos listed in the 
\dQuote{See Also} section below.

Although it is off by default, these functions allow an unknown objective to
be modeled (\code{fhat = TRUE}), rather than assuming a known linear one.  For examples see
\code{demo("ALfhat")} and \code{demo("GSBP")} which illustrate the AL and comparators
in inequality and mixed constraints setups, respectively.

The \code{optim.efi} function is provided as a comparator.  This method uses
the same underlying GP models to with the hybrid EI and probability of satisfying
the constraints heuristic from Schonlau, et al., (1998).  See \code{demo("GSBP")}
and \code{demo("LAH")} for \code{optim.efi} examples and comparisons between 
the original AL, the slack variable enhancement(s) on mixed constraint
problems with known and blackbox objectives, respectively
}
\value{
The output is a \code{list} summarizing the progress of the evaluations of the
blackbox under optimization

\item{prog }{ vector giving the best valid (\code{c(x) < 0}) value 
  of the objective over the trials }
\item{obj }{ vector giving the value of the objective for the input under consideration 
  at each trial }
\item{X }{ \code{matrix} giving the input values at which the blackbox function was
evaluated }
\item{C }{ \code{matrix} giving the value of the constraint function for the input 
  under consideration at each trial}
\item{d }{ \code{matrix} of lengthscale values obtained at the final update of the 
  GP emulator for each constraint}
\item{df }{ if \code{fhat = TRUE} then this is a \code{matrix} of lengthscale values
for the objective obtained at the final update of the GP emulator}
\item{lambda }{ a \code{matrix} containing \code{lambda} vectors used in each \dQuote{outer
loop} AL iteration }
\item{rho }{ a vector of \code{rho} values used in each \dQuote{outer loop} AL iteration }
}
\references{
Gramacy, R. B. (2020) \emph{Surrogates: Gaussian Process Modeling,
Design and Optimization for the Applied Sciences}. Boca Raton,
Florida: Chapman Hall/CRC.  (See Chapter 7.)
\url{https://bobby.gramacy.com/surrogates/}

Picheny, V., Gramacy, R.B., Wild, S.M., Le Digabel, S. (2016). 
\dQuote{Bayesian optimization under mixed constraints
with a slack-variable augmented Lagrangian}. Preprint available on arXiv:1605.09466;
\url{https://arxiv.org/abs/1605.09466}

Gramacy, R.B, Gray, G.A, Lee, H.K.H, Le Digabel, S., Ranjan P., Wells, G., Wild, S.M. (2016)
\dQuote{Modeling an Augmented Lagrangian for Improved
  Blackbox Constrained Optimization}, \emph{Technometrics} (with discussion), 
  58(1), 1-11. Preprint available on arXiv:1403.4890;
  \url{https://arxiv.org/abs/1403.4890}

Jones, D., Schonlau, M., and Welch, W. J. (1998). 
\dQuote{Efficient Global Optimization of Expensive Black Box Functions.} 
\emph{Journal of Global Optimization}, 13, 455-492.

Schonlau, M., Jones, D.R., and Welch, W. J. (1998). \dQuote{Global Versus Local Search in
Constrained Optimization of Computer Models.} In \emph{New Developments and Applications
in Experimental Design}, vol. 34, 11-25. Institute of Mathematical Statistics.

Nocedal, J. and Wright, S.J. (2006). \emph{Numerical Optimization.}
2nd ed. Springer.

}
\author{
  Robert B. Gramacy \email{rbg@vt.edu}
}
\note{
This function is under active development, especially the newest features
including separable GP surrogate modeling, surrogate modeling of a
blackbox objective, and the use of slack variables for exact EI calculations and
the support if equality constraints. Also note that, compared with earlier versions, it is now
required to augment your blackbox function (\code{fn}) with an argument named
\code{known.only}. This allows the user to specify if a potentially different
object 
(with a subset of the outputs, those that are \dQuote{known}) gets returned in
certain circumstances. For example, the objective is treated as known in many of our
examples. When a non-null \code{cknown} object is used, the \code{known.only}
flag can be used to return only the outputs which are known.

Older versions of this function provided an argument called \code{nomax}.
The NoMax feature is no longer supported
}


\seealso{
\code{vignette("laGP")}, \code{demo("ALfhat")} for blackbox objective,
\code{demo("GSBP")} for a mixed constraints problem with blackbox objective,
\code{demo("LAH")} for mix constraints with known objective,
\code{\link[tgp]{optim.step.tgp}} for unconstrained optimization;
\code{optim} with \code{method="L-BFGS-B"} for box constraints, or
\code{optim} with \code{method="SANN"} for simulated annealing
}
\examples{
## this example assumes a known linear objective; further examples
## are in the optim.auglag demo

## a test function returning linear objective evaluations and 
## non-linear constraints
aimprob <- function(X, known.only = FALSE)
{
  if(is.null(nrow(X))) X <- matrix(X, nrow=1)
  f <- rowSums(X)
  if(known.only) return(list(obj=f))
  c1 <- 1.5-X[,1]-2*X[,2]-0.5*sin(2*pi*(X[,1]^2-2*X[,2]))
  c2 <- rowSums(X^2)-1.5
  return(list(obj=f, c=cbind(c1,c2)))
}

## set bounding rectangle for adaptive sampling
B <- matrix(c(rep(0,2),rep(1,2)),ncol=2)

## optimization (primarily) by EI, change 25 to 100 for
## 99% chance of finding the global optimum with value 0.6
library(akima) ## for plotprog=interp
out <- optim.auglag(aimprob, B, end=25, plotprog=interp)

## using the slack variable implementation which is a little slower
## but more precise; slack=2 augments random search with L-BFGS-B
  
out2 <- optim.auglag(aimprob, B, end=25, slack=TRUE)
\dontrun{
out3 <- optim.auglag(aimprob, B, end=25, slack=2)
}

## for more slack examples and comparison to optim.efi on problems
## involving equality and mixed (equality and inequality) constraints,
## see demo("ALfhat"), demo("GSBP") and demo("LAH")

## for comparison, here is a version that uses simulated annealing
## with the Additive Penalty Method (APM) for constraints
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
out4 <- optim(runif(2), aimprob.apm, method="SANN") 
## check the final value, which typically does not satisfy both
## constraints
aimprob(out4$par)
}

## for a version with a modeled objective see demo("ALfhat")
}
%  one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ optimize }
\keyword{ design }% __ONLY ONE__ keyword per line
