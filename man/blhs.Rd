\name{blhs}
\alias{blhs}
\alias{blhs.loop}

\title{
  Bootstrapped block Latin hypercube subsampling
}

\description{
  Provides bootstrapped block Latin hypercube subsampling under a given 
  data set to aid in consistent estimation of a global separable lengthscale 
  parameter
}

\usage{
  blhs(y, X, m)
  blhs.loop(y, X, m, K, da, g = 1e-3, maxit = 100, verb = 0, plot.it = FALSE)
}

\arguments{
  \item{y}{ a vector of responses/dependent values with \code{length(y) = nrow(X)}}
  \item{X}{ a \code{matrix} or \code{data.frame} containing the full (large) design matrix of input locations}
  \item{m}{ a positive scalar integer giving the number of divisions on each coordinate of input space defining the block structure }
  \item{K}{ a positive scalar integer specifying the number of Bootstrap replicates desired }
  \item{da}{ a lengthscale prior, say as generated by \code{\link{darg}} }
  \item{g}{ a positive scalar giving the fixed nugget value of the nugget parameter; by default \code{g = 1e-3}}
  \item{maxit}{ a positive scalar integer giving the maximum number of iterations for MLE calculations via \code{"L-BFGS-B"}; 
      see \code{\link{mleGPsep}} for more details}
  \item{verb}{ a non-negative integer specifying the verbosity level; \code{verb = 0} (by default) is quiet, 
      and larger values cause more progress information to be printed to the screen}
  \item{plot.it}{ \code{plot.it = FALSE} by default; if \code{plot.it = TRUE}, then each of the \code{K} 
      lengthscale estimates from bootstrap iterations will be shown 
      via \code{\link{boxplot}}}
}

\details{
  Bootstrapped block Latin hypercube subsampling (BLHS) yields a global lengthscale estimator 
  which is asymptotically consistent with the MLE calculated on the full data set. However, since it
  works on data subsets, it comes at a much reduced computational cost.  Intuitively, the BLHS 
  guarantees a good mix of short and long pairwise distances. A single bootstrap LH subsample 
  may be obtained by dividing each dimension of the input space equally into \code{m} 
  intervals, yielding \eqn{m^d}{m^d} mutually exclusive hypercubes.  It is easy to show 
  that the average number of observations in each hypercube is \eqn{Nm^{-d}}{N*m^(-d)} 
  if there are \eqn{N}{N} samples in the original design. From each of these hypercubes, 
  \code{m} \code{blocks} are randomly selected following the LH paradigm, i.e., so that 
  only one interval is chosen from each of the \code{m} segments. The average number of 
  observations in the subsample, combining the \code{m} randomly selected blocks, 
  is \eqn{Nm^{-d+1}}{N*m^(-d+1)}. 

  Ensuring a subsample size of at least \code{one} requires having \eqn{m\leq N^{\frac{1}{d-1}}}{m <= N^[1/(d-1)]}, 
  thereby linking the parameter \code{m} to computational effort.  Smaller \code{m} is preferred so long 
  as GP inference on data of that size remains tractable.  Since the blocks follow 
  an LH structure, the resulting sub-design inherits the usual LHS properties, 
  e.g., retaining marginal properties like univariate stratification modulo features present in the original, 
  large \code{N}, design. 

  For more details, see Liu (2014), Zhao, et al. (2017) and Sun, et al. (2019).
  
  \code{blhs} returns the subsampled input space and the corresponding responses. 
  
  \code{blhs.loop} returns the median of the \code{K} lengthscale maximum likelihood estimates, the subsampled data size to which
  that corresponds, and the subsampled data, including the input space and the responses, from the bootstrap iterations
}

\value{
\code{blhs} returns
  \item{\code{xs}}{the subsampled input space}
  \item{\code{ys}}{the subsampled responses, \code{length(ys) = nrow(xs)}}

\code{blhs.loop} returns
  \item{\code{that}}{the lengthscale estimate (median), \code{length(that) = ncol(X)}}
  \item{\code{ly}}{the subsampled data size (median)}
  \item{\code{xm}}{the subsampled input space (median)}
  \item{\code{ym}}{the subsampled responses (median)}
}

\note{
  This implementation assums that \code{X} has been coded to the unit cube (\eqn{[0,1]^p}{[0,1]^p}),
  where \code{p = ncol(X)}.
  
  \code{X} should be relatively homogeneous. A space-filling design (input) \code{X}
  is ideal, but not required
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
  
  Y. Zhao, Y. Hung, and Y. Amemiya (2017).
  \emph{Efficient Gaussian Process Modeling using Experimental Design-Based Subagging},
  Statistica Sinica, to appear;
  
  Yufan Liu (2014)
  \emph{Recent Advances in Computer Experiment Modeling}.
  Ph.D. Thesis at Rutgers, The State University of New Jersey.
  \url{https://dx.doi.org/doi:10.7282/T38G8J1H}
}

\author{
  Robert B. Gramacy \email{rbg@vt.edu} and Furong Sun \email{furongs@vt.edu}
}

\examples{
  # input space based on latin-hypercube sampling (not required)
  # two dimensional example with N=216 sized sample
  if(require(lhs)) { X <- randomLHS(216, 2)  
  } else { X <- matrix(runif(216*2), ncol=2) }
  # pseudo responses, not important for visualizing design
  Y <- runif(216) 
  
  ## BLHS sample with m=6 divisions in each coordinate
  sub <- blhs(y=Y, X=X, m=6)
  Xsub <- sub$xs # the bootstrapped subsample
  
  # visualization
  plot(X, xaxt="n", yaxt="n", xlim=c(0,1), ylim=c(0,1), xlab="factor 1", 
    ylab="factor 2", col="cyan", main="BLHS")
  b <- seq(0, 1, by=1/6)
  abline(h=b, v=b, col="black", lty=2)
  axis(1, at=seq (0, 1, by=1/6), cex.axis=0.8, 
    labels=expression(0, 1/6, 2/6, 3/6, 4/6, 5/6, 1))
  axis(2, at=seq (0, 1, by=1/6), cex.axis=0.8, 
    labels=expression(0, 1/6, 2/6, 3/6, 4/6, 5/6, 1), las=1)
  points(Xsub, col="red", pch=19, cex=1.25)
  
  ## Comparing global lengthscale MLE based on BLHS and random subsampling
  \dontrun{
    # famous borehole function
    borehole <- function(x){
      rw <- x[1] * (0.15 - 0.05) + 0.05
      r <-  x[2] * (50000 - 100) + 100
      Tu <- x[3] * (115600 - 63070) + 63070
      Tl <- x[5] * (116 - 63.1) + 63.1
      Hu <- x[4] * (1110 - 990) + 990
      Hl <- x[6] * (820 - 700) + 700
      L <-  x[7] * (1680 - 1120) + 1120
      Kw <- x[8] * (12045 - 9855) + 9855
      m1 <- 2 * pi * Tu * (Hu - Hl)
      m2 <- log(r / rw)
      m3 <- 1 + 2*L*Tu/(m2*rw^2*Kw) + Tu/Tl
      return(m1/m2/m3)
    }
    
    N <- 100000                   # number of observations
    if(require(lhs)) { xt <- randomLHS(N, 8)   # input space
    } else { xt <- matrix(runif(N*8), ncol=8) }
    yt <- apply(xt, 1, borehole)  # response
    colnames(xt) <- c("rw", "r", "Tu", "Tl", "Hu", "Hl", "L", "Kw")

    ## prior on the GP lengthscale parameter
    da <- darg(list(mle=TRUE, max=100), xt)

    ## make space for two sets of boxplots
    par(mfrow=c(1,2))
    
    # BLHS calculating with visualization of the K MLE lengthscale estimates
    K <- 10  # number of Bootstrap samples; Sun, et al (2017) uses K <- 31
    sub_blhs <- blhs.loop(y=yt, X=xt, K=K, m=2, da=da, maxit=200, plot.it=TRUE)
  
    # a random subsampling analog for comparison
    sn <- sub_blhs$ly # extract a size that is consistent with the BLHS
    that.rand <- matrix(NA, ncol=8, nrow=K)
    for(i in 1:K){
      sub <- sample(1:nrow(xt), sn)
      gpsepi <- newGPsep(xt[sub,], yt[sub], d=da$start, g=1e-3, dK=TRUE)
      mle <- mleGPsep(gpsepi, tmin=da$min, tmax=10*da$max, ab=da$ab, maxit=200)
      deleteGPsep(gpsepi)
      that.rand[i,] <- mle$d
    }

    ## put random boxplots next to BLHS ones
    boxplot(that.rand, xlab="input", ylab="theta-hat", col=2, 
      main="random subsampling")
  }
}

\keyword{design}
\keyword{models}
\keyword{spatial}
\keyword{regression}
\keyword{nonparametric}
