\name{randLine}
\alias{randLine}

\title{
  Generate two-dimensional random paths
}

\description{
  Generate two-dimensional random paths (one-dimensional manifolds in 2d)
  comprising of different randomly chosen line types: linear, quadratic,
  cubic, exponential, and natural logarithm. If the input dimensionality 
  is higher than 2, then a line in two randomly chosen input coordinates 
  is generated }

\usage{
  randLine(a, D, N, smin, res)
}

\arguments{
  \item{a}{ a fixed two-element vector denoting the range of the bounding box (lower bound and upper bound) of all input coordinates }
  \item{D}{ a scalar denoting the dimensionality of input space }
  \item{N}{ a scalar denoting the desired total number of random lines }
  \item{smin}{ a scalar denoting the minimum absolute scaling constant, i.e., the length of the shortest line that could be generated }
  \item{res}{ a scalar denoting the number of data points, i.e., the resolution on the random path }
}

\details{
  This two-dimensional random line generating function produces different
  types of \code{2d} random paths, including linear, quadratic, cubic,
  exponential, and natural logarithm.
  
  First, one of these line types is chosen uniformly at random. The line is
  then drawn, via a collection of discrete points, from the origin according 
  to the arguments, e.g., resolution and length, provided by the user. The
  discrete set of coordinates are then shifted and scaled, uniformly at
  random, into the specified 2d rectangle, e.g., \eqn{[-2,2]^2}{[-2,2]^2}, with
  the restriction that at least half of the points comprising the line lie
  within the rectangle.
  
  For a quick visualization, see Figure 15 in Sun, et al. (2017). Figure 7 
  in the same manuscript illustrates the application of this function in
  out-of-sample prediction using \code{\link{laGP}}, in \code{2d} and \code{4d}, respectively.
  
  \code{randLine} returns different types of random paths and the indices of
  the randomly selected pair, i.e., subset, of input coordinates (when \code{D > 2}).
}

\value{
  \code{randLine} returns a \code{list} of \code{list}s.  The outer list is of
  length six, representing each of the five possible line types (linear, quadratic, 
  cubic, exponential, and natural logarithm), with the sixth entry providing the 
  randomly chosen input dimensions.
  
  The inner \code{list}s are comprised of \eqn{res \times 2}{res * 2}
  \code{data.frame}s, the number of which span \code{N} samples across all
  inner \code{list}s.
}

\references{
  F. Sun, R.B. Gramacy, B. Haaland, E. Lawrence, and A. Walker (2017)
  \emph{Emulating satellite drag from large simulation experiments};
  preprint on arXiv:1712.00182. \url{http://arxiv.org/abs/1712.00182}
}

\author{
  Furong Sun \email{furongs@vt.edu} and Robert B. Gramacy \email{rbg@vt.edu}
}

\note{
  Users should scale each coordinate of global input space to the same coded
  range, e.g., \eqn{[-2,2]^D}{[-2,2]^D}, in order to avoid computational burden
  caused by passing global input space argument. Users may convert back to the 
  natural units when necessary.
}

\seealso{
  \code{\link{laGP}}, \code{\link{laGPsep}}
}

\examples{
## 1. visualization of the randomly generated paths

## generate the paths
D <- 4
a <- c(-2, 2)
N <- 30
smin <- 0.1
res <- 100
line.set <- randLine(a=a, D=D, N=N, smin=smin, res=res)
  
## the indices of the randomly selected pair of input coordinates
d <- line.set$d

## visualization

## first create an empty plot
plot(0, xlim=a, ylim=a, type="l", xlab=paste("factor ", d[1], sep=""), 
     ylab=paste("factor ", d[2], sep=""), main="2d random paths")
abline(h=(a[1]+a[2])/2, v=(a[1]+a[2])/2, lty=2)
  
## merge each path type together
W <- unlist(list(line.set$lin, line.set$qua, line.set$cub, line.set$ep, line.set$ln), 
  recursive=FALSE)

## calculate colors to retain 
n <- unlist(lapply(line.set, length)[-6])
cols <- rep(c("orange", "blue", "forestgreen", "magenta", "cornflowerblue"), n)
  
#* plot randomly generated paths with a centering dot in red at the midway point
for(i in 1:N){
  lines(W[[i]][,1], W[[i]][,2], col=cols[i])
  points(W[[i]][res/2,1], W[[i]][res/2,2], col=2, pch=20)
}
  
## 2. use the random paths for out-of-sample prediction via laGPsep
  
## test function (same 2d function as in other examples package)
## (ignoring 4d nature of path generation above)
f2d <- function(x, y=NULL){
  if(is.null(y)){
     if(!is.matrix(x) && !is.data.frame(x)) x <- matrix(x, ncol=2)
     y <- x[,2]; x <- x[,1]
  }
  g <- function(z)
  return(exp(-(z-1)^2) + exp(-0.8*(z+1)^2) - 0.05*sin(8*(z+0.1)))
  z <- -g(x)*g(y)
}
    
## generate training data using 2d input space
x <- seq(a[1], a[2], by=0.02)
X <- as.matrix(expand.grid(x, x))
Y <- f2d(X)

## example of joint path calculation folowed by RMSE calculation
## on the first random path
WW <- W[[sample(1:N, 1)]]
WY <- f2d(WW)

## exhaustive search via ``joint" ALC
j.exh <- laGPsep(WW, 6, 100, X, Y, method="alcopt", close=10000, lite=FALSE)
sqrt(mean((WY - j.exh$mean)^2)) ## RMSE

## repeat for all thirty path elements (way too slow for checking) and other
## local design choices and visualize RMSE distribution(s) side-by-side
\dontrun{   
  ## pre-allocate to save RMSE
  rmse.exh <- rmse.opt <- rmse.nn <- rmse.pw <- rmse.pwnn <- rep(NA, N)
  for(t in 1:N){
     
    WW <- W[[t]]
    WY <- f2d(WW)
       
    ## joint local design exhaustive search via ALC
    j.exh <- laGPsep(WW, 6, 100, X, Y, method="alc", close=10000, lite=FALSE)
    rmse.exh[t] <- sqrt(mean((WY - j.exh$mean)^2))
     
    ## joint local design gradient-based search via ALC
    j.opt <- laGPsep(WW, 6, 100, X, Y, method="alcopt", close=10000, lite=FALSE)
    rmse.opt[t] <- sqrt(mean((WY - j.opt$mean)^2))
    
    ## joint local design exhaustive search via NN
    j.nn <- laGPsep(WW, 6, 100, X, Y, method="nn", close=10000, lite=FALSE)
    rmse.nn[t] <- sqrt(mean((WY - j.nn$mean)^2))
     
    ## pointwise local design via ALC
    pw <- aGPsep(X, Y, WW, start=6, end=50, d=list(max=20), method="alc", verb=0)
    rmse.pw[t] <- sqrt(mean((WY - pw$mean)^2))
     
    ## pointwise local design via NN
    pw.nn <- aGPsep(X, Y, WW, start=6, end=50, d=list(max=20), method="nn", verb=0)   
    rmse.pwnn[t] <- sqrt(mean((WY - pw.nn$mean)^2))
     
    ## progress meter
    print(t)
  }
  
  ## justify the y range
  ylim_RMSE <- log(range(rmse.exh, rmse.opt, rmse.nn, rmse.pw, rmse.pwnn))
     
  ## plot the distribution of RMSE output
  boxplot(log(rmse.exh), log(rmse.opt), log(rmse.nn), log(rmse.pw), log(rmse.pwnn),
          xaxt='n', xlab="", ylab="log(RMSE)", ylim=ylim_RMSE, main="")
  axis(1, at=1:5, labels=c("ALC-ex", "ALC-opt", "NN", "ALC-pw", "NN-pw"), las=1)
}
}

\keyword{spatial}
\keyword{random}
