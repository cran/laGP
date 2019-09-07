## duplicates one iteration of comparison on the GSBP problem 
## from Picheny, et al (2016) paper;  This code was called
## 100 times (with end=150) in a MC experiment to obtain the
## figures in that paper

library(DiceOptim)  ## required for branin function

goldstein.price <- function(X) 
{
    if(is.null(nrow(X))) X <- matrix(X, nrow=1)

    m <- 8.6928
    s <- 2.4269
    x1 <- 4 * X[,1] - 2
    x2 <- 4 * X[,2] - 2
    a <- 1 + (x1 + x2 + 1)^2 * (19 - 14 * x1 + 3 * x1^2 - 14 * 
        x2 + 6 * x1 * x2 + 3 * x2^2)
    b <- 30 + (2 * x1 - 3 * x2)^2 * (18 - 32 * x1 + 12 * x1^2 + 
        48 * x2 - 36 * x1 * x2 + 27 * x2^2)
    f <- log(a * b)
    f <- (f - m)/s
    return(f)
}

toy.c1 <- function(X) {
  if (is.null(dim(X))) X <- matrix(X, nrow=1)
  c1 <- 3/2 - X[,1] - 2*X[,2] - .5*sin(2*pi*(X[,1]^2 - 2*X[,2]))
  return(cbind(c1, -apply(X, 1, branin) + 25))
}

parr <- function(X){
  if (is.null(dim(X))) X <- matrix(X, nrow=1)

  x1 <- (2 * X[,1] - 1)
  x2 <- (2 * X[,2] - 1)
  g <- (4-2.1*x1^2+1/3*x1^4)*x1^2 + x1*x2 + (-4+4*x2^2)*x2^2+3*sin(6*(1-x1)) + 3*sin(6*(1-x2))
  return(-g+6)
}

gsbp.constraints <- function(x){
  return(cbind(toy.c1(x), parr(x)-2))
}

## problem definition for AL
gsbpprob <- function(X, known.only=FALSE)
{ 
  if(is.null(nrow(X))) X <- matrix(X, nrow=1)
  if(known.only) stop("known.only not supported for this example")
  f <- goldstein.price(X)
  C <- gsbp.constraints(X)
  return(list(obj=f, c=cbind(C[,1], C[,2]/100, C[,3]/10)))
}

## problem definition for EFI
gsbpprob.efi <- function(X, known.only=FALSE)
{ 
  if(is.null(nrow(X))) X <- matrix(X, nrow=1)
  if(known.only) stop("known.only not supported for this example")
  f <- goldstein.price(X)
  C <- gsbp.constraints(X)

  return(list(obj=f, c=cbind(C[,1],C[,2]/100,-C[,2]/100,C[,3]/10,-C[,3]/10)))
}


## set bounding rectangle for aquisitions
dim <- 2
B <- matrix(c(rep(0,dim),rep(1,dim)),ncol=2)
ncandf <- function(t) { 1000 }

## original AL adapted for mixed constriants
AL <- optim.auglag(gsbpprob, B, equal=c(0,1,1), fhat=TRUE, 
              urate=5, ncandf=ncandf)

## EFI comparator with "-h,h" trick
EFI <- optim.efi(gsbpprob.efi, B, urate=5, fhat=TRUE, ncandf=ncandf)

## slack-variable AL
ALslack <- optim.auglag(gsbpprob, B, equal=c(0,1,1), fhat=TRUE, urate=5, 
              slack=TRUE, ncandf=ncandf)

## slack-variable AL, finishing with L-BFGSB
ALslack2 <- optim.auglag(gsbpprob, B, equal=c(0,1,1), fhat=TRUE, urate=5, 
              slack=2, ncandf=ncandf)

## replace infinite progress with 2.1 for visuals
AL$prog[is.infinite(AL$prog)] <- 2.1
EFI$prog[is.infinite(EFI$prog)]  <- 2.1
ALslack$prog[is.infinite(ALslack$prog)] <- 2.1 
ALslack2$prog[is.infinite(ALslack2$prog)] <- 2.1 

## visuals
matplot(cbind(AL$prog, EFI$prog, ALslack$prog, ALslack2$prog),
  type="l", lwd=2)
legend("topright", c("AL", "EFI", "ALslack", "ALslack + L-BFGS-B"), 
    lty=1:4, col=1:4, lwd=2)
