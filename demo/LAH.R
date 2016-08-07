## duplicates one iteration of comparison on the LAH problem 
## from Picheny, et al (2016) paper;  This code was called
## 100 times in a MC experiment to obtain the figures in that paper

library(DiceOptim) ## for hartman4 function

hartman4v <- function(x)
{ 
  if (is.null(dim(x))) x <- matrix(x, nrow=1) 
  return(-apply(x, 1, hartman4))
}

ackley <- function(x)
{ 
  if (is.null(dim(x))) x <- matrix(x, nrow=1) 
  n <- ncol(x) 
  x <- x*3 - 1  
  a=20; b=0.2; c=2*pi; 
  return(-a*exp(-b*sqrt(1/n*rowSums(x^2))) - exp(1/n*rowSums(cos(c*x))) + a + exp(1))
}

lah.constraints <- function(x)
{ 
  return(c(hartman4v(x),-ackley(x)+3))
}

## problem definition for AL
lahprob <- function(X, known.only=FALSE)
{ 
  if(is.null(nrow(X))) X <- matrix(X, nrow=1)
  
  f <- rowSums(X)
  if(known.only) return(list(obj=f))

  cs <- lah.constraints(X)

  return(list(obj=f, c=cbind(cs[1],cs[2])))
}

## problem definition for EFI
lahprob.efi <- function(X, known.only=FALSE)
{ 
  if(is.null(nrow(X))) X <- matrix(X, nrow=1)
  f <- rowSums(X)

  if(known.only) return(list(obj=f))
  cs <- lah.constraints(X)

  return(list(obj=f, c=cbind(cs[1],-cs[1],cs[2])))
}


## set bounding rectangle for aquisitions
dim <- 4
B <- matrix(c(rep(0,dim),rep(1,dim)),ncol=2)
ncandf <- function(t) { 1000 }

## original AL adapted for mixed constriants
AL <- optim.auglag(lahprob, B, end=50, equal=c(1,0), urate=5, 
                  ncandf=ncandf) 

## EFI comparator with "-h,h" trick
EFI <- optim.efi(lahprob.efi, B, end=50, urate=5, ncandf=ncandf)

## slack-variable AL
ALslack <- optim.auglag(lahprob, B, end=50, equal=c(1,0), urate=5, 
                  slack=TRUE, ncandf=ncandf)

## slack-variable AL, finishing with L-BFGSB
ALslack2 <- optim.auglag(lahprob, B, end=50, equal=c(1,0), urate=5, 
                  slack=2, ncandf=ncandf)

## plot progress
progress <- function(out, eps=1e-2)
  {
    v <- out$C[,2] < 0 & abs(out$C[,1]) < eps
    vobj <- out$obj
    vobj[!v] <- 4
    for(i in 2:length(vobj)) 
      if(vobj[i] > vobj[i-1]) vobj[i] <- vobj[i-1] 
    return(vobj)
  }

AL$vobj <- progress(AL)
EFI$vobj <- progress(EFI)
ALslack$vobj <- progress(ALslack)
ALslack2$vobj <- progress(ALslack2)
matplot(cbind(AL$vobj, EFI$vobj, ALslack$vobj, ALslack2$vobj),
  type="l", lwd=2)
legend("left", c("AL", "EFI", "ALslack", "ALslack + L-BFGS-B"), 
    lty=1:4, col=1:4, lwd=2)