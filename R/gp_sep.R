#*******************************************************************************
#
# Local Approximate Gaussian Process Regression
# Copyright (C) 2013, The University of Chicago
#
# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with this library; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
#
# Questions? Contact Robert B. Gramacy (rbgramacy@chicagobooth.edu)
#
#*******************************************************************************


## newGPsep:
##
## build an initial separable GP representation on the C-side
## using the X-Z data and d/g paramterization.  

newGPsep <- function(X, Z, d, g)
  {
    n <- nrow(X)
    m <- ncol(X)
    if(is.null(n)) stop("X must be a matrix")
    if(length(Z) != n) stop("must have nrow(X) = length(Z)")
    if(length(d) != m) stop("must have length(d) = ncol(X)")
    
    out <- .C("newGPsep_R",
              m = as.integer(m),
              n = as.integer(n),
              X = as.double(t(X)),
              Z = as.double(Z),
              d = as.double(d),
              g = as.double(g),
              gpsepi = integer(1),
              package = "laGP")

    ## return C-side GP index
    return(out$gpsepi)
  }


## deleteGPsep:
##
## deletes the C-side of a particular separable GP

deleteGPsep <- function(gpsepi)
  {
    .C("deleteGPsep_R",
       gpsepi = as.integer(gpsepi), package="laGP")
    invisible(NULL)
  }


## deleteGPseps:
##
## deletes all gpseps on the C side

deleteGPseps <- function()
  {
    .C("deleteGPseps_R", package="laGP")
    invisible(NULL)
  }


## llikGPSEP:
##
## calculate the log likelihood of the GP

llikGPsep <- function(gpsepi, dab=c(0,0), gab=c(0,0))
  {
    r <- .C("llikGPsep_R",
            gpi = as.integer(gpsepi),
            dab = as.double(dab),
            gab = as.double(gab),
            llik = double(1),
            package = "laGP")

    return(r$llik)
  }


## getmGPsep
##
## acces the input dimension of a separable GP
##
## totall new to GPsep

getmGPsep <- function(gpsepi)
  {
    .C("getmGPsep_R", gpsepi = as.integer(gpsepi), m = integer(1), package="laGP")$m
  }


## getdGPsep
##
## acces the separable lengthscale of a separable gp
##
## totally new to GPsep

getdGPsep <- function(gpsepi)
  {
    m <- getmGPsep(gpsepi) 
    .C("getdGPsep_R", gpsepi = as.integer(gpsepi), d = double(m), package="laGP")$d
  }

## getgGPsep
##
## acces the input dimension of a separable GP
##
## totally new to GPsep

getgGPsep <- function(gpsepi)
  { 
    .C("getgGPsep_R", gpsepi = as.integer(gpsepi), g = double(1), package="laGP")$g
  }



## dllikGPsep:
##
## calculate the first and second derivative of the
## log likelihood of the GP with respect to d, the
## lengthscale parameter
##
## SIMILAR to dllikGP except with vector d and gpsep
## isntead of gp

dllikGPsep <- function(gpsepi, ab=c(0,0), param=c("d", "g"), d2nug=FALSE)
  {
    param <- match.arg(param)
    if(param == "d") {
      dim <- getmGPsep(gpsepi)
      r <- .C("dllikGPsep_R",
            gpsepi = as.integer(gpsepi),
            ab = as.double(ab),
            d = double(dim),
            package = "laGP")
      return(r$d)
    } else {
      if(d2nug) d2 <- 1
      else d2 <- 0
      r <- .C("dllikGPsep_nug_R",
            gpsepi = as.integer(gpsepi),
            ab = as.double(ab),
            d = double(1),
            d2 = as.double(d2),
            package = "laGP")
      if(d2nug) return(list(d=r$d, d2=r$r2))
      else return(r$d)
    }
  }



## newparamsGPsep:
##
## change the separable GP lengthscale and nugget parameerization
## (without destroying the object and creating a new one)

newparamsGPsep <- function(gpsepi, d, g=-1)
  {
    if(all(d <= 0) & g < 0) stop("one of d or g must be new")
    m <- getmGPsep(gpsepi)
    if(length(d) != m) stop("length(d) !=", m)

    r <- .C("newparamsGPsep_R",
            gpi = as.integer(gpsepi),
            d = as.double(d),
            g = as.double(g),
            package = "laGP")

    invisible(NULL)
  }


## mleGPsep:
##
## updates the separable GP to use its MLE lengthscale
## parameterization using the current data;
## 
## differs substantially from mleGP in that L-BFGS-B from
## optim is used to optimize over the separable lengthscale;
## an option is also provided to include the nugget in that
## optimization, or do to a mleGP style profile optimization
## for the the nugget instead

mleGPsep <- function(gpsepi, param=c("d", "g"), 
                  tmin=sqrt(.Machine$double.eps), 
                  tmax=-1, ab=c(0,0), maxit=100, verb=0)
  {
    param <- match.arg(param)

    if(param == "d") { ## LENGTHSCALE with L-BFGS-B given nugget

      theta <- getdGPsep(gpsepi)
      if(length(ab) != 2 || any(ab < 0)) stop("ab should be a positive 2-vector")   
 
      ## objective
      f <- function(theta, gpsepi, dab) 
        {
          newparamsGPsep(gpsepi, d=theta)
          -llikGPsep(gpsepi, dab=dab)
        }
      ## gradient of objective
      g <- function(theta, gpsepi, dab)
        {
          newparamsGPsep(gpsepi, d=theta)
          -dllikGPsep(gpsepi, param="d", ab=dab)
        }

      ## for compatibility with mleGP
      tmax[tmax < 0] <- Inf

      ## call R's optim function
      out <- optim(theta, fn=f, gr=g, method="L-BFGS-B",
        control=list(trace=verb, maxit=maxit), lower=tmin, upper=tmax, 
        gpsepi=gpsepi, dab=ab)

      ## sanity check completion of scheme
      if(sqrt(mean((out$par - getdGPsep(gpsepi))^2)) > sqrt(.Machine$double.eps))
        warning("stored d not same as theta-hat")
    }
    else { ## NUGGET conditionally on lengthscale

      ## sanity check
      if(length(ab) != 2 || any(ab < 0)) stop("ab should be a positive 2-vector");

      r <- .C("mleGPsep_nug_R",
            gpsepi = as.integer(gpsepi),
            verb = as.integer(verb),
            tmin = as.double(tmin),
            tmax = as.double(tmax),
            ab = as.double(ab),
            g = double(1),
            its = integer(1),
            package = "laGP")
    }

    ## build object for returning
    if(param == "d") return(list(d=out$par, its=max(out$counts), conv=out$convergence))
    else return(list(g=r$g, its=r$its))
  }


## jmleGPsep:
##
## joint MLE for lengthscale (d) and nugget (g) parameters;
## updates the internal GP parameterization (since mleGP does);
## R-only version

jmleGPsep <- function(gpsepi, N=100, drange=c(sqrt(.Machine$double.eps), 10), 
  grange=c(sqrt(.Machine$double.eps), 1), dab=c(0,0), gab=c(0,0), maxit=100, verb=0)
  {
    ## sanity check N
    if(length(N) != 1 && N > 0) 
      stop("N should be a positive scalar integer")
    dmle <- matrix(NA, nrow=N, ncol=getmGPsep(gpsepi))
    gmle <- dits <- dconv <- gits <- rep(NA, N)

    ## sanity check tmin and tmax
    if(length(drange) != 2) stop("drange should be a 2-vector for c(min,max)")
    if(length(grange) != 2) stop("grange should be a 2-vector for c(min,max)")

    ## loop over outer interations
    for(i in 1:N) {
      d <- mleGPsep(gpsepi, param="d", tmin=drange[1], tmax=drange[2],
                    ab=dab, maxit=maxit, verb=verb)
      dmle[i,] <- d$d; dits[i] <- d$its; dconv[i] <- d$conv
      g <- mleGPsep(gpsepi, param="g", tmin=grange[1], tmax=grange[2],
                    ab=gab, verb=verb)
      gmle[i] <- g$g; gits[i] <- g$its
      if((gits[i] <= 1 && (dits[i] <= 3 && dconv[i] == 0)) || dconv[i] > 1) break;
    }

    ## check if not converged
    if(i == N) warning("max outer its (N=", N, ") reached", sep="")
    else {
      dmle <- dmle[1:i,]; dits <- dits[1:i]; dconv <- dconv[1:i]
      gmle <- gmle[1:i]; gits <- gits[1:i]
    }

    ## total iteration count
    totits <- sum(c(dits, gits), na.rm=TRUE)

    ## assemble return objects
    return(list(mle=data.frame(d=dmle[i,,drop=FALSE], g=gmle[i], tot.its=totits, conv=dconv[i]), 
      prog=data.frame(dmle=dmle, dits=dits, dconv=dconv, gmle=gmle, gits=gits)))
  }


