## optim.auglag:
##
## Optimization of known objective under unknown constraints via
## the Augmented Lagriangian framework with GP surrogate modeling
## of the constraint functions

optim.auglag <- function(fn, B, start=10, end=100, Xstart=NULL, ab=c(3/2,4), 
  lambda=rep(1,ncol(B)), rho=1/2, urate=10, ncandf=function(t) { t }, 
  dg.start=c(0.1,1e-6), dlim=sqrt(ncol(B))*c(1/100,10), obj.norm=1,
  tol=list(ei=1e-5, ey=0.05, its=10), nomax=FALSE, N=1000, plotprog=TRUE, 
  verb=2, ...)
{
  ## get initial design
  X <- dopt.gp(start, Xcand=lhs(10*start, B))$XX
  X <- rbind(Xstart, X)
  start <- nrow(X)

  ## first run to determine dimensionlity of the constraint
  out <- fn(X[1,]*obj.norm, ...)
  nc <- length(out$c)

  ## allocate progress objects, and initialize
  prog <- obj <- rep(NA, start)
  d <- C <- matrix(NA, nrow=start, ncol=nc)
  obj[1] <- out$obj; C[1,] <- out$c
  if(all(out$c <= 0)) prog[1] <- out$obj
  else prog[1] <- Inf

  ## remainder of starting run
  for(t in 2:start) {
    out <- fn(X[t,]*obj.norm, ...)
    obj[t] <- out$obj; C[t,] <- out$c
    ## update best so far
    if(all(out$c <= 0) && out$obj < min(prog, na.rm=TRUE)) prog[t] <- out$obj
    else prog[t] <- prog[t-1]
  }

  ## best auglag seen so far
  valid <- C <= 0
  Cm <- C; Cm[valid] <- 0
  al <- obj + C %*% lambda + Cm^2 %*% rep(1/(2*rho), nc)
  ybest <- min(al)
  since <- 0
  ## best valid so far
  valid <- apply(valid, 1, prod)
  m2 <- min(obj[valid])

  ## initializing constraint surrogates
  d <- Cgpi <- rep(NA, nc)
  Cnorm <- apply(abs(C), 2, max)
  for(j in 1:nc) {
    Cgpi[j] <- newGP(X, C[,j]/Cnorm[j], dg.start[1], dg.start[2], dK=TRUE)
    d[j] <- mleGP(Cgpi[j], param="d", tmin=dlim[1], tmax=dlim[2], ab=ab, verb=verb-1)$d
  }
  ds <- matrix(d, nrow=1)

  ## init for loop
  mei <- Inf
  new.params <- FALSE

  ## iterating over the black box evaluations
  for(t in (start+1):end) {

    ## update if inner problem has converged
    if(mei/obj.norm < tol$ei || since >= tol$its) { 

      ## lambda and rho update
      valid <- C <= 0
      Cm <- C; Cm[valid] <- 0
      al <- obj + C %*% lambda + Cm^2 %*% rep(1/(2*rho), nc)
      ck <- C[which.min(al),]
      lambda.new <- pmax(0, lambda + (1/rho) * ck)
      if(nomax > 1 || any(ck > 0)) rho.new <- rho/2
      else rho.new <- rho

      ## printing progress  
      if(lambda.new != lambda || rho.new != rho) {
        if(verb > 0) {
          cat("updating La: rho=", rho.new, sep="")
          cat("; lambda=(", paste(signif(lambda.new,3), collapse=", "), ")\n", sep="")
        }
        new.params <- TRUE
      } else new.params <- FALSE

      ## confirm update of augmented lagrangian
      lambda <- lambda.new; rho <- rho.new
      ybest <- min(obj + C %*% lambda + Cm^2 %*% rep(1/(2*rho), nc))
      since <- 0
      valid <- apply(valid, 1, prod)
    }

    ## update constraint surrogates
    if(new.params || (t > (start+1) && (t %% urate == 0))) {
    
      Cnorm <- apply(abs(C), 2, max)
      for(j in 1:nc) {
        deleteGP(Cgpi[j])
        Cgpi[j] <- newGP(X, C[,j]/Cnorm[j], d[j], dg.start[2], dK=TRUE)
        d[j] <- mleGP(Cgpi[j], param="d", tmin=dlim[1], tmax=dlim[2], ab=ab, verb=verb-1)$d
      }
      ds <- rbind(ds, d)

      new.params <- FALSE
    }
  
    ## random candidate grid
    ncand <- ncandf(t)
    XX <- rbetter(ncand, B, sum(X[which(obj == m2),]))
    ## XX <- lhs(ncandf(t), B)
  
    ## calculate composite surrogate, calling it eis for now
    eyei <- alGP(Cgpi, XX, obj.norm, Cnorm, lambda, rep(1/(2*rho), nc), ybest, 
      nomax=as.logical(nomax), N=N)
    eis <- eyei$ei; by <- "ei"
    mei <- max(eis)
    nzei <- sum(eis > 0)
    if(nzei <= tol$ey*ncand) { eis <- -(eyei$ey); by <- "ey"; mei <- Inf }

    ## continue plotting 
    if(plotprog) {
      par(mfrow=c(1,3))
      plot(prog, type="l")
      image(interp(XX[,1], XX[,2], eis), xlim=range(X[,1]), ylim=range(X[,2]))
      points(X[,1:2], col=valid+3, pch=19)
      matplot(ds, type="l", lty=1)
    }

    ## calculate next point
    m <- which.max(eis)
    xstar <- matrix(XX[m,], ncol=ncol(X))
    if(verb > 0) {
      cat("t=", t, " ", sep="")
      cat(by, "=", eis[m]/obj.norm, " (", nzei,  "/", ncandf(t), ")", sep="")
      cat("; xbest=[", paste(signif(X[obj == m2,],3), collapse=" "), sep="")
      cat("]; ybest (v=", m2, ", al=", ybest, ", since=", since, ")\n", sep="")
    }

    ## new run
    out <- fn(obj.norm*xstar, ...)
    ystar <- out$obj; obj <- c(obj, ystar); C <- rbind(C, out$c)

    ## update GP fits
    X <- rbind(X, xstar)
    for(j in 1:nc) updateGP(Cgpi[j], xstar, out$c[j]/Cnorm[j], verb=verb-2)
    
    ## check if best valid has changed
    since <- since + 1
    valid <- apply(C, 1, function(x) { all(x <= 0) })
    if(min(obj[valid]) != m2) { m2 <- min(obj[valid]); since <- 0 }

    ## check if best auglag has changed
    alstar <- out$obj + lambda %*% out$c + rep(1/(2*rho),nc) %*% pmax(0, out$c)^2
    ## alstar <- out$obj + lambda %*% out$c + rep(1/(2*rho),nc) %*% out$c^2
    if(alstar < ybest) { ybest <- alstar; since <- 0 }

    ## augment progress
    prog <- c(prog, m2)
  }

  ## delete GP surrogates
  for(j in 1:nc) deleteGP(Cgpi[j])

  ## return output objects
  return(list(prog=prog, obj=obj, C=C, d=d))
}
