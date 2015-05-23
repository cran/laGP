## optim.auglag:
##
## Optimization of known objective under unknown constraints via
## the Augmented Lagriangian framework with GP surrogate modeling
## of the constraint functions

optim.auglag <- function(fn, B, fhat=FALSE, start=10, end=100, Xstart=NULL, 
  sep=FALSE, ab=c(3/2,4), lambda=rep(1,ncol(B)), rho=1/2, urate=10, 
  ncandf=function(t) { t },  dg.start=c(0.1,1e-6), dlim=sqrt(ncol(B))*c(1/100,10), 
  Bscale=1, tol=list(ei=1e-5, ey=0.05, its=10), nomax=FALSE, N=1000, plotprog=FALSE, 
  verb=2, ...)
{
  ## check start
  if(start >= end) stop("must have start < end")

  ## check sep and determine whether to use GP or GPsep commands
  if(sep) { newM <- newGPsep; mleM <- mleGPsep; updateM <- updateGPsep; 
    alM <- alGPsep; deleteM <- deleteGPsep; nd <- nrow(B) }
  else { newM <- newGP; mleM <- mleGP; updateM <- updateGP; 
    alM <- alGP; deleteM <- deleteGP; formals(newM)$dK <- TRUE; nd <- 1 }

  ## get initial design
  X <- dopt.gp(start, Xcand=lhs(10*start, B))$XX
  X <- rbind(Xstart, X)
  start <- nrow(X)

  ## first run to determine dimensionality of the constraint
  out <- fn(X[1,]*Bscale, ...)
  nc <- length(out$c)

  ## allocate progress objects, and initialize
  prog <- obj <- rep(NA, start)
  C <- matrix(NA, nrow=start, ncol=nc)
  obj[1] <- out$obj; C[1,] <- out$c
  if(all(out$c <= 0)) prog[1] <- out$obj
  else prog[1] <- Inf

  ## remainder of starting run
  for(t in 2:start) {
    out <- fn(X[t,]*Bscale, ...)
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
  Cgpi <- rep(NA, nc)
  d <- matrix(NA, nrow=nc, ncol=nd)
  Cnorm <- apply(abs(C), 2, max)
  for(j in 1:nc) {
    Cgpi[j] <- newM(X, C[,j]/Cnorm[j], dg.start[1], dg.start[2])
    d[j,] <- mleM(Cgpi[j], param="d", tmin=dlim[1], tmax=dlim[2], ab=ab, verb=verb-1)$d
  }
  ds <- matrix(rowMeans(d), nrow=1)

  ## possibly initialize objective surrogate
  if(fhat) {
    fnorm <- max(abs(obj))
    fgpi <- newM(X, obj/fnorm, dg.start[1], dg.start[2])
    df <- mleM(fgpi, param="d", tmin=dlim[1], tmax=dlim[2], ab=ab, verb=verb-1)$d
    dfs <- matrix(df, nrow=1)
  } else { fgpi <- -1; fnorm <- Bscale }

  ## init for loop
  mei <- Inf
  new.params <- FALSE

  ## iterating over the black box evaluations
  for(t in (start+1):end) {

    ## update if inner problem has converged
    if(mei/fnorm < tol$ei || since >= tol$its) { 

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

    ## rebuild surrogates periodically under new normalized responses
    if(new.params || (t > (start+1) && (t %% urate == 0))) {
    
      ## constraint surrogates 
      Cnorm <- apply(abs(C), 2, max)
      for(j in 1:nc) {
        deleteM(Cgpi[j])
        Cgpi[j] <- newM(X, C[,j]/Cnorm[j], d[j,], dg.start[2])
        d[j,] <- mleM(Cgpi[j], param="d", tmin=dlim[1], tmax=dlim[2], ab=ab, verb=verb-1)$d
      }
      ds <- rbind(ds, rowMeans(d))

      ## possible objective surrogate
      if(fhat) {
        deleteM(fgpi)
        fnorm <- max(abs(obj))
        fgpi <- newM(X, obj/fnorm, df, dg.start[2])
        df <- mleM(fgpi, param="d", tmin=dlim[1], tmax=dlim[2], ab=ab, verb=verb-1)$d
        dfs <- rbind(dfs, df)
      } else { df <- NULL }

      new.params <- FALSE
    }
  
    ## random candidate grid
    ncand <- ncandf(t)
    if(fhat) XX <- lhs(ncand, B)
    else XX <- rbetter(ncand, B, sum(X[which(obj == m2),]))
    ## NOTE: might be a version of rbetter for fhat

    ## calculate composite surrogate, calling it eis for now
    eyei <- alM(XX, fgpi, fnorm, Cgpi, Cnorm, lambda, rep(1/(2*rho), nc), ybest, 
      nomax=as.logical(nomax), N=N)
    eis <- eyei$ei; by <- "ei"
    mei <- max(eis)
    nzei <- sum(eis > 0)
    if(nzei <= tol$ey*ncand) { eis <- -(eyei$ey); by <- "ey"; mei <- Inf }

    ## continue plotting 
    if(!is.logical(plotprog) || plotprog) {
      par(mfrow=c(1,3+fhat))
      plot(prog, type="l", main="progress")
      if(is.logical(plotprog)) {
        if(length(eis) < 30) { span <- 0.5 } else { span <- 0.1 }
        g <- interp.loess(XX[,1], XX[,2], eis, span=span)
      } else g <- plotprog(XX[,1], XX[,2], eis)
      image(g, xlim=range(X[,1]), ylim=range(X[,2]), main="EI")
      points(X[,1:2], col=valid+3, pch=19)
      matplot(ds, type="l", lty=1, main="constraint lengthscale")
      if(fhat) matplot(dfs, type="l", lty=1, main="objective lengthscale")
    }

    ## calculate next point
    m <- which.max(eis)
    xstar <- matrix(XX[m,], ncol=ncol(X))
    if(verb > 0) {
      cat("t=", t, " ", sep="")
      cat(by, "=", eis[m]/Bscale, " (", nzei,  "/", ncandf(t), ")", sep="")
      cat("; xbest=[", paste(signif(X[obj == m2,],3), collapse=" "), sep="")
      cat("]; ybest (v=", m2, ", al=", ybest, ", since=", since, ")\n", sep="")
    }

    ## new run
    out <- fn(xstar*Bscale, ...)
    ystar <- out$obj; obj <- c(obj, ystar); C <- rbind(C, out$c)

    ## update GP fits
    X <- rbind(X, xstar)
    for(j in 1:nc) updateM(Cgpi[j], xstar, out$c[j]/Cnorm[j], verb=verb-2)
    
    ## check if best valid has changed
    since <- since + 1
    valid <- apply(C, 1, function(x) { all(x <= 0) })
    if(min(obj[valid]) != m2) { m2 <- min(obj[valid]); since <- 0 }

    ## check if best auglag has changed
    alstar <- out$obj + lambda %*% drop(out$c) + rep(1/(2*rho),nc) %*% pmax(0, drop(out$c))^2
    ## alstar <- out$obj + lambda %*% out$c + rep(1/(2*rho),nc) %*% out$c^2
    if(alstar < ybest) { ybest <- alstar; since <- 0 }

    ## augment progress
    prog <- c(prog, m2)
  }

  ## delete GP surrogates
  for(j in 1:nc) deleteM(Cgpi[j])

  ## return output objects
  return(list(prog=prog, obj=obj, X=X, C=C, d=d, df=df, lambda=lambda, rho=rho))
}
