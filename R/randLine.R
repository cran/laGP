#*******************************************************************************
#
# Local Approximate Gaussian Process Regression
# Copyright (C) 2013, Virginia Tech
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
# Questions? Contact Robert B. Gramacy (rbg@vt.edu) and Furong Sun (furongs@vt.edu)
#
#*******************************************************************************


## randLine:
##
## generate two-dimensional random paths (one-dimensional manifolds in 2d)
## comprising of different randomly chosen line types: linear, quadratic,
## cubic, exponential, and natural logarithm. If the input dimensionality 
## is higher than 2, then a line in two randomly chosen input coordinates 
## is generated 

randLine <- function(a, D, N, smin, res){
  
  # sanity check
  if(length(a) == 1) stop("'a' should be a two-element vector :-)")
  if(D < 2) stop("D must be >= 2 :-)")
  if((smin <= 0) || (res <= 0)) stop("smin and resolution must be positive :-)")
  
  # randomly choose a pair of input coordinates
  d <- sample(1:D, 2, replace=FALSE) 
  
  # to make sure that the starting line passes the origin
  x1 <- seq(-1, 1, length.out=res)
  
  # define each line prototype based on the starting line 
  x2.lin <- x1; x2.qua <- x1^2
  x2.cub <- x1^3; x2.exp <- exp(x1) - 1
  x1.ln <- seq(1e-9, 2, length.out=res); x2.ln <- log(x1.ln) ## natural log: [0, 2]
  
  # the probability of each line type is equal
  n.sub <- rmultinom(n=1, size=N, prob=rep(1/5, 5))
  n.l <- n.sub[1,]; n.q <- n.sub[2,]; n.c <- n.sub[3,] 
  n.e <- n.sub[4,]; n.ln <- n.sub[5,]
  
  # shifting and scaling arguments
  if(a[1] >= 0){
     d1.shift <- runif(N, -a[1], a[2])
     d1.scale <- runif(N, -a[1]/4, a[2]/4)
    
     d2.shift <- runif(N, -a[1], a[2])
     d2.scale <- runif(N, -a[1]/4, a[2]/4)
  }else if(a[2] <= 0){
     d1.shift <- runif(N, a[1], -a[2])
     d1.scale <- runif(N, a[1]/4, -a[2]/4)
    
     d2.shift <- runif(N, a[1], -a[2])
     d2.scale <- runif(N, a[1]/4, -a[2]/4)
  }else{
     d1.shift <- runif(N, a[1], a[2])
     d1.scale <- runif(N, a[1]/4, a[2]/4)
    
     d2.shift <- runif(N, a[1], a[2])
     d2.scale <- runif(N, a[1]/4, a[2]/4)
  }
  
  # define the minimum scaling constant
  for(i in 1:length(d1.scale)){
    while(abs(d1.scale[i]) < smin){
      if(d1.scale[i] > 0) d1.scale[i] <- smin
      if(d1.scale[i] < 0) d1.scale[i] <- -smin
      if(d1.scale[i] == 0) d1.scale[i] <- sample(c(smin, -smin), 1)
    }
  }
  
  for(i in 1:length(d2.scale)){
    while(abs(d2.scale[i]) < smin){
      if(d2.scale[i] > 0) d2.scale[i] <- smin
      if(d2.scale[i] < 0) d2.scale[i] <- -smin
      if(d2.scale[i] == 0) d2.scale[i] <- sample(c(smin, -smin), 1)
    }
  }
  
  # linear
  xx.lin <- list()
  
  n1.ld <- sample(1:length(d1.shift), n.l, replace=FALSE)
  d1.shift.lin <- d1.shift[n1.ld]
  n1.ls <- sample(1:length(d1.scale), n.l, replace=FALSE)
  d1.scale.lin <- d1.scale[n1.ls]
  
  n2.ld <- sample(1:length(d2.shift), n.l, replace=FALSE)
  d2.shift.lin <- d2.shift[n2.ld]
  n2.ls <- sample(1:length(d2.scale), n.l, replace=FALSE)
  d2.scale.lin <- d2.scale[n2.ls]
  for(i in 1:n.l){
      c1 <- d1.shift.lin[i]; s1 <- d1.scale.lin[i]
      c2 <- d2.shift.lin[i]; s2 <- d2.scale.lin[i]
    
      x1.lin <- s1 * x1 + c1
      x22.lin <- s2 * x2.lin + c2
      xx.lin[[i]] <- as.data.frame(cbind(x1.lin, x22.lin))
      names(xx.lin[[i]]) <- paste0("x", 1:2)
  }
  
  # quadratic
  xx.qua <- list()
  
  s.remain <- setdiff(1:length(d1.shift), n1.ld)
  d1.shift <- d1.shift[s.remain] # updated d1.shift
  n1.qd <- sample(1:length(d1.shift), n.q, replace=FALSE)
  d1.shift.qua <-  d1.shift[n1.qd]
  
  c.remain <- setdiff(1:length(d1.scale), n1.ls)
  d1.scale <- d1.scale[c.remain] # updated d1.scale
  n1.qs <- sample(1:length(d1.scale), n.q, replace=FALSE)
  d1.scale.qua <- d1.scale[n1.qs]
  
  s.remain <- setdiff(1:length(d2.shift), n2.ld)
  d2.shift <- d2.shift[s.remain] # updated d2.shift
  n2.qd <- sample(1:length(d2.shift), n.q, replace=FALSE)
  d2.shift.qua <-  d2.shift[n2.qd]
  
  c.remain <- setdiff(1:length(d2.scale), n2.ls)
  d2.scale <- d2.scale[c.remain] # updated d2.scale
  n2.qs <- sample(1:length(d2.scale), n.q, replace=FALSE)
  d2.scale.qua <- d2.scale[n2.qs]
  for(i in 1:n.q){
      c1 <- d1.shift.qua[i]; s1 <- d1.scale.qua[i]
      c2 <- d2.shift.qua[i]; s2 <- d2.scale.qua[i]
    
      x1.qua <- s1 * x1 + c1
      x22.qua <- s2 * x2.qua + c2
      xx.qua[[i]] <- as.data.frame(cbind(x1.qua, x22.qua))
      names(xx.qua[[i]]) <- paste0("x", 1:2)
  }
  
  # cubic
  xx.cub <- list()
  
  s.remain <- setdiff(1:length(d1.shift), n1.qd)
  d1.shift <- d1.shift[s.remain]
  n1.cd <- sample(1:length(d1.shift), n.c, replace=FALSE)
  d1.shift.cub <- d1.shift[n1.cd]
  
  c.remain <- setdiff(1:length(d1.scale), n1.qs)
  d1.scale <- d1.scale[c.remain]
  n1.cs <- sample(1:length(d1.scale), n.c, replace=FALSE)
  d1.scale.cub <- d1.scale[n1.cs]
  
  s.remain <- setdiff(1:length(d2.shift), n2.qd)
  d2.shift <- d2.shift[s.remain]
  n2.cd <- sample(1:length(d2.shift), n.c, replace=FALSE)
  d2.shift.cub <- d2.shift[n2.cd]
  
  c.remain <- setdiff(1:length(d2.scale), n2.qs)
  d2.scale <- d2.scale[c.remain]
  n2.cs <- sample(1:length(d2.scale), n.c, replace=FALSE)
  d2.scale.cub <- d2.scale[n2.cs]
  for(i in 1:n.c){
      c1 <- d1.shift.cub[i]; s1 <- d1.scale.cub[i]
      c2 <- d2.shift.cub[i]; s2 <- d2.scale.cub[i]
    
      x1.cub <- s1 * x1 + c1
      x22.cub <- s2 * x2.cub + c2
      xx.cub[[i]] <- as.data.frame(cbind(x1.cub, x22.cub))
      names(xx.cub[[i]]) <- paste0("x", 1:2)
  }
  
  # exponential
  xx.exp <- list()
  
  s.remain <- setdiff(1:length(d1.shift), n1.cd)
  d1.shift <- d1.shift[s.remain]
  n1.ed <- sample(1:length(d1.shift), n.e, replace=FALSE)
  d1.shift.exp <-  d1.shift[n1.ed]
  
  c.remain <- setdiff(1:length(d1.scale), n1.cs)
  d1.scale <- d1.scale[c.remain]
  n1.es <- sample(1:length(d1.scale), n.e, replace=FALSE)
  d1.scale.exp <- d1.scale[n1.es]
  
  s.remain <- setdiff(1:length(d2.shift), n2.cd)
  d2.shift <- d2.shift[s.remain]
  n2.ed <- sample(1:length(d2.shift), n.e, replace=FALSE)
  d2.shift.exp <-  d2.shift[n2.ed]
  
  c.remain <- setdiff(1:length(d2.scale), n2.cs)
  d2.scale <- d2.scale[c.remain]
  n2.es <- sample(1:length(d2.scale), n.e, replace=FALSE)
  d2.scale.exp <- d2.scale[n2.es]
  for(i in 1:n.e){
      c1 <- d1.shift.exp[i]; s1 <- d1.scale.exp[i]
      c2 <- d2.shift.exp[i]; s2 <- d2.scale.exp[i]
    
      x1.exp <- s1 * x1 + c1
      x22.exp <- s2 * x2.exp + c2
      xx.exp[[i]] <- as.data.frame(cbind(x1.exp, x22.exp))
      names(xx.exp[[i]]) <- paste0("x", 1:2)
  }
  
  # natural logarithm
  xx.ln <- list()
  x1.ln <- x1.ln - 1
  
  s.remain <- setdiff(1:length(d1.shift), n1.ed)
  d1.shift <- d1.shift[s.remain]
  d1.shift.ln <-  d1.shift
  
  c.remain <- setdiff(1:length(d1.scale), n1.es)
  d1.scale <- d1.scale[c.remain]
  d1.scale.ln <-  d1.scale
  
  s.remain <- setdiff(1:length(d2.shift), n2.ed)
  d2.shift <- d2.shift[s.remain]
  d2.shift.ln <- d2.shift
  
  c.remain <- setdiff(1:length(d2.scale), n2.es)
  d2.scale <- d2.scale[c.remain]
  d2.scale.ln <-  d2.scale
  for(i in 1:n.ln){
      c1 <- d1.shift.ln[i]; s1 <- d1.scale.ln[i]
      c2 <- d2.shift.ln[i]; s2 <- d2.scale.ln[i]
      
      x1s.ln <- s1 * x1.ln + c1
      x22.ln <- s2 * x2.ln + c2
      xx.ln[[i]] <- as.data.frame(cbind(x1s.ln, x22.ln))
      names(xx.ln[[i]]) <- paste0("x", 1:2)
  }
  
  ## combine all of the line types and indices to a list
  randline <- list(lin=xx.lin, qua=xx.qua, cub=xx.cub, ep=xx.exp, ln=xx.ln, d=d) 
  return(randline)
}
