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


## blhs:
##
## provides a bootstrapped block Latin hypercube subsample, including
## the subsampled input space and the corresponding responses

blhs <- function(y, X, m)
  {
    D <- as.data.frame(cbind(y, X))  
    d <- ncol(D)          
    k <- d - 1    ## dimensionality of input space        
    names(D)[1:d] <- c("y", paste("x", 1:k, sep = "")) 
    xblock <- paste("block_x", 1:k, sep = "") 
    D[xblock] <- 1
    for (i in 1:m){
      for (j in 1:k){
           D[,d+j] <- ifelse(D[,j+1] > (i-1)/m & D[,j+1] < i/m, i, D[,d+j])
      }
    }
    global_block <- D[,d+1]
    for (u in 2:k){
         new <- (D[,d+u] -1) * m^(u-1)
         global_block <- cbind(global_block, new)    
    }
    D$Global_block <- rowSums(global_block[,1:k])
            
    Label <- matrix(NA, nrow = m, ncol = k)
    for (h in 1:k){
         Label[,h] <- sample(1:m, m, replace = FALSE)
    }
    su <- Label[,1]
    for (f in 2:k){
         nn <- (Label[,f] - 1) * m^(f-1) 
         su <- cbind(su, nn)
    }
    sub <- rowSums(su[,1:k])
            
    ysub <- D[D$Global_block %in% sub, 1]    
    Xsub <- D[D$Global_block %in% sub, 2:d]  
    return(list(xs = Xsub, ys = ysub))
}


## blhs.loop:
##
## the main purpose is to return the median of K times maximum likelihood estimated 
## lengthscales, and it also returns the median of K times bootstrapped input spaces 
## and responses, as well as the median of response lengths

blhs.loop <- function(y, X, m, K, da, g = 1e-3, maxit = 100, verb = 0, plot.it = FALSE)
  {
    ly_loop <- rep(NA, K)
    boot_theta_loop <- matrix(NA, nrow = K, ncol = ncol(X))
    X_loop <- list()  
    y_loop <- list()  
    for (i in 1:K){
         sub <- blhs(y = y, X = X, m = m)
         X_loop[[i]] <- as.matrix(sub$xs)
         y_loop[[i]] <- sub$ys
         ly_loop[i] <- length(sub$ys)
         gpsepi <- newGPsep(as.matrix(sub$xs), sub$ys, d = da$start, g = g, dK = TRUE)
         mle <- mleGPsep(gpsepi, tmin = da$min, tmax = 10*da$max, ab = da$ab, maxit = maxit)  
         deleteGPsep(gpsepi)
         boot_theta_loop[i,] <- mle$d

         if(verb > 0){
            cat("BLHS-sub ", i, " of ", K, ", nsub=", length(sub$y), ", 
              its=", mle$its, ", msg=", mle$msg, "\n", sep = "")
         }
         
    }
    if(plot.it) boxplot(boot_theta_loop, xlab = "input", ylab = "theta-hat", main = "BLHS")
    
    ly_median <- round(median(ly_loop))
    theta.hat_median <- apply(boot_theta_loop, 2, median) 
    
    if(K %% 2 != 0){
       X_median <- X_loop[[(K+1)/2]]
       y_median <- y_loop[[(K+1)/2]]
    }else{
       X_median <- X_loop[[K/2]]
       y_median <- y_loop[[K/2]]
    }
    
    return(list(that = theta.hat_median, ly = ly_median, xm = X_median, ym = y_median))  
}
