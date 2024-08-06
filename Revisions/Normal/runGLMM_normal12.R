r = getOption("repos")
r["CRAN"] = "http://cran.us.r-project.org"
options(repos = r)
install.packages("RcppDist")

library(Rcpp)
library(RcppArmadillo)
library(RcppDist)
library(RcppParallel)
library(tidyr)
library(ggplot2)
library(dplyr)
library(MASS)
library(bayestestR)

nThreads=defaultNumThreads()
Rcpp::sourceCpp("~/Project3/Code/LLRM3.cpp")
load("/work/users/p/h/phauser/Sim/Normal/X.Rdata")

# number of replications
G=100

# matrix to hold results
res = matrix(NA, nrow=G, ncol=8)
colnames(res)= c("MSE(B)","bias(B)","CP(B)","MIW(B)","MSE(C)","bias(C)","CP(C)","MIW(C)")

# function to calculate CP, MIW, MSE, bias
stats=function(B0, B, PostB){
  CP = matrix(NA, nrow=nrow(B0), ncol=ncol(B0))
  MIW = matrix(NA, nrow=nrow(B0), ncol=ncol(B0))
  MSE = matrix(NA, nrow=nrow(B0), ncol=ncol(B0))
  bias = matrix(NA, nrow=nrow(B0), ncol=ncol(B0))
  
  for(i in 1:p){
    for(j in 1:d){
    out       = ci(B[i,j,], method="HDI")
    MIW[i,j]  = out$CI_high-out$CI_low
    CP[i,j]   = ifelse(between(B0[i,j], out$CI_low, out$CI_high), 1, 0)
    bias[i,j] = B0[i,j]-PostB[i,j]
    MSE[i,j]  = bias[i,j]^2
    }
  }
  out = list(MSE=mean(MSE), bias=mean(bias), CP=mean(CP), MIW=mean(MIW))
  return(out)
}


for(g in 1:G){
  n=nrow(x11)
  p=ncol(x11)
  d1=12
  d2=13
  d=d1+d2
  r0=2
  q0=2
  
  # B
  delta1 = diag(sort(abs(rnorm(r0, mean=0, sd=sqrt(d1))),decreasing=TRUE))
  U1 = matrix(rnorm(p*r0, mean=0, sd=1/sqrt(p)), nrow=p)
  V1 = matrix(rnorm(d1*r0, mean=0, sd=1/sqrt(d1)), nrow=d1)
  B1 = U1%*%delta1%*%t(V1)
  
  delta2 = diag(sort(abs(rnorm(r0, mean=0, sd=sqrt(d2))),decreasing=TRUE))
  U2 = matrix(rnorm(p*r0, mean=0, sd=1/sqrt(p)), nrow=p)
  V2 = matrix(rnorm(d2*r0, mean=0, sd=1/sqrt(d2)), nrow=d2)
  B2 = U2%*%delta2%*%t(V2)
  
  B0=cbind(B1,B2)
  C = matrix(rnorm(p*d, 0, 0.5)*rbinom(p*d, 1, 0.05), nrow=p)
  B = B0 + C

  # eta
  eta=matrix(rnorm(n*q0, 0, 1), nrow=n, ncol=q0)
  
  # lambda
  phi = matrix(rgamma(d*q0, 3/2, 2/3), nrow=d, ncol=q0)
  tau = cumprod(c(rgamma(1, 2, 1), rgamma(q0-1, 3, 1)))
  Dinv = phi*tau
  lambda=matrix(NA, nrow=d, ncol=q0)
  for(k in 1:d){
    for(h in 1:q0){
      lambda[k,h]=rbinom(1,1,0.8)*round(rnorm(1, mean=0, sd=0.2*sqrt(Dinv[k,h])),1)
      if(h>k){lambda[k,h]=0}
    }
  }
  diag(lambda)=1
  
  # Y
  nu = x11%*%B+eta%*%t(lambda)
  Y=matrix(nrow=n, ncol=d)
  for(a in 1:n){Y[a,] = mvrnorm(n=1, mu=nu[a,], Sigma=lambda%*%t(lambda)+diag(d))}

  mod = GLMLRM(x11, Y, "SS", d1=d1, d2=d2, r1=r0, r2=r0, q=q0, nBurnin=1000, nCollect=5000)
  res[g,] = c(unlist(stats(B0, mod$B0, mod$PostB0)), unlist(stats(C, mod$C, mod$PostC)))
}

# take average across replications
RES = as.data.frame(res) %>% summarise(
  `MSE(B)`  =round(mean(as.numeric(`MSE(B)`)),4),
  `bias(B)` =round(mean(as.numeric(`bias(B)`)),4),
  `CP(B)`   =round(mean(as.numeric(`CP(B)`)),4),
  `MIW(B)`  =round(mean(as.numeric(`MIW(B)`)),4),
  `MSE(C)`  =round(mean(as.numeric(`MSE(C)`)),4),
  `bias(C)` =round(mean(as.numeric(`bias(C)`)),4),
  `CP(C)`   =round(mean(as.numeric(`CP(C)`)),4),
  `MIW(C)`  =round(mean(as.numeric(`MIW(C)`)),4))


save(res, RES, file="/nas/longleaf/home/phauser/Project3/Sim/Revisions/Normal/normal_12.Rdata")
