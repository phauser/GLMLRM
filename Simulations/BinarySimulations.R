library(Rcpp)
library(RcppArmadillo)
library(RcppDist)
library(RcppParallel)
library(MASS)

# Simulations to compare the GLLRM+C to the GLMLRM
nThreads=defaultNumThreads()

Rcpp::sourceCpp("~/Simulations/GLMLRM.cpp")
load("~/Simulations/X_binary.Rdata") # covariate matrix bootstrapped from 2022 VAERS data
G=100 # number of replications
r=2; q=2 # r and q values

# function to calculate sensitivity and specificity
eval=function(B0, B1, d_star){
  TP = length(intersect(which(B1==1), which((B0)>d_star)))
  TN = length(intersect(which(B1==0), which((B0)<d_star)))
  FP = length(intersect(which(B1==1), which((B0)<d_star)))
  FN = length(intersect(which(B1==0), which((B0)>d_star)))
  sens = TP/(TP+FN)
  spec = TN/(TN+FP)
  out = c(TP, TN, FP, FN, sens, spec)
  names(out)=c("TP", "TN", "FP", "FN", "sensitivity", "specificity")
  return(out)
}

# Case 1.2
for(g in 1:G){
  # specify dimensions of the data
  n=nrow(x1)
  d1=12
  d2=13
  d=d1+d2
  p=ncol(x1)
  r0=2
  q0=2
  
  # generate B0
  delta1 = diag(sort(abs(rnorm(r0, mean=0, sd=sqrt(d1))),decreasing=TRUE))
  U1 = matrix(rnorm(p*r0, mean=0, sd=1/sqrt(p)), nrow=p)
  V1 = matrix(rnorm(d1*r0, mean=0, sd=1/sqrt(d1)), nrow=d1)
  B1 = U1%*%delta1%*%t(V1)

  delta2 = diag(sort(abs(rnorm(r0, mean=0, sd=sqrt(d2))),decreasing=TRUE))
  U2 = matrix(rnorm(p*r0, mean=0, sd=1/sqrt(p)), nrow=p)
  V2 = matrix(rnorm(d2*r0, mean=0, sd=1/sqrt(d2)), nrow=d2)
  B2 = U2%*%delta2%*%t(V2)
  
  B0=cbind(B1,B2)
  
  # generate C
  Q=0.05
  C = matrix(rnorm(p*d, 0, 0.5)*rbinom(p*d, 1, Q), nrow=p)
  B = B0 + C

  # generate eta
  eta=matrix(rnorm(n*q0, 0, 1), nrow=n, ncol=q0)
  
  # generate lambda
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
  lambda

  # generate Y
  nu = x1%*%B+eta%*%t(lambda)
  prob = 1/(1+exp(-nu))
  Y=matrix(nrow=n, ncol=d)
  for(j in 1:d){Y[,j] = rbinom(n, 1, prob[,j])}

  # one decomposition - "GLLRM+C"
  mod11 = GLMLRM1(x1, Y, r=r, q=q, 
                 epsE=5, epsU=.4, epsV=.7,  epsD=.05, epsL=.5, epsC=.4,
                 nBurnin=1, nCollect=5, thin=1, numThreads=nThreads)
  
  save(mod1, file=paste0("~/mod11_", g, ".Rdata"))

  # two decompositions - "GLMLRM"
  mod12 = GLMLRM2(x1, Y, d1=d1, d2=d2, r1=r, r2=r, q=q,
                epsE=8, epsU1=1, epsU2=1, epsV1=.5, epsV2=.5,
                epsD1=.08, epsD2=.08, epsL=.2, epsC=.05,
                nBurnin=1000, nCollect=5000, thin=1, numThreads=nThreads)
  save(mod2, file=paste0("~/mod12_", g, ".Rdata"))
}

# Case 2.2
for(g in 1:G){
  # specify dimensions of the data
  n=nrow(x1)
  d1=25
  d2=25
  d=d1+d2
  p=ncol(x2)
  r0=3
  q0=4
  
  # generate B0
  delta1 = diag(sort(abs(rnorm(r0, mean=0, sd=sqrt(d1))),decreasing=TRUE))
  U1 = matrix(rnorm(p*r0, mean=0, sd=1/sqrt(p)), nrow=p)
  V1 = matrix(rnorm(d1*r0, mean=0, sd=1/sqrt(d1)), nrow=d1)
  B1 = U1%*%delta1%*%t(V1)
  
  delta2 = diag(sort(abs(rnorm(r0, mean=0, sd=sqrt(d2))),decreasing=TRUE))
  U2 = matrix(rnorm(p*r0, mean=0, sd=1/sqrt(p)), nrow=p)
  V2 = matrix(rnorm(d2*r0, mean=0, sd=1/sqrt(d2)), nrow=d2)
  B2 = U2%*%delta2%*%t(V2)
  
  B0=cbind(B1,B2)
  
  # generate C
  Q = 0.05
  C = matrix(rnorm(p*d, 0, 0.5)*rbinom(p*d, 1, Q), nrow=p)
  B = B0 + C
  
  # generate eta
  eta=matrix(rnorm(n*q0, 0, 1), nrow=n, ncol=q0)
  
  # generate lambda
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
  lambda
  
  # generate Y
  nu = x2%*%B+eta%*%t(lambda)
  prob = 1/(1+exp(-nu))
  Y=matrix(nrow=n, ncol=d)
  for(j in 1:d){Y[,j] = rbinom(n, 1, prob[,j])}
  
  # one decomposition - "GLLRM+C"
  mod21 = GLMLRM1(x2, Y, r=r, q=q, 
                 epsE=5, epsU=.4, epsV=.7,  epsD=.05, epsL=.5, epsC=.4,
                 nBurnin=1000, nCollect=5000, thin=1, numThreads=nThreads)
  
  save(mod21, file=paste0("~/mod21_", g, ".Rdata"))
  # two decompositions - "GLMLRM"
  mod22 = GLMLRM2(x2, Y, d1=d1, d2=d2, r1=r, r2=r, q=q, 
                 epsE=8, epsU1=1, epsU2=1, epsV1=.5, epsV2=.5, 
                 epsD1=.08, epsD2=.08, epsL=.2, epsC=.05,
                 nBurnin=1000, nCollect=5000, thin=1, numThreads=nThreads)
  save(mod22, file=paste0("~/mod22_", g, ".Rdata"))
}

# example to calculate DIC
DIC(Y, x1, mod11$PostB, mod11$Postlambda, mod11$PostEta, mod11$dev)

# example to calculate MSE & bias
Perform(B, mod11$B, C, mod11$C)

# example to calculate sensitivity & specificity
d_star=1; p_star=0.5
eval(B, Signal(mod11$B, d_star, p_star)$Final, d_star)

