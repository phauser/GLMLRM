library(Rcpp)
library(RcppArmadillo)
library(RcppDist)
library(RcppParallel)
library(MASS)
nThreads=defaultNumThreads()

# Simulations to compare the GLLRM+C to the GLMLRM

Rcpp::sourceCpp("~/Simulations/GLMLRM_normal.cpp")
load("~/Simulations/X_normal.Rdata") # covariate matrix bootstrapped from 2022 VAERS data
G=100 # number of replications
r=2; q=2 # r and q values

# Case 1.2
for(g in 1:G){
  # specify dimensions of the data
  n=nrow(x11)
  d1=12
  d2=13
  d=d1+d2
  p=ncol(x11)
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
  nu = x11%*%B+eta%*%t(lambda)
  Y=matrix(nrow=n, ncol=d)
  for(a in 1:n){Y[a,] = mvrnorm(n=1, mu=nu[a,], Sigma=lambda%*%t(lambda)+diag(d))}
  
  mod11 = GLMLRM_normal1(x11, Y, r=r, q=q, nBurnin=1000, nCollect=5000)
  mod12 = GLMLRM_normal2(x11, Y, d1=d1, d2=d2, r1=r, r2=r, q=q, nBurnin=1000, nCollect=5000)

  save(mod11, file=paste0("~/mod11_", g, ".Rdata"))
  save(mod12, file=paste0("~/mod12_", g, ".Rdata"))
}

# Case 1.2
for(g in 1:G){
  # specify dimensions of the data
  n=nrow(x21)
  d1=25
  d2=25
  d=d1+d2
  p=ncol(x21)
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
  nu = x21%*%B+eta%*%t(lambda)
  Y=matrix(nrow=n, ncol=d)
  for(a in 1:n){Y[a,] = mvrnorm(n=1, mu=nu[a,], Sigma=lambda%*%t(lambda)+diag(d))}
  
  mod21 = GLMLRM_normal1(x21, Y, r=r, q=q, nBurnin=1000, nCollect=5000)
  mod22 = GLMLRM_normal2(x21, Y, d1=d1, d2=d2, r1=r, r2=r, q=q, nBurnin=1000, nCollect=5000)

  save(mod21, file=paste0("~/mod21_", g, ".Rdata"))
  save(mod22, file=paste0("~/mod22_", g, ".Rdata"))
}


# DIC
mod11$DIC

# example to calculate MSE & bias
Perform(B, mod11$B, C, mod11$C)


