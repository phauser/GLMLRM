library(Rcpp)
library(RcppArmadillo)
library(RcppDist)
library(RcppParallel)
nThreads=defaultNumThreads()
Rcpp::sourceCpp("~/GLMLRM.cpp")

# columns of X have the vaccine names
# columns of Y have the AE names
load("~/VAERS.Rdata")

# Input Parameters
# X: covariate matrix (n x p)
# Y: outcome matrix (n x d)
# r: number of layers in decomposition
# q: number of latent factors
# epsE, epsU, etc...tuning parameters - adjust the tuning parameters such that the
# sampling acceptance rates are between 20-40%, check these by looking at mod$AR_E, mod$AR_U, etc.

# GLMLRM with 1 outcome group
mod11 = GLMLRM1(X=x.small, Y=y.small, r=4, q=4, 
                epsE=6, epsU=.1, epsV=.1,  epsD=.005, epsL=.3, epsC=.5,
                nBurnin=1000, nCollect=5000, thin=1, numThreads=nThreads)
mod21 = GLMLRM1(X=x.large, Y=y.large, r=7, q=4, 
             epsE=6, epsU=.1, epsV=.1,  epsD=.005, epsL=.3, epsC=.5,
             nBurnin=1000, nCollect=5000, thin=1, numThreads=nThreads)

# GLMLRM with 2 outcome groups
# d1=number of outcomes in the first group (should correspond to the d1 leftmost columns)
# d2=number of outcomes in the second group (should correspond to the d2 rightmost columns)
mod12 = GLMLRM2(X=x.small, Y=y.small, d1=13, d2=12, r1=2, r2=2, q=4, 
                epsE=7, epsU1=.5, epsU2=.5, epsV1=.3, epsV2=.1, epsD1=.01, epsD2=.01, epsL=.1, epsC=.05,
                nBurnin=1000, nCollect=5000, thin=1, numThreads=nThreads)

mod22 = GLMLRM2(X=x.large, Y=y.large, d1=31, d2=19, r1=5, r2=5, q=4, 
                epsE=7, epsU1=.5, epsU2=.5, epsV1=.3, epsV2=.1, epsD1=.01, epsD2=.01, epsL=.1, epsC=.05,
                nBurnin=1000, nCollect=5000, thin=1, numThreads=nThreads)

# get signals
# d0 and p0 are the parameters when calculating the posterior exceedance probabilities
sig11 = Signal(mod11$B, d0=1, p0=0.9)
sig11$Prob  # posterior exceedance probabilities based on d0
sig11$Final # signals using a cut-off of p0 on the posterior exceedance probabilities

