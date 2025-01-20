# Improved Pharmacovigilance Signal Detection using Bayesian Generalized Linear Mixed Models
Vaccine safety monitoring is a critical component of public health given the extensive vaccination rate among the general population but most signal detection approaches overlook the inherently related biological nature of adverse events (AEs). We hypothesize that integration of AE field knowledge into the statistical process can facilitate in and improve accuracy of identifying vaccine-AE associations. 

Our Bayesian generalized linear multiple low-rank mixed model (GLMLRM) was developed for the analysis of high-dimensional post-market drug safety databases. The GLMLRM combines integration of AE ontology in the form of outcome-level groupings, low-rank matrices corresponding to these groupings to approximate the high-dimensional regression coefficient matrix, a factor analysis model to describe the dependence among responses, and a sparse coefficient matrix to capture uncertainty in both the imposed low-rank structures and user-specified groupings. An efficient Metropolis/ Gamerman-within-Gibbs sampling procedure is employed to obtain posterior estimates of the regression coefficients and other model parameters, from which testing of outcome-covariate pair associations is based. 

## Applying the GLMLRM on VAERS data
To run the GLMLRM on VAERS data, see the VAERS folder. runGLMLRMonVAERS.R contains R code to load in data extracted from the 2022 VAERS database (contained in VAERS.Rdata), run the GLMLRM (contained in GLMLRM.cpp) with either 1 or 2 outcome groupings, and get the final signals detected based on the posterior exceedance probablities.

## Simulations
If one is interested in running simulations, see the Simulations folder. BinarySimulations.R and NormalSimulations.R include R code to simulate data, run the GLMLRM, contained in GLMLRM.cpp and GLMLRM_normal.cpp, and calculate performance evaluations for binary and normal outcomes, respectively.

