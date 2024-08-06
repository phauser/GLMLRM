#include <RcppDist.h>
using namespace Rcpp;
using namespace arma;

// [[Rcpp::depends(RcppArmadillo, RcppDist)]]

// [[Rcpp::export]]
List GLRR(mat X, mat Y0, std::string priorC, std::string est, double Q, int r, int q, int nBurnin, int nCollect){

  // get dimensions    
  int n = X.n_rows; int p = X.n_cols; int d = Y0.n_cols;
  
  // initialize hyperparameters
  double a0 = 0.001;

  double a_sig = 1;      // gamma shape hyperparameter for sigma_k^(-2)
  double b_sig = 0.3;    // gamma rate=1/scale hyperparameter for sigma_k^(-2)
  double nu = 3;         // gamma shape & rate hyperparameter for phi_{kh}
  double a_psi1 = 2;     // gamma shape hyperparameter for psi_1
  double b_psi1 = 1;     // gamma rate=1/scale hyperparameter for psi_1
  double a_psi2 = 3;     // gamma shape hyperparameter for psi_g, g >= 2
  double b_psi2 = 1;     // gamma rate=1/scale hyperparameter for psi_g
  
  // initialize mean parameters
  mat U1 = ones(p,r)*0.1;
  mat V1 = ones(d,r)*0.1;
  vec Delta1; Delta1.ones(r);
  mat B0 = U1*diagmat(Delta1)*V1.t();
  // mat B0 = zeros(p,d);

  mat C = rnorm(p*d, 0, 0.5); C.reshape(p,d);
  mat B = B0+C;
  mat Z = rbinom(p*d, 1, Q); Z.reshape(p,d);
  
  // initialize precision parameters
  mat O = eye(d,d);
  double tau_del1 = 1; double tau_u1 = 1; mat Tau_v1 = ones(d,1);
  double tau0 = 1000; double tau1 = 0.1; double tau_c=0.1;
  // double tau1 = 1; double tau_e = 1; double tau_b = 1; 
  
  vec sig = rgamma(d,a_sig,1.0/b_sig);           // diagonals of sigma_k^(-2), d x 1
  mat Sigma_xi = diagmat(1.0/sig);               // Sigma_xi
  mat Lambda = zeros(d,q);                       // factor loading matrix, d x q 
  mat eta = rmvnorm(n,zeros(q,1),eye(q,q));      // latent factors (n x q)
  mat phi = rgamma(d*q,nu/2,2.0/nu); phi.reshape(d,q); 
  mat psi1 = rgamma(1,a_psi1,b_psi1);                   
  mat psi2 = rgamma(q-1,a_psi2,b_psi2);
  vec psi = join_cols(psi1, psi2);                // global shrinkage coefficients multipliers
  vec tau = cumprod(psi);                         // global shrinkage parameter (q x 1)
  mat Dinv = phi%repmat(tau.t(),d,1);             // D inverse (since we didn't invert tau and phi), dxq (not D_k)
  
  // initialize outputs
  mat SumB = zeros(p,d);
  mat SumSB = zeros(p,d);
  mat SumB0 = zeros(p,d);
  mat SumO = zeros(d,d);
  mat SumSO = zeros(d,d);
  mat SumC = zeros(p,d);
  mat SumZ = zeros(p,d);
  mat SumSC = zeros(p,d);
  mat SumL = zeros(d,q);
  mat SumSigmaXi = zeros(d,d);
  mat SumE = zeros(n,q);
  double SumVarC = 0;
  
  cube Output_B(p,d,nCollect);
  cube Output_B0(p,d,nCollect);
  cube Output_C(p,d,nCollect);
  cube Output_Z(p,d,nCollect);
  cube Output_O(d,d,nCollect);
  vec  Output_varC(nCollect);
  vec rate;
  vec dev0(nCollect);
  
  // start the sampling                
  for(int iter=0; iter<nBurnin+nCollect; ++iter){
      
    // subtract all the layers
    mat Y = Y0-X*B;
    // mat Y = Y0-X*B0; // this improved acc(B0) but decreases acc(B)
   
    // sample B1
    for(int l=0; l<r; ++l){
      // add back the l-th layer
      Y = Y+X*Delta1(l)*U1.col(l)*V1.col(l).t();

      // sample U, sig_Ul is p x p
      double temp = as_scalar(pow(Delta1(l),2)*V1.col(l).t()*O*V1.col(l)); // error with dimensions of 1x1 times pxp
      mat sig_Ul = inv(tau_u1*eye(p,p)+temp*X.t()*X);
      vec mu_Ul = Delta1(l)*sig_Ul*(X.t()*Y*O*V1.col(l));                 // p x 1
      U1.col(l) = rmvnorm(1,mu_Ul,sig_Ul).t();

      // sample V, sig_vl is d x d
      double temp2=as_scalar(pow(Delta1(l),2)*U1.col(l).t()*X.t()*X*U1.col(l));
      mat sig_Vl = inv(diagmat(Tau_v1)+temp2*O);
      vec mu_Vl = Delta1(l)*sig_Vl*O*Y.t()*X*U1.col(l);
      V1.col(l) = rmvnorm(1, mu_Vl,sig_Vl).t();

      // sample Delta
      double sig_Delta = 1/(as_scalar(tau_del1+V1.col(l).t()*O*V1.col(l)*U1.col(l).t()*X.t()*X*U1.col(l)));
      double mu_Delta = as_scalar(sig_Delta*U1.col(l).t()*X.t()*Y*O*V1.col(l));
      Delta1(l) = rnorm(1,mu_Delta, sqrt(sig_Delta))[0];

      // subtracting the l-th layer again
      Y = Y-X*Delta1(l)*U1.col(l)*V1.col(l).t();
    }
    B0 = U1*diagmat(Delta1)*V1.t();
    B = B0+C;
    Y = Y0-X*B;
    
    // sample C
    if(priorC=="normal"){
      for(int l=0; l<d; ++l){
        mat sig_Cl = inv(tau_c*eye(p,p)+O(l,l)*X.t()*X);
        vec mu_Cl = O(l,l)*sig_Cl*(X.t()*Y.col(l));
        C.col(l) = rmvnorm(1,mu_Cl,sig_Cl).t();
      }
      if(est=="yes"){
        rate   = 0.5*a0 + 0.5*sum(sum(pow(C,2)));
        tau_c  = rgamma(1, 0.5*a0 + 0.5*d*p, 1.0/rate(0))(0);
      }

    }
    if(priorC=="SS"){
      tau_c=tau1;
      for(int l=0; l<d; ++l){
        mat sig_Cl1 = inv(tau0*eye(p,p)+O(l,l)*X.t()*X);
        vec mu_Cl1 = O(l,l)*sig_Cl1*(X.t()*Y.col(l)); 
        vec tempC1 = rmvnorm(1,mu_Cl1,sig_Cl1).t();
        
        mat sig_Cl2 = inv(tau1*eye(p,p)+O(l,l)*X.t()*X);
        vec mu_Cl2 = O(l,l)*sig_Cl2*(X.t()*Y.col(l)); 
        vec tempC2 = rmvnorm(1,mu_Cl2,sig_Cl2).t();
        
        C.col(l) = (1-Z.col(l))%tempC1+Z.col(l)%tempC2;
        
        // update Z
        for(int ll=0; ll<p; ++ll){
          double pi = Q*R::dnorm(C(ll,l), 0, 1.0/sqrt(tau1), FALSE)/(Q*R::dnorm(C(ll,l), 0, 1.0/sqrt(tau1), FALSE)+ (1-Q)*R::dnorm(C(ll,l), 0, 1.0/sqrt(tau0), FALSE));
          Z(ll,l) = rbinom(1, 1, pi)(0);
        }

      }
      
      if(est=="yes"){
        rate   = 0.5*a0 + 0.5*sum(sum(pow(C,2)));
        tau_c  = rgamma(1, 0.5*a0 + 0.5*d*p, 1.0/rate(0))(0);
        tau0=tau_c;
      }
    }
    //B=C;
    B = B0+C;
    mat E = Y0-X*B;

    // Update Lambda - use Cholesky/QR decomp to find the inverse (which is the variance)
    for(int k=0; k<d; ++k){
      mat V_lam1 = sig(k)*(eta.t()*eta)+diagmat(Dinv.row(k));
      mat T = chol(V_lam1);
      mat Q; mat R; qr(Q,R,T);
      mat S = inv(R);
      mat V_lam = S*S.t();
      mat E_lam = V_lam*sig(k)*eta.t()*E.col(k);
      Lambda.row(k) = rmvnorm(1,E_lam,V_lam);
    }

    // Update Sigma_xi
    for(int k=0; k<d; ++k){
      vec temp(n); vec sig_scale(d);

      for(int i=0; i<n; ++i){
        temp(i)=as_scalar(pow(E(i,k)-Lambda.row(k)*eta.row(i).t(),2));
      }
      sig_scale(k)=1.0/(b_sig+0.5*sum(temp));
      sig(k) = rgamma(1, a_sig + n/2, sig_scale(k))[0];
    }
    Sigma_xi = diagmat(1.0/sig);

    // Update eta
    mat V_eta1 = eye(q,q)+Lambda.t()*Sigma_xi*Lambda;
    mat T = chol(V_eta1);
    mat Q; mat R; qr(Q, R, T);
    mat S = inv(R);
    mat V_eta = S*S.t();
    mat M_eta(q,n);

    for(int i=0; i<n; ++i){
      M_eta.col(i)= V_eta*Lambda.t()*Sigma_xi*E.row(i).t();
      eta.row(i) = rmvnorm(1,M_eta.col(i),V_eta);
    }

    // Update phi_jhs
    mat phi_scale=0.5*nu+0.5*pow(Lambda,2)%repmat(tau.t(),d,1);
    for(int j=0; j<d; ++j){
      for(int h=0; h<q; ++h){
        phi(j,h) = rgamma(1, 0.5*nu + 0.5, 1.0/phi_scale(j,h))[0];
      }
    }

    // Update psi
    mat phi_lam         = sum(phi%pow(Lambda, 2));
    double phi_lam_tau  = arma::as_scalar(tau.t()*phi_lam.t());
    double b_psi        = b_psi1 + 0.5*(1.0/psi(0))*phi_lam_tau;
    psi(0)              = rgamma(1, a_psi1 + 0.5*d*q, 1.0/b_psi)(0);
    tau                 = cumprod(psi);

    for(int j=1; j<q; ++j){
      double a_psi = a_psi2 + 0.5*d*(q-j+1); // (q-j)
      vec temp1 = (tau.t()%phi_lam).t();
      double b_psi = b_psi1+0.5*(1.0/psi(j))*accu(temp1.subvec(j-1,q-1)); //.subvec(k,k-1)
      psi(j) = rgamma(1,a_psi,1.0/b_psi)[0];
      tau = cumprod(psi);
    }

    // Update precision parameters
    O = inv(Lambda*Lambda.t()+Sigma_xi);
    
    // log-likehood for deviance calculation
    mat ll0 = n+log(det(Sigma_xi))+(Y0-X*B-eta*Lambda.t())*inv(Sigma_xi)*(Y0-X*B-eta*Lambda.t()).t();
    
    
    // Collect samples
    if((iter+1) > nBurnin){
      int idx = iter-nBurnin; 
      Output_B.slice(idx)   = B;
      Output_B0.slice(idx)  = B0;
      Output_C.slice(idx)   = C;
      Output_Z.slice(idx)   = Z;
      Output_varC(idx)      = 1.0/tau_c;
      dev0(idx)             = sum(ll0.diag());
      SumB += B;
      SumB0 += B0;
      SumC += C;
      SumZ += Z;
      SumO += O;
      SumL += Lambda;
      SumSigmaXi += Sigma_xi;
      SumE += eta;
      SumVarC += 1.0/tau_c;
    }
  }
  
  // things for the funtion to output
  List Output;
  int N0 = nCollect;
  mat PostB = SumB/N0;
  mat PostL = SumL/N0;
  mat PostE = SumE/N0;
  mat PostSig = SumSigmaXi/N0;
  mat dev1 = n+log(det(PostSig))+(Y0-X*PostB-PostE*PostL.t())*inv(PostSig)*(Y0-X*PostB-PostE*PostL.t()).t();
  Output["DIC"] = 2*mean(dev0)-sum(dev1.diag());
  Output["B"]=Output_B;
  Output["B0"]=Output_B0;
  Output["C"]=Output_C;
  Output["Z"]=Output_Z;
  Output["varC"]=Output_varC;
  Output["PostB"]  = SumB/N0;
  Output["PostB0"]  = SumB0/N0;
  Output["PostC"]  = SumC/N0;
  Output["PostZ"]  = SumZ/N0;
  Output["PostO"] = SumO/N0;
  Output["PostLambda"] = SumL/N0;
  Output["PostSigmaXi"] = SumSigmaXi/N0;
  Output["PostE"] = SumE/N0;
  Output["PostVarC"] = SumVarC/N0;
  return Output;
}


// [[Rcpp::export]]
List PerformGLRR(mat Y, mat Yh, mat SigX, int r, mat Bh, mat B, mat B0h, mat B0, mat Ch, mat C){
  int p = Bh.n_rows;
  int d = Bh.n_cols;
  int n = Y.n_rows;
  int df = r*(p+d);
  
  Yh = Yh-ones(n,1)*mean(Yh);
  Y = Y-ones(n,1)*mean(Y);
  
  double SSe = trace((Yh-Y).t()*(Yh-Y));
  
  // double MENB = 100*trace((Bh-B).t()*SigX*(Bh-B))/trace(B.t()*SigX*B);
  // double AIC = log(SSe) + 2*df/(n*d);
  double BIC = log(SSe) + df*log(n*d)/(n*d);
  // double PEN = 100*SSe/trace(Y.t()*Y);
  // double Rsq1 = 100*trace(Yh.t()*Yh)/trace(Y.t()*Y);
  
  mat bias_B = zeros(p,d);
  mat MSE_B = zeros(p,d);
  mat bias_B0 = zeros(p,d);
  mat MSE_B0 = zeros(p,d);
  mat bias_C = zeros(p,d);
  mat MSE_C = zeros(p,d);
  mat accB0 = zeros(p,d);
  mat accC = zeros(p,d);
  mat acc = zeros(p,d);
  for(int i=0; i<p; i++){
    for(int j=0; j<d; j++){;
      bias_B(i,j)=Bh(i,j)-B(i,j);
      MSE_B(i,j)=pow(bias_B(i,j),2);
      bias_B0(i,j)=B0h(i,j)-B0(i,j);
      MSE_B0(i,j)=pow(bias_B0(i,j),2);
      bias_C(i,j)=Ch(i,j)-C(i,j);
      MSE_C(i,j)=pow(bias_C(i,j),2);
      // to determine how many elements are correctly estimated
      if(bias_B0(i,j)<0.3 & bias_B0(i,j)>-0.3){accB0(i,j)=1;}
      if(bias_C(i,j)<0.3 & bias_C(i,j)>-0.3){accC(i,j)=1;}
      if(bias_B(i,j)<0.3 & bias_B(i,j)>-0.3){acc(i,j)=1;}
    }
  }
  
  return List::create(// Named("MENB")=MENB, 
          Named("MSE(B)")=median(median(MSE_B)), 
          Named("bias(B)")=median(median(abs(bias_B))),
          Named("accB")=sum(sum(acc)),
          Named("MSE(B0)")=median(median(MSE_B0)), 
          Named("bias(B0)")=median(median(abs(bias_B0))),
          Named("accB0")=sum(sum(accB0)),
          Named("MSE(C)")=median(median(MSE_C)), 
          Named("bias(C)")=median(median(abs(bias_C))),
          Named("accC")=sum(sum(accC)),
          Named("BIC")=BIC);
}


