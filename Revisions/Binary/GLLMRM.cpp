#include <RcppDist.h>
#include <RcppArmadillo.h>
#include <RcppParallel.h>

// #include <cmath>
// #include <random>
// #include <string>
// #include <vector>

using namespace Rcpp;
using namespace arma;
using namespace RcppParallel;

// [[Rcpp::depends(RcppArmadillo, RcppDist, RcppParallel)]]

inline double postDelta(std::size_t i, 
    const arma::mat &Y,      const arma::mat &X, const arma::mat &eta, 
    const arma::mat &lambda, const arma::mat &U, const arma::mat &V,   
    const arma::vec &delta,  double tau) {
  double lp = arma::as_scalar(-0.5*tau*pow(delta(i),2) + 
                delta(i)*U.col(i).t()*X.t()*Y*V.col(i) - 
                sum(sum(log(1+exp(X*U*diagmat(delta)*V.t()+eta*lambda.t())))));
  return lp;  
}

inline double postU(std::size_t j, 
    const arma::mat &Y,      const arma::mat &X, const arma::mat &eta,   
    const arma::mat &lambda, const arma::mat &U, const arma::mat &V,
    const arma::vec &delta,  double tau) {
  double lp = arma::as_scalar(-0.5*tau*U.col(j).t()*U.col(j) + 
                      delta(j)*U.col(j).t()*X.t()*Y*V.col(j) -
                      sum(sum(log(1+exp(X*U*diagmat(delta)*V.t()+eta*lambda.t())))));
  return lp;  
}

inline double postV(std::size_t j, 
    const arma::mat &Y,      const arma::mat &X, const arma::mat &eta, 
    const arma::mat &lambda, const arma::mat &U, const arma::mat &V,   
    const arma::vec &delta,  double tau) {
  
  double lp = arma::as_scalar(-0.5*tau*V.col(j).t()*V.col(j) + 
                              delta(j)*U.col(j).t()*X.t()*Y*V.col(j) -
                              sum(sum(log(1+exp(X*U*diagmat(delta)*V.t()+eta*lambda.t())))));
  return lp;  
}

inline double postLambda(std::size_t i, 
    const arma::mat &Y,      const arma::mat &X, const arma::mat &eta,   
    const arma::mat &lambda, const arma::mat &B, const arma::mat &Dinvi) {
  
  double lp = arma::as_scalar(-0.5*lambda.row(i)*Dinvi*lambda.row(i).t() +
            lambda.row(i)*eta.t()*Y.col(i) -
            sum(log(1+exp(X*B.col(i)+eta*lambda.row(i).t()))));
  return lp;  
}

inline double postEta(std::size_t i, 
    const arma::mat &Y,      const arma::mat &X, const arma::mat &eta,
    const arma::mat &lambda, const arma::mat &B, double tau) {
  
  double lp = arma::as_scalar(-0.5*tau*eta.row(i)*eta.row(i).t()
                                + sum(Y.row(i)*lambda*eta.row(i).t())
                                - sum(log(1+exp(X.row(i)*B+eta.row(i)*lambda.t()))));
                                return lp;  
}

inline double postC(std::size_t j, 
                      const arma::mat &Y,      const arma::mat &X, const arma::mat &eta,
                      const arma::mat &lambda, const arma::mat &B0, const arma::mat &C, double tau) {
  
  double lp = arma::as_scalar(-0.5*tau*C.col(j).t()*C.col(j) +
                              C.col(j).t()*X.t()*Y.col(j) -
                              sum(sum(log(1+exp(X*B0.col(j)+X*C.col(j)+eta*lambda.row(j).t())))));
  return lp;  
}

arma::vec UpdateDelta(std::size_t i, 
    const arma::mat &Y,      const arma::mat &X, const arma::mat &eta, 
    const arma::mat &lambda, const arma::mat &U, const arma::mat &V, 
    const arma::vec &delta,  double tau, double eps) {
  arma::vec res(2);
  for (int i = 0; i < delta.size(); i++) {
    arma::vec delta_new = delta;
    delta_new(i) = arma::as_scalar(arma::randn(1)*eps+delta(i));
    double l0 = postDelta(i, Y, X, eta, lambda, U, V, delta,     tau);
    double l1 = postDelta(i, Y, X, eta, lambda, U, V, delta_new, tau);
    
    // sample with acceptance probability min(1, exp(l1-l0))
    res(0) = arma::as_scalar(arma::randu(1)<std::min(1.0, exp(l1-l0)));
    
    // return old value if acc=0 and new value if acc=1
    res(1) = res(0)*delta_new(i) + (1-res(0))*delta(i);
  }
  
  return res;
}

struct updateU : public Worker {
  
  // input matrix to read from
  const RMatrix<double> Y;
  const RMatrix<double> X;
  const RMatrix<double> eta;
  const RMatrix<double> lambda;
  const RMatrix<double> U;
  const RMatrix<double> V;
  const RVector<double> delta;
  double tau;
  double eps;
  int j;
  int n;
  int p;
  int d;
  int q;
  int r;
  
  // output matrix to write to
  RMatrix<double> res;
  
  // initialize from Rcpp input and output matrixes (the RMatrix class
  // can be automatically converted to from the Rcpp matrix type)
  updateU(const NumericMatrix Y,     const NumericMatrix X, 
          const NumericMatrix eta,   const NumericMatrix lambda,
          const NumericMatrix U,     const NumericMatrix V,
          const NumericVector delta, double tau, double eps,
          int j, int n, int p, int d, int q, int r, NumericMatrix res)
    : Y(Y), X(X), eta(eta), lambda(lambda), U(U), V(V), delta(delta), 
      tau(tau), eps(eps), j(j), n(n), p(p), d(d), q(q), r(r), res(res) {}
  
  arma::vec convertDelta(){
    RVector<double> d = delta;
    arma::vec VEC(d.begin(), r, false);
    return VEC;
  }
  arma::mat convertY(){
    RMatrix<double> y = Y;
    arma::mat MAT(y.begin(), n, d, false);
    return MAT;
  }
  arma::mat convertX(){
    RMatrix<double> x = X;
    arma::mat MAT(x.begin(), n, p, false);
    return MAT;
  }
  arma::mat convertEta(){
    RMatrix<double> Eta = eta;
    arma::mat MAT(Eta.begin(), n, q, false);
    return MAT;
  }
  arma::mat convertLambda(){
    RMatrix<double> Lambda = lambda;
    arma::mat MAT(Lambda.begin(), d, q, false);
    return MAT;
  }
  arma::mat convertU(){
    RMatrix<double> u = U;
    arma::mat MAT(u.begin(), p, r, false);
    return MAT;
  }
  arma::mat convertV(){
    RMatrix<double> v = V;
    arma::mat MAT(v.begin(), d, r, false);
    return MAT;
  }
  
  void operator()(std::size_t begin, std::size_t end) {
    for (std::size_t i = begin; i < end; i++) {
      arma::mat y = convertY();
      arma::mat x = convertX();
      arma::mat e = convertEta();
      arma::mat l = convertLambda();
      arma::vec d = convertDelta();
      arma::mat u = convertU();
      arma::mat v = convertV();
      arma::mat u_new = u;
      
      u_new(i,j) = arma::as_scalar(arma::randn(1)*eps+u(i,j));
      double l0 = postU(j, y, x, e, l, u,     v, d, tau);
      double l1 = postU(j, y, x, e, l, u_new, v, d, tau);
      
      // decide if we will accept w/prob=min(1, exp(l1-l0))
      res(i,0) = arma::as_scalar(arma::randu(1)<std::min(1.0, exp(l1-l0)));
      
      // return old value if acc=0 and new value if acc=1
      res(i,1) = res(i,0)*u_new(i,j) + (1-res(i,0))*u(i,j);
    }
  }
};

arma::mat UpdateU(int j, 
    const NumericMatrix Y,      const NumericMatrix X,  const NumericMatrix eta, 
    const NumericMatrix lambda, const NumericMatrix U,  const NumericMatrix V, 
    const NumericVector delta,  double tau, double eps, int numThreads) {
  
  int n = Y.nrow();
  int d = Y.ncol();
  int p = X.ncol();
  int q = eta.ncol();
  int r = delta.length();
  
  // allocate the matrix we will return
  NumericMatrix res(U.nrow(), 2);
  
  // create the worker
  updateU obj(Y, X, eta, lambda, U, V, delta, tau, eps, j, n, p, d, q, r, res);
  
  // call it with parallelFor
  // parallelFor(0, U.nrow(), obj, numThreads);
  parallelFor(0, U.nrow(), obj);
  
  return as<arma::mat>(wrap(res));
}

struct updateV : public Worker {
  
  // input matrix to read from
  const RMatrix<double> Y;
  const RMatrix<double> X;
  const RMatrix<double> eta;
  const RMatrix<double> lambda;
  const RMatrix<double> U;
  const RMatrix<double> V;
  const RVector<double> delta;
  double tau;
  double eps;
  int j;
  int n;
  int p;
  int d;
  int q;
  int r;
  
  // output matrix to write to
  RMatrix<double> res;
  
  // initialize from Rcpp input and output matrixes (the RMatrix class
  // can be automatically converted to from the Rcpp matrix type)
  updateV(const NumericMatrix Y,     const NumericMatrix X, 
          const NumericMatrix eta,   const NumericMatrix lambda,
          const NumericMatrix U,     const NumericMatrix V,
          const NumericVector delta, double tau, 
          double eps, int j, int n, int p, int d, int q, int r, NumericMatrix res)
    : Y(Y), X(X), eta(eta), lambda(lambda), U(U), V(V), delta(delta), 
      tau(tau), eps(eps), j(j), n(n), p(p), d(d), q(q), r(r), res(res) {}
  
  arma::vec convertDelta(){
    RVector<double> d = delta;
    arma::vec VEC(d.begin(), r, false);
    return VEC;
  }

  arma::mat convertY(){
    RMatrix<double> y = Y;
    arma::mat MAT(y.begin(), n, d, false);
    return MAT;
  }
  arma::mat convertX(){
    RMatrix<double> x = X;
    arma::mat MAT(x.begin(), n, p, false);
    return MAT;
  }
  arma::mat convertEta(){
    RMatrix<double> Eta = eta;
    arma::mat MAT(Eta.begin(), n, q, false);
    return MAT;
  }
  arma::mat convertLambda(){
    RMatrix<double> Lambda = lambda;
    arma::mat MAT(Lambda.begin(), d, q, false);
    return MAT;
  }
  arma::mat convertU(){
    RMatrix<double> u = U;
    arma::mat MAT(u.begin(), p, r, false);
    return MAT;
  }
  arma::mat convertV(){
    RMatrix<double> v = V;
    arma::mat MAT(v.begin(), d, r, false);
    return MAT;
  }
  
  void operator()(std::size_t begin, std::size_t end) {
    for (std::size_t i = begin; i < end; i++) {
      arma::mat y = convertY();
      arma::mat x = convertX();
      arma::mat e = convertEta();
      arma::mat l = convertLambda();
      arma::vec d = convertDelta();
      arma::mat u = convertU();
      arma::mat v = convertV();
      arma::mat v_new = v;
      
      v_new(i,j) = arma::as_scalar(arma::randn(1)*eps+v(i,j));
      double l0 = postV(j, y, x, e, l, u, v,     d, tau);
      double l1 = postV(j, y, x, e, l, u, v_new, d, tau);
      
      // decide if we will accept w/prob=min(1, exp(l1-l0))
      res(i,0) = arma::as_scalar(arma::randu(1)<std::min(1.0, exp(l1-l0)));
      
      // return old value if acc=0 and new value if acc=1
      res(i,1) = res(i,0)*v_new(i,j) + (1-res(i,0))*v(i,j);
    }
  }
};

arma::mat UpdateV(int j, 
    const NumericMatrix Y,      const NumericMatrix X,   const NumericMatrix eta,   
    const NumericMatrix lambda, const NumericMatrix U,   const NumericMatrix V, 
    const NumericVector delta,  double tau, double eps, int numThreads) {
  
  int n = Y.nrow();
  int d = Y.ncol();
  int p = X.ncol();
  int q = eta.ncol();
  int r = delta.length();
  
  // allocate the matrix we will return
  NumericMatrix res(V.nrow(), 2);
  
  // create the worker
  updateV obj(Y, X, eta, lambda, U, V, delta, tau, eps, j, n, p, d, q, r, res);
  
  // call it with parallelFor
  // parallelFor(0, V.nrow(), obj, numThreads);
  parallelFor(0, V.nrow(), obj);
  
  return as<arma::mat>(wrap(res));
}

struct updateLambda : public Worker {
  
  // input matrix to read from
  std::string factor;
  const RMatrix<double> Y;
  const RMatrix<double> X;
  const RMatrix<double> eta;
  const RMatrix<double> lambda;
  const RMatrix<double> B;
  const RMatrix<double> tau;
  double eps;
  int n;
  int p;
  int d;
  int q;
  
  // output matrix to write to
  RMatrix<double> res;
  
  // initialize from Rcpp input and output matrixes (the RMatrix class
  // can be automatically converted to from the Rcpp matrix type)
  updateLambda(
               const NumericMatrix Y,     const NumericMatrix X, 
               const NumericMatrix eta,   const NumericMatrix lambda,
               const NumericMatrix B, const NumericMatrix tau, 
               double eps, int n, int p, int d, int q, NumericMatrix res)
    : Y(Y), X(X), eta(eta), lambda(lambda), B(B), 
      tau(tau), eps(eps), n(n), p(p), d(d), q(q), res(res) {}
  

  
  arma::mat convertTau(){
    RMatrix<double> Tau = tau;
    arma::mat MAT(Tau.begin(), d, q, false);
    return MAT;
  }
  arma::mat convertY(){
    RMatrix<double> y = Y;
    arma::mat MAT(y.begin(), n, d, false);
    return MAT;
  }
  arma::mat convertX(){
    RMatrix<double> x = X;
    arma::mat MAT(x.begin(), n, p, false);
    return MAT;
  }
  arma::mat convertEta(){
    RMatrix<double> Eta = eta;
    arma::mat MAT(Eta.begin(), n, q, false);
    return MAT;
  }
  arma::mat convertLambda(){
    RMatrix<double> Lambda = lambda;
    arma::mat MAT(Lambda.begin(), d, q, false);
    return MAT;
  }
  arma::mat convertB(){
    RMatrix<double> b = B;
    arma::mat MAT(b.begin(), p, d, false);
    return MAT;
  }

  void operator()(std::size_t begin, std::size_t end) {
    for (std::size_t i = begin; i < end; i++) {
      for(std::size_t j=0; j<lambda.ncol(); j++){
        arma::mat y = convertY();
        arma::mat x = convertX();
        arma::mat e = convertEta();
        arma::mat l = convertLambda();
        arma::mat b = convertB();
        arma::mat t = convertTau();
        arma::mat l_new = l;
        arma::mat Dinvi(q,q, arma::fill::eye);
        Dinvi = arma::diagmat(t.row(i));
        if(i>j){l_new(i,j) = arma::as_scalar(arma::randn(1)*eps+l(i,j));}
        
        double Q = arma::log_normpdf(l(i,j), l_new(i,j), eps) - arma::log_normpdf(l_new(i,j), l(i,j),     eps);
        
        double l0 = postLambda(i, y, x, e, l,     b, Dinvi);
        double l1 = postLambda(i, y, x, e, l_new, b, Dinvi);
        
        // decide if we will accept w/prob=min(1, exp(l1-l0))
        res(i,j) = arma::as_scalar(arma::randu(1)<std::min(1.0, exp(l1-l0+Q)));
        
        // return old value if acc=0 and new value if acc=1
        res(i,j+l.n_cols) = res(i,j)*l_new(i,j) + (1-res(i,j))*l(i,j);
      }
    }
  }
};
 
arma::mat UpdateLambda( 
    NumericMatrix Y, NumericMatrix X, NumericMatrix eta, NumericMatrix lambda, 
    NumericMatrix B, NumericMatrix tau, double eps, int numThreads) {
  
  int n = Y.nrow();
  int d = Y.ncol();
  int p = X.ncol();
  int q = lambda.ncol();
  
  // allocate the matrix we will return
  NumericMatrix res(lambda.nrow(), 2*lambda.ncol());
  
  // create the worker
  updateLambda obj(Y, X, eta, lambda, B, tau, eps, n, p, d, q, res);
  
  // call it with parallelFor
  // parallelFor(0, lambda.nrow(), obj, numThreads);
  parallelFor(0, lambda.nrow(), obj);
  
  return as<arma::mat>(wrap(res));
}

struct updateEta : public Worker {
  
  // input matrix to read from
  const RMatrix<double> Y;
  const RMatrix<double> X;
  const RMatrix<double> eta;
  const RMatrix<double> lambda;
  const RMatrix<double> B;
  double tau;
  double eps;
  int n;
  int p;
  int d;
  int q;
  std::string proposal;
  
  // output matrix to write to
  RMatrix<double> res;

  // initialize from Rcpp input and output matrixes (the RMatrix class
  // can be automatically converted to from the Rcpp matrix type)
  updateEta(std::string proposal, const NumericMatrix Y, const NumericMatrix X, 
            const NumericMatrix eta, const NumericMatrix lambda, const NumericMatrix B,
            double tau, double eps,
            int n, int p, int d, int q, NumericMatrix res)
    : proposal(proposal), Y(Y), X(X), eta(eta), lambda(lambda), B(B),
      tau(tau), eps(eps), n(n), p(p), d(d), q(q), res(res) {}
  
  arma::mat convertY(){
    RMatrix<double> y = Y;
    arma::mat MAT(y.begin(), n, d, false);
    return MAT;
  }
  arma::mat convertX(){
    RMatrix<double> x = X;
    arma::mat MAT(x.begin(), n, p, false);
    return MAT;
  }
  arma::mat convertEta(){
    RMatrix<double> Eta = eta;
    arma::mat MAT(Eta.begin(), n, q, false);
    return MAT;
  }
  arma::mat convertLambda(){
    RMatrix<double> Lambda = lambda;
    arma::mat MAT(Lambda.begin(), d, q, false);
    return MAT;
  }
  arma::mat convertB(){
    RMatrix<double> b = B;
    arma::mat MAT(b.begin(), p, d, false);
    return MAT;
  }

  void operator()(std::size_t begin, std::size_t end) {
    for (std::size_t i = begin; i < end; i++) {
      for(std::size_t j = 0; j < eta.ncol(); j++)
      {
        arma::mat y = convertY();
        arma::mat x = convertX();
        arma::mat e = convertEta();
        arma::mat l = convertLambda();
        arma::mat b = convertB();
        arma::mat e_new = e;
        double q;
        
        if(proposal=="normal"){
          e_new(i,j) = arma::as_scalar(arma::randn(1)*eps+e(i,j));
          q = arma::log_normpdf(e(i,j), e_new(i,j), eps) 
            - arma::log_normpdf(e_new(i,j), e(i,j), eps);
        }
        
        if(proposal=="Gamerman"){
          arma::vec W_ij0(d), W_ij1(d);
          arma::vec Y_XB0(d), Y_XB1(d);
          arma::mat I = arma::ones(1,d);
          
          for(int j0=0; j0<d; j0++){
            double eta_ij = arma::as_scalar(e.row(i)*l.row(j0).t());
            // double theta_ij = arma::as_scalar(x.row(i)*b.col(j0).t()+eta_ij);
            double theta_ij = arma::as_scalar(x.row(i)*b.col(j0)+eta_ij);
            W_ij0(j0) = arma::as_scalar(exp(theta_ij)/pow(1+exp(theta_ij),2));
            Y_XB0(j0) = eta_ij+pow(1.0+exp(theta_ij),2)/exp(theta_ij)*(y(i,j0)-exp(theta_ij)/(1.0+exp(theta_ij)));
          }
          
          double c_i0 = arma::as_scalar(1.0/(tau+I*arma::diagmat(W_ij0)*I.t()));
          double m_i0 = arma::as_scalar(c_i0*I*arma::diagmat(W_ij0)*Y_XB0);
          
          e_new(i,j) = arma::as_scalar(arma::randn(1)*eps*sqrt(c_i0)+m_i0);
          if(e_new(i,j)>1000 | e_new(i,j)<-1000){e_new(i,j) = arma::as_scalar(randn(1))*sqrt(c_i0)+m_i0;}
          if(e_new(i,j)>1000 | e_new(i,j)<-1000){e_new(i,j) = arma::as_scalar(randn(1))*sqrt(c_i0)+m_i0;}
          if(e_new(i,j)>1000 | e_new(i,j)<-1000){e_new(i,j) = arma::as_scalar(randn(1))*sqrt(c_i0)+m_i0;}
          if(e_new(i,j)>1000 | e_new(i,j)<-1000){e_new(i,j) = arma::as_scalar(randn(1))*sqrt(c_i0)+m_i0;}
          if(e_new(i,j)>1000 | e_new(i,j)<-1000){e_new(i,j) = arma::as_scalar(randn(1))*sqrt(c_i0)+m_i0;}
          if(e_new(i,j)>1000 | e_new(i,j)<-1000){e_new(i,j) = arma::as_scalar(randn(1))*sqrt(c_i0)+m_i0;}
          if(e_new(i,j)>1000 | e_new(i,j)<-1000){e_new(i,j) = arma::as_scalar(randn(1))*sqrt(c_i0)+m_i0;}
          if(e_new(i,j)>1000 | e_new(i,j)<-1000){e_new(i,j) = arma::as_scalar(randn(1))*sqrt(c_i0)+m_i0;}
          if(e_new(i,j)>1000 | e_new(i,j)<-1000){e_new(i,j) = arma::as_scalar(randn(1))*sqrt(c_i0)+m_i0;}
          if(e_new(i,j)>1000 | e_new(i,j)<-1000){e_new(i,j) = arma::as_scalar(randn(1))*sqrt(c_i0)+m_i0;}
          if(e_new(i,j)>1000 | e_new(i,j)<-1000){e_new(i,j) = e(i,j);}
          if(Rcpp::traits::is_nan<REALSXP>(e_new(i,j))){e_new(i,j) = arma::as_scalar(randn(1))*sqrt(c_i0)+m_i0;}
          if(Rcpp::traits::is_nan<REALSXP>(e_new(i,j))){e_new(i,j) = arma::as_scalar(randn(1))*sqrt(c_i0)+m_i0;}
          if(Rcpp::traits::is_nan<REALSXP>(e_new(i,j))){e_new(i,j) = arma::as_scalar(randn(1))*sqrt(c_i0)+m_i0;}
          if(Rcpp::traits::is_nan<REALSXP>(e_new(i,j))){e_new(i,j) = arma::as_scalar(randn(1))*sqrt(c_i0)+m_i0;}
          if(Rcpp::traits::is_nan<REALSXP>(e_new(i,j))){e_new(i,j) = arma::as_scalar(randn(1))*sqrt(c_i0)+m_i0;}
          if(Rcpp::traits::is_nan<REALSXP>(e_new(i,j))){e_new(i,j) = arma::as_scalar(randn(1))*sqrt(c_i0)+m_i0;}
          if(Rcpp::traits::is_nan<REALSXP>(e_new(i,j))){e_new(i,j) = arma::as_scalar(randn(1))*sqrt(c_i0)+m_i0;}
          if(Rcpp::traits::is_nan<REALSXP>(e_new(i,j))){e_new(i,j) = arma::as_scalar(randn(1))*sqrt(c_i0)+m_i0;}
          if(Rcpp::traits::is_nan<REALSXP>(e_new(i,j))){e_new(i,j) = arma::as_scalar(randn(1))*sqrt(c_i0)+m_i0;}
          if(Rcpp::traits::is_nan<REALSXP>(e_new(i,j))){e_new(i,j) = arma::as_scalar(randn(1))*sqrt(c_i0)+m_i0;}
          if(Rcpp::traits::is_nan<REALSXP>(e_new(i,j))){e_new(i,j) = e(i,j);}
          
          for(int j1=0; j1<d; j1++){
            double eta_ij = arma::as_scalar(e_new.row(i)*l.row(j1).t());
            // double theta_ij = arma::as_scalar(x.row(i)*b.col(j1).t()+eta_ij);
            double theta_ij = arma::as_scalar(x.row(i)*b.col(j1)+eta_ij);
            W_ij1(j1) = arma::as_scalar(exp(theta_ij)/pow(1+exp(theta_ij),2));
            Y_XB1(j1) = eta_ij+pow(1.0+exp(theta_ij),2)/exp(theta_ij)*(y(i,j1)-exp(theta_ij)/(1.0+exp(theta_ij)));
          }
          
          double c_i1 = arma::as_scalar(1.0/(tau+I*arma::diagmat(W_ij1)*I.t()));
          double m_i1 = arma::as_scalar(c_i1*I*arma::diagmat(W_ij1)*Y_XB1);
          
          q = arma::log_normpdf(e(i,j),     m_i1, eps*pow(c_i1, 0.5))
            - arma::log_normpdf(e_new(i,j), m_i0, eps*pow(c_i0, 0.5));
        }
        
        double l0 = postEta(i, y, x, e,     l, b, tau);
        double l1 = postEta(i, y, x, e_new, l, b, tau);
        
        // decide if we will accept w/prob=min(1, exp(l1-l0))
        res(i,j) = arma::as_scalar(arma::randu(1)<std::min(1.0, exp(l1-l0+q)));
        
        // return old value if acc=0 and new value if acc=1
        res(i,j+e.n_cols) = res(i,j)*e_new(i,j) + (1-res(i,j))*e(i,j);
      }
    }
  }
};


arma::mat UpdateEta(std::string proposal,
    const NumericMatrix Y,      const NumericMatrix X,  const NumericMatrix eta, 
    const NumericMatrix lambda, const NumericMatrix B,  double tau, double eps, int numThreads) {
  
  int n = Y.nrow();
  int d = Y.ncol();
  int p = X.ncol();
  int q = eta.ncol();
  
  // allocate the matrix we will return
  NumericMatrix res(eta.nrow(), 2*eta.ncol());
  
  // create the worker
  updateEta obj(proposal, Y, X, eta, lambda, B, tau, eps, n, p, d, q, res);
  
  // call it with parallelFor
  // parallelFor(0, eta.nrow(), obj, numThreads);
  parallelFor(0, eta.nrow(), obj);
  
  return as<arma::mat>(wrap(res));
}

struct updateC : public Worker {
  
  // input matrix to read from
  const RMatrix<double> Y;
  const RMatrix<double> X;
  const RMatrix<double> eta;
  const RMatrix<double> lambda;
  const RMatrix<double> B0;
  const RMatrix<double> C;
  double tau;
  double eps;
  int n;
  int p;
  int d;
  int q;
  
  // output matrix to write to
  RMatrix<double> res;
  
  // initialize from Rcpp input and output matrixes (the RMatrix class
  // can be automatically converted to from the Rcpp matrix type)
  updateC(const NumericMatrix Y,     const NumericMatrix X, 
          const NumericMatrix eta,   const NumericMatrix lambda,
          const NumericMatrix B0,    const NumericMatrix C,
          double tau, double eps,
          int n, int p, int d, int q, NumericMatrix res)
    : Y(Y), X(X), eta(eta), lambda(lambda), B0(B0), C(C), 
      tau(tau), eps(eps), n(n), p(p), d(d), q(q), res(res) {}
  
  arma::mat convertY(){
    RMatrix<double> y = Y;
    arma::mat MAT(y.begin(), n, d, false);
    return MAT;
  }
  arma::mat convertX(){
    RMatrix<double> x = X;
    arma::mat MAT(x.begin(), n, p, false);
    return MAT;
  }
  arma::mat convertEta(){
    RMatrix<double> Eta = eta;
    arma::mat MAT(Eta.begin(), n, q, false);
    return MAT;
  }
  arma::mat convertLambda(){
    RMatrix<double> Lambda = lambda;
    arma::mat MAT(Lambda.begin(), d, q, false);
    return MAT;
  }
  arma::mat convertB0(){
    RMatrix<double> b0 = B0;
    arma::mat MAT(b0.begin(), p, d, false);
    return MAT;
  }
  arma::mat convertC(){
    RMatrix<double> c = C;
    arma::mat MAT(c.begin(), p, d, false);
    return MAT;
  }
  

  void operator()(std::size_t begin, std::size_t end) {
    for (std::size_t i = begin; i < end; i++) {
      for(std::size_t j = 0; j < C.ncol(); j++){
        arma::mat y = convertY();
        arma::mat x = convertX();
        arma::mat e = convertEta();
        arma::mat l = convertLambda();
        arma::mat b0 = convertB0();
        arma::mat c = convertC();
        arma::mat c_new = c;
        
        c_new(i,j) = arma::as_scalar(arma::randn(1)*eps+c(i,j));
        double l0 = postC(j, y, x, e, l, b0, c, tau);
        double l1 = postC(j, y, x, e, l, b0, c_new, tau);
        
        // decide if we will accept w/prob=min(1, exp(l1-l0))
        res(i,j) = arma::as_scalar(arma::randu(1)<std::min(1.0, exp(l1-l0)));
        
        // return old value if acc=0 and new value if acc=1
        res(i,j+c.n_cols) = res(i,j)*c_new(i,j) + (1-res(i,j))*c(i,j);
      }
    }
  }
};

arma::mat UpdateC(const NumericMatrix Y,      const NumericMatrix X,  const NumericMatrix eta, 
                  const NumericMatrix lambda, const NumericMatrix B0, const NumericMatrix C, 
                  double tau, double eps, int numThreads) {
  
  int n = Y.nrow();
  int d = Y.ncol();
  int p = X.ncol();
  int q = eta.ncol();

  // allocate the matrix we will return
  NumericMatrix res(C.nrow(), 2*C.ncol());
  
  // create the worker
  updateC obj(Y, X, eta, lambda, B0, C, tau, eps, n, p, d, q, res);
  
  // call it with parallelFor
  parallelFor(0, C.nrow(), obj);
  
  return as<arma::mat>(wrap(res));
}

// [[Rcpp::export]]
List GLMLRM2(std::string priorC, const arma::mat &X, const arma::mat &Y, 
           int & d1, int & d2, int & r1, int & r2, int & q,
           double & epsb, double & epsU1, double & epsU2, 
           double & epsV1, double & epsV2,
           double & epsD1, double & epsD2, 
           double & epsL, double & epsC,
           int & nBurnin, int & nCollect, int & thin, int numThreads){
  
  std::string proposal = "Gamerman"; // Gamerman sampling
  // std::string proposal = "normal"; // normal MRW sampling

  // get dimensions
  int n = X.n_rows, p = X.n_cols, d = Y.n_cols;
  
  // initialize parameters for random effects
  mat b         = zeros(n, q);
  double alpha  = 0.01;
  double taub   = rgamma(1, alpha, 1.0/alpha)(0); // Rcpp is shape and scale, R is shape and rate
  double varb   = 1.0/taub;
  vec rate;
  
  // initialize parameters for decomposition
  mat U1 = ones(p, r1)*0.1;
  mat V1 = ones(d1, r1)*0.1;
  vec delta1; delta1.ones(r1);
  mat B1 = U1*diagmat(delta1)*V1.t();
  
  mat U2 = ones(p, r2)*0.1;
  mat V2 = ones(d2, r2)*0.1;
  vec delta2; delta2.ones(r2);
  mat B2 = U2*diagmat(delta2)*V2.t();
  
  double tauU = 1.0/sqrt(p);
  double tauV1 = 1.0/sqrt(d1);
  double tauV2 = 1.0/sqrt(d2);
  double tauD = sqrt(r1);
 
  mat B0 = join_horiz(B1,B2);
  mat C = zeros(p,d);
  mat B = B0+C;
  double Q = 0.1;
  mat Z = rbinom(p*d, 1, Q); Z.reshape(p,d);
  double a0 = 0.01;
  double tauC = 10000;
  double tauC1 = 0.01;
  double varC = 1.0/tauC;
  
  // initialize parameters for estimating Lambda
  mat lambda     = rnorm(d*q, 0, 1);
  lambda.reshape(d,q);
  for(int i=0; i<d; i++){
    for(int j=0; j<q; j++){
      if(i==j){lambda(i,j)=1;}
      if(j>i){lambda(i,j)=0;}
    }
  }
  
  // for hierarchical factor model
  double nu      = 3.0;     // gamma shape & rate hyperparameter for phi_{jh}
  double a_psi1  = 2.0;     // gamma shape hyperparameter for psi_1
  double b_psi1  = 1.0;     // gamma rate = 1/scale hyperparameter for psi_1
  double a_psi2  = 3.0;     // gamma shape hyperparameter for psi_g, g >= 2
  double b_psi2  = 1.0;     // gamma rate=1/scale hyperparameter for psi_g
  
  mat phi        = rgamma(d*q, nu/2.0, nu/2.0);  // local shrinkage parameter
  phi.reshape(d,q);                              // d x q
  mat psi1       = rgamma(1,   a_psi1, b_psi1);
  mat psi2       = rgamma(q-1, a_psi2, b_psi2);
  vec psi        = join_cols(psi1, psi2);        // global shrinkage coefficients multipliers
  vec tau        = cumprod(psi);                 // global shrinkage parameter (q x 1)
  mat Dinv       = phi%repmat(tau.t(), d, 1.0);  // D inverse (since we didn't invert tau and phi), d x q
  // for Dinv_j we do diag(Dinv.row(j))
  // for thinning
  int runs       = (nCollect+nBurnin)*thin;
  int totalIter  = 0;                        // counter for total iterations
  int idx        = 0;                        // counter for thinned iterations
  
  // initialize outputs
  mat SumB       = zeros(p, d);
  mat SumB0      = zeros(p, d);
  mat SumC       = zeros(p, d);
  mat Sumb       = zeros(n, q);
  mat Sumlambda  = zeros(d, q);
  double Sumvarb = 0;
  double SumvarC = 0;
  
  cube Output_B(p, d, nCollect);
  cube Output_B0(p, d, nCollect);
  cube Output_C(p, d, nCollect);
  cube Output_Z(p, d, nCollect);
  // cube Output_b(n, d, nCollect);
  cube Output_lambda(d, q, nCollect);
  vec  Output_varb(nCollect);
  vec  Output_varC(nCollect);
  vec  Output_dev(nCollect);

  mat AR_U1         = zeros(p, r1);
  mat SumAR_U1      = zeros(p, r1);
  mat AR_V1         = zeros(d1, r1);
  mat SumAR_V1      = zeros(d1, r1);
  mat AR_delta1     = zeros(r1, 1);
  mat SumAR_delta1  = zeros(r1, 1);
  
  mat AR_U2         = zeros(p, r2);
  mat SumAR_U2      = zeros(p, r2);
  mat AR_V2         = zeros(d2, r2);
  mat SumAR_V2      = zeros(d2, r2);
  mat AR_delta2     = zeros(r2, 1);
  mat SumAR_delta2  = zeros(r2, 1);
  
  mat AR_C          = zeros(p, d);
  mat SumAR_C       = zeros(p, d);
  
  mat AR_b          = zeros(n, q);
  mat SumAR_b       = zeros(n, q);
  mat AR_lambda     = zeros(d, q);
  mat SumAR_lambda  = zeros(d, q);
  mat MH_U1, MH_U2, MH_V1, MH_V2, MH_b, MH_lambda, MH_C0, MH_C1, MH_C;
  vec MH_delta1, MH_delta2;
  
  // start the sampling
  for(int iter=0; iter<runs; iter++){
    
    // count the total number of iterations - want (nBurnin+nCollect)*thin total
    totalIter = iter;

    // sample B1
    for(int l=0; l<r1; l++){
      // Step 1: Update U
      MH_U1 = UpdateU(l, as<NumericMatrix>(wrap(Y.submat(0,0,n-1,d1-1))), as<NumericMatrix>(wrap(X)),
          as<NumericMatrix>(wrap(b)), as<NumericMatrix>(wrap(lambda.submat(0,0,d1-1,q-1))),
          as<NumericMatrix>(wrap(U1)), as<NumericMatrix>(wrap(V1)),
          as<NumericVector>(wrap(delta1)), tauU, epsU1, numThreads);
      U1.col(l)    = MH_U1.col(1);
      AR_U1.col(l) = MH_U1.col(0);

      // Step 2: Update V
      MH_V1 = UpdateV(l, as<NumericMatrix>(wrap(Y.submat(0,0,n-1,d1-1))), as<NumericMatrix>(wrap(X)),
          as<NumericMatrix>(wrap(b)), as<NumericMatrix>(wrap(lambda.submat(0,0,d1-1,q-1))),
          as<NumericMatrix>(wrap(U1)), as<NumericMatrix>(wrap(V1)),
          as<NumericVector>(wrap(delta1)), tauV1, epsV1, numThreads);
      V1.col(l)   = MH_V1.col(1);
      AR_V1.col(l) = MH_V1.col(0);

      // Step 3: Update delta
      MH_delta1 = UpdateDelta(l, Y.submat(0,0,n-1,d1-1), X, b, lambda.submat(0,0,d1-1,q-1), U1, V1, delta1, tauD, epsD1);
      delta1(l) = MH_delta1(1);
      AR_delta1(l) = MH_delta1(0);
    }
    B1 = U1*diagmat(delta1)*V1.t();

    // sample B2
    for(int l=0; l<r2; l++){
      // Step 1: Update U
      MH_U2 = UpdateU(l, as<NumericMatrix>(wrap(Y.submat(0,d1,n-1,d1+d2-1))), as<NumericMatrix>(wrap(X)),
                      as<NumericMatrix>(wrap(b)), as<NumericMatrix>(wrap(lambda.submat(d1,0,d1+d2-1,q-1))),
                      as<NumericMatrix>(wrap(U2)), as<NumericMatrix>(wrap(V2)),
                      as<NumericVector>(wrap(delta2)), tauU, epsU2, numThreads);
      U2.col(l)    = MH_U2.col(1);
      AR_U2.col(l) = MH_U2.col(0);

      // Step 2: Update V
      MH_V2 = UpdateV(l, as<NumericMatrix>(wrap(Y.submat(0,d1,n-1,d1+d2-1))), as<NumericMatrix>(wrap(X)),
                      as<NumericMatrix>(wrap(b)), as<NumericMatrix>(wrap(lambda.submat(d1,0,d1+d2-1,q-1))),
                      as<NumericMatrix>(wrap(U2)), as<NumericMatrix>(wrap(V2)),
                      as<NumericVector>(wrap(delta2)), tauV2, epsV2, numThreads);
      V2.col(l)    = MH_V2.col(1);
      AR_V2.col(l) = MH_V2.col(0);

      // Step 3: Update delta
      MH_delta2    = UpdateDelta(l, Y.submat(0,d1,n-1,d1+d2-1), X, b, lambda.submat(d1,0,d1+d2-1,q-1), U2, V2, delta2, tauD, epsD2);
      delta2(l)    = MH_delta2(1);
      AR_delta2(l) = MH_delta2(0);
    }
    B2 = U2*diagmat(delta2)*V2.t();

    B0 = join_horiz(B1,B2);

    // Sample C
    if(priorC=="SS"){
      MH_C0 = UpdateC(as<NumericMatrix>(wrap(Y)), as<NumericMatrix>(wrap(X)),
                      as<NumericMatrix>(wrap(b)), as<NumericMatrix>(wrap(lambda)),
                      as<NumericMatrix>(wrap(B0)), as<NumericMatrix>(wrap(C)),
                      tauC, epsC, numThreads);
      mat tempC0 = MH_C0.submat(0,d,C.n_rows-1,2*d-1);
      mat tempAR_C0 = MH_C0.submat(0,0,C.n_rows-1,d-1);
      MH_C1 = UpdateC(as<NumericMatrix>(wrap(Y)), as<NumericMatrix>(wrap(X)),
                      as<NumericMatrix>(wrap(b)), as<NumericMatrix>(wrap(lambda)),
                      as<NumericMatrix>(wrap(B0)), as<NumericMatrix>(wrap(C)),
                      tauC1, epsC, numThreads);
      mat tempC1 = MH_C1.submat(0,d,C.n_rows-1,2*d-1);
      mat tempAR_C1 = MH_C1.submat(0,0,C.n_rows-1,d-1);
      C =  (1-Z)%tempC0+Z%tempC1;
      AR_C = (1-Z)%tempAR_C0+Z%tempAR_C1;
      
     // Sample Z
    for(int l=0; l<d; ++l){
      for(int ll=0; ll<p; ++ll){
        double pi = Q*R::dnorm(C(ll,l), 0, 1.0/sqrt(tauC1), FALSE)/(Q*R::dnorm(C(ll,l), 0, 1.0/sqrt(tauC1), FALSE)+ (1-Q)*R::dnorm(C(ll,l), 0, 1.0/sqrt(tauC), FALSE));
        Z(ll,l) = rbinom(1, 1, pi)(0);
      }
    }
      
      //rate   = 0.5*a0 + 0.5*sum(sum(pow(C,2)));
      //tauC  = rgamma(1, 0.5*a0 + 0.5*d*p, 1.0/rate(0))(0);
      //varC  = 1.0/tauC;
    }
    if(priorC=="none"){C=zeros(p,d);}
    if(priorC=="normal"){
      MH_C = UpdateC(as<NumericMatrix>(wrap(Y)), as<NumericMatrix>(wrap(X)),
                      as<NumericMatrix>(wrap(b)), as<NumericMatrix>(wrap(lambda)),
                      as<NumericMatrix>(wrap(B0)), as<NumericMatrix>(wrap(C)),
                      tauC, epsC, numThreads);
      C = MH_C.submat(0,d,C.n_rows-1,2*d-1);
      AR_C = MH_C.submat(0,0,C.n_rows-1,d-1);
      rate   = 0.5*a0 + 0.5*sum(sum(pow(C,2)));
      tauC  = rgamma(1, 0.5*a0 + 0.5*d*p, 1.0/rate(0))(0);
      varC  = 1.0/tauC;
    }



    // Step 3.5: Update B
    B = B0+C;

    // Step 4: Update b
    MH_b = UpdateEta(proposal, as<NumericMatrix>(wrap(Y)), as<NumericMatrix>(wrap(X)),
        as<NumericMatrix>(wrap(b)), as<NumericMatrix>(wrap(lambda)), as<NumericMatrix>(wrap(B)),
        taub, epsb, numThreads);
    b = MH_b.submat(0, q, b.n_rows-1, 2*q-1);
    AR_b = MH_b.submat(0, 0, b.n_rows-1, q-1);

    // Step 5: Update tau_b
    rate = 0.5*(1.0/alpha) + 0.5*sum(sum(pow(b,2)));
    taub = rgamma(1, 0.5*n*q + 0.5*alpha, 1.0/rate(0))(0);
    varb = 1.0/taub;


    // Step 9: Update lambda d x q
    MH_lambda = UpdateLambda(as<NumericMatrix>(wrap(Y)),
                                     as<NumericMatrix>(wrap(X)),
                                     as<NumericMatrix>(wrap(b)),
                                     as<NumericMatrix>(wrap(lambda)),
                                     as<NumericMatrix>(wrap(B)),
                                     as<NumericMatrix>(wrap(Dinv)), epsL, numThreads);
    lambda = MH_lambda.submat(0, q, d-1, 2*q-1);
    AR_lambda = MH_lambda.submat(0, 0, d-1, q-1);

    // Step 10: Update phi  d x q
    mat phi_scale = 0.5*nu + 0.5*pow(lambda, 2)%repmat(tau.t(), d, 1);
    for(int j=0; j<d; ++j){for(int h=0; h<q; ++h){phi(j,h) = rgamma(1, 0.5*nu + 0.5, 1.0/phi_scale(j,h))(0);}}

    // Step 11: Update psi
    mat phi_lam         = sum(phi%pow(lambda, 2));
    double phi_lam_tau  = arma::as_scalar(tau.t()*phi_lam.t());  // July 14, 2021 1x2 1x1?
    double b_psi        = b_psi1 + 0.5*(1.0/psi(0))*phi_lam_tau;
    psi(0)              = rgamma(1, a_psi1 + 0.5*d*q, 1.0/b_psi)(0);
    tau                 = cumprod(psi);

    for(int j=1; j<q; ++j){
      double a_psi = a_psi2 + 0.5*d*(q-j);
      vec temp1    = (tau.t()%phi_lam).t();
      double b_psi = b_psi1 + 0.5*(1.0/psi(j))*accu(temp1.subvec(j,q-1));
      psi(j)       = rgamma(1, a_psi, 1.0/b_psi)(0);
      tau          = cumprod(psi);
    }

    // Step 12: Update Dinv
    Dinv = phi%repmat(tau.t(), d, 1);

    // calculate deviance
    vec ll10 = zeros(n);
    for(int i=0; i<n; i++){ll10(i) = arma::as_scalar(Y.row(i)*(B.t()*X.row(i).t()+lambda*b.row(i).t()));}
    double ll20 = arma::as_scalar(sum(sum(log(1+exp(X*B+b*lambda.t())))));
    double dev = -2*arma::as_scalar(sum(ll10) - ll20);
  
    // Collect samples

    if((iter+1) > (nBurnin*thin)){
      if(iter % thin == 0){
        Output_B.slice(idx)         = B;
        Output_B0.slice(idx)        = B0;
        Output_C.slice(idx)         = C;
        Output_Z.slice(idx)         = Z;
        Output_lambda.slice(idx)    = lambda;
        Output_varb(idx)            = arma::as_scalar(varb);
        Output_varC(idx)            = arma::as_scalar(varC);
        Output_dev(idx)             = dev;
        SumAR_U1      += AR_U1;
        SumAR_V1      += AR_V1;
        SumAR_delta1  += AR_delta1;
        SumAR_U2      += AR_U2;
        SumAR_V2      += AR_V2;
        SumAR_delta2  += AR_delta2;
        SumAR_b       += AR_b;
        SumAR_lambda  += AR_lambda;
        SumAR_C       += AR_C;
        SumC          += C;
        SumB          += B;
        SumB0         += B0;
        Sumb          += b;
        Sumlambda     += lambda;
        Sumvarb       += arma::as_scalar(varb);
        SumvarC       += arma::as_scalar(varC);

        idx = idx+1;
      }
    }
  }
  
  
  // Output
  List Output;
  int N0 = nCollect;
  Output["B"]               = Output_B;
  Output["B0"]              = Output_B0;
  Output["C"]               = Output_C;
  Output["Z"]               = Output_Z;
  Output["L"]               = Output_lambda;
  Output["varb"]            = Output_varb;
  Output["dev"]             = Output_dev;

  Output["PostB"]           = SumB/N0;
  Output["PostB0"]          = SumB0/N0;
  Output["PostC"]           = SumC/N0;
  Output["Postb"]           = Sumb/N0;
  Output["Postlambda"]      = Sumlambda/N0;
  Output["Postvarb"]        = Sumvarb/N0;

  Output["AR_U1"]           = SumAR_U1/N0;
  Output["AR_V1"]           = SumAR_V1/N0;
  Output["AR_delta1"]       = SumAR_delta1/N0;
  Output["AR_U2"]           = SumAR_U2/N0;
  Output["AR_V2"]           = SumAR_V2/N0;
  Output["AR_delta2"]       = SumAR_delta2/N0;
  Output["AR_C"]            = SumAR_C/N0;
  Output["AR_b"]            = mean(SumAR_b/N0);
  Output["AR_lambda"]       = SumAR_lambda/N0;
  return Output;
}


List perform(mat Y, mat X, mat B, cube Bh, mat C, cube Ch, mat lambda, mat b, vec dev0){
  int n = Y.n_rows;
  int p = B.n_rows;
  int d = B.n_cols;
  
  vec ll1_d = zeros(n);
  for(int i=0; i<n; i++){ll1_d(i) = arma::as_scalar(Y.row(i)*(B.t()*X.row(i).t()+lambda*b.row(i).t()));}
  double ll2_d = arma::as_scalar(sum(sum(log(1+exp(X*B+b*lambda.t())))));
  double dev_d = -2*arma::as_scalar(sum(ll1_d) - ll2_d);
  
  double DIC = 2*mean(dev0) - dev_d;
  
  mat biasB = zeros(p,d);
  mat MSEB = zeros(p,d);
  mat biasC = zeros(p,d);
  mat MSEC = zeros(p,d);
  for(int i=0; i<p; i++){
    for(int j=0; j<d; j++){
      vec Bij = Bh.tube(i,j);
      biasB(i,j)=abs(mean(Bij)-B(i,j));
      MSEB(i,j)=pow(mean(B(i,j)-mean(Bij)),2);
      vec Cij = Ch.tube(i,j);
      biasC(i,j)=abs(mean(Cij)-C(i,j));
      MSEC(i,j)=pow(mean(C(i,j)-mean(Cij)),2);
    }
  }

  return List::create( Named("MSE(B)")=mean(mean(MSEB)), 
                       Named("bias(B)")=mean(mean(biasB)),
                       Named("MSE(C)")=mean(mean(MSEC)), 
                       Named("bias(C)")=mean(mean(biasC)),
                       Named("DIC")=DIC);
}

// [[Rcpp::export]]
double DIC(mat Y, mat X, mat B, mat lambda, mat b, vec dev0){
  int n = Y.n_rows; int p = B.n_rows; int d = B.n_cols;
  vec ll1_d = zeros(n);
  for(int i=0; i<n; i++){ll1_d(i) = arma::as_scalar(Y.row(i)*(B.t()*X.row(i).t()+lambda*b.row(i).t()));}
  double ll2_d = arma::as_scalar(sum(sum(log(1+exp(X*B+b*lambda.t())))));
  double dev_d = -2*arma::as_scalar(sum(ll1_d) - ll2_d);
  double DIC = 2*mean(dev0) - dev_d;
  return DIC;
}

List Signal(cube B, double d0, double p0){
  int p = B.n_rows;
  int d = B.n_cols;
  mat prob = zeros(p,d);
  mat B1 = zeros(p,d);
  double slices = B.n_slices;
  
  for(int i=0; i<p; i++){
    for(int j=0; j<d; j++){
      vec b = B.tube(i,j);
      prob(i,j) = sum(exp(b)>d0)/slices;
      if(prob(i,j)>p0){B1(i,j)=1;}
    }
  }

  return List::create(Named("Prob")=prob, 
                      Named("Final")=B1) ;
}


