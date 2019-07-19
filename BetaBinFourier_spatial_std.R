rm(list=ls())
library(PrevMap)
library(mvtnorm)
library(numDeriv)
data(loaloa)

y <- loaloa$NO_INF
units.m <- loaloa$NO_EXAM

beta0 <- -2.47388
sigma2.0 <- exp(0.277666)
delta0 <- exp(-2.31985)
phi0 <- exp(0.00277671)

X <- as.matrix(loaloa[,c("LONGITUDE","LATITUDE")])
m <- 500
quad.points <- rmvt(m,df=0.5)
Omega0 <- quad.points/phi0
F.mat0 <- cbind(cos(X%*%t(Omega0)),sin(X%*%t(Omega0)))/sqrt(m)

n <- nrow(loaloa)
D <- cbind(rep(1,n))
mu0 <- as.numeric(D%*%beta0)

integrand <- function(Z) {
  eta <- as.numeric(mu0+F.mat0%*%Z)
  prob <- 1/(1+exp(-eta))

  llik <- rep(NA,n)

  for(i in 1:n) {
    if(y[i]==0) {
      llik[i] <- sum(log(1-prob[i]+(0:(units.m[i]-1))*delta0))-
                 sum(log(1+(0:(units.m[i]-1))*delta0))
    } else if(y[i]==units.m[i]) {
      llik[i] <- sum(log(prob[i]+(0:(y[i]-1))*delta0))-
        sum(log(1+(0:(units.m[i]-1))*delta0))
    } else {
      llik[i] <- sum(log(prob[i]+(0:(y[i]-1))*delta0))+
                 sum(log(1-prob[i]+(0:(units.m[i]-y[i]-1))*delta0))-
                 sum(log(1+(0:(units.m[i]-1))*delta0))
    }
  }  
  -0.5*sum(Z^2)/sigma2.0+sum(llik)
}

grad.integrand <- function(Z) {
  eta <- as.numeric(mu0+F.mat0%*%Z)
  prob <- 1/(1+exp(-eta))

  grad.llik <- rep(NA,n)
  
  for(i in 1:n) {
    if(y[i]==0) {
      grad.llik[i] <- -sum(1/(1-prob[i]+(0:(units.m[i]-1))*delta0))
    } else if(y[i]==units.m[i]) {
      grad.llik[i] <- sum(1/(prob[i]+(0:(y[i]-1))*delta0))
    } else {
      grad.llik[i] <- sum(1/(prob[i]+(0:(y[i]-1))*delta0))-
        sum(1/(1-prob[i]+(0:(units.m[i]-y[i]-1))*delta0))
    }
  }
  grad.llik <- grad.llik*prob/(1+exp(eta))
  
  out <- as.numeric(-Z/sigma2.0+t(F.mat0)%*%grad.llik)
  return(out)
}

hessian.integrand <- function(Z) {
  eta <- as.numeric(mu0+F.mat0%*%Z)
  prob <- 1/(1+exp(-eta))
  
  grad.llik <- rep(NA,n)
  
  for(i in 1:n) {
    if(y[i]==0) {
      grad.llik[i] <- -sum(1/(1-prob[i]+(0:(units.m[i]-1))*delta0))
    } else if(y[i]==units.m[i]) {
      grad.llik[i] <- sum(1/(prob[i]+(0:(y[i]-1))*delta0))
    } else {
      grad.llik[i] <- sum(1/(prob[i]+(0:(y[i]-1))*delta0))-
        sum(1/(1-prob[i]+(0:(units.m[i]-y[i]-1))*delta0))
    }
  }
  
  hess.llik <- rep(NA,)
  
  for(i in 1:n) {
    if(y[i]==0) {
      hess.llik[i] <- -sum(1/(1-prob[i]+(0:(units.m[i]-1))*delta0)^2)
    } else if(y[i]==units.m[i]) {
      hess.llik[i] <- -sum(1/(prob[i]+(0:(y[i]-1))*delta0)^2)
    } else {
      hess.llik[i] <- -sum(1/(prob[i]+(0:(y[i]-1))*delta0)^2)+
        -sum(1/(1-prob[i]+(0:(units.m[i]-y[i]-1))*delta0)^2)
    }
  }
  hess.llik <- grad.llik* (-((exp(eta)-1)*exp(eta))/(1+exp(eta))^3)+
               hess.llik*(prob/(1+exp(eta)))^2
  
 
  
  out <- t(F.mat0)%*%(F.mat0*hess.llik)
  diag(out) <- diag(out)+-1/sigma2.0
  return(out)
}

estim.int <- 
nlminb(rep(0,2*m),
       function(Z) -integrand(Z),
       function(Z) -grad.integrand(Z),
       function(Z) -hessian.integrand(Z),
       control=list(trace=1))

H <- hessian.integrand(estim.int$par)

Sigma.tilde <- solve(-H)
Sigma.sroot <- t(chol(Sigma.tilde))
A <- solve(Sigma.sroot)
Sigma.W.inv <- solve(sigma2.0*A%*%t(A))
mu.W <- -as.numeric(A%*%estim.int$par)


cond.dens.W <- function(W,Z) {
  diff.w <- W-mu.W
  eta <- as.numeric(mu0+F.mat0%*%Z)
  prob <- 1/(1+exp(-eta))
  
  llik <- rep(NA,n)
  
  for(i in 1:n) {
    if(y[i]==0) {
      llik[i] <- sum(log(1-prob[i]+(0:(units.m[i]-1))*delta0))-
        sum(log(1+(0:(units.m[i]-1))*delta0))
    } else if(y[i]==units.m[i]) {
      llik[i] <- sum(log(prob[i]+(0:(y[i]-1))*delta0))-
        sum(log(1+(0:(units.m[i]-1))*delta0))
    } else {
      llik[i] <- sum(log(prob[i]+(0:(y[i]-1))*delta0))+
        sum(log(1-prob[i]+(0:(units.m[i]-y[i]-1))*delta0))-
        sum(log(1+(0:(units.m[i]-1))*delta0))
    }
  }  
  -0.5*as.numeric(t(diff.w)%*%Sigma.W.inv%*%diff.w)+
   sum(llik)
}

lang.grad <- function(W,Z) {
  diff.w <- W-mu.W
  eta <- as.numeric(mu0+F.mat0%*%Z)
  prob <- 1/(1+exp(-eta))
  
  grad.llik <- rep(NA,n)
  
  for(i in 1:n) {
    if(y[i]==0) {
      grad.llik[i] <- -sum(1/(1-prob[i]+(0:(units.m[i]-1))*delta0))
    } else if(y[i]==units.m[i]) {
      grad.llik[i] <- sum(1/(prob[i]+(0:(y[i]-1))*delta0))
    } else {
      grad.llik[i] <- sum(1/(prob[i]+(0:(y[i]-1))*delta0))-
        sum(1/(1-prob[i]+(0:(units.m[i]-y[i]-1))*delta0))
    }
  }
  grad.llik <- grad.llik*prob/(1+exp(eta))
  
  grad.z <- as.numeric(t(F.mat0)%*%grad.llik)
  
  as.numeric(-Sigma.W.inv%*%(W-mu.W)+
               t(Sigma.sroot)%*%c(grad.z))
}

h <- 1.65/((2*m)^(1/6))
n.sim <- 10000
burnin <- 2000
thin <- 8
c1.h <- 0.001
c2.h <- 0.0001
W.curr <- rep(0,2*m)
Z.curr <- as.numeric(Sigma.sroot%*%W.curr+estim.int$par)
mean.curr <- as.numeric(W.curr + (h^2/2)*lang.grad(W.curr,Z.curr))
lp.curr <- cond.dens.W(W.curr,Z.curr)
acc <- 0
sim <- matrix(NA,nrow=(n.sim-burnin)/thin,ncol=2*m)
h.vec <- rep(NA,n.sim)
for(i in 1:n.sim) {
  W.prop <- mean.curr+h*rnorm(2*m)
  Z.prop <-  as.numeric(Sigma.sroot%*%W.prop+estim.int$par)
  mean.prop <- as.numeric(W.prop + (h^2/2)*lang.grad(W.prop,Z.prop))
  lp.prop <- cond.dens.W(W.prop,Z.prop)
  
  dprop.curr <- -sum((W.prop-mean.curr)^2)/(2*(h^2))
  dprop.prop <- -sum((W.curr-mean.prop)^2)/(2*(h^2))
  
  log.prob <- lp.prop+dprop.prop-lp.curr-dprop.curr
  
  if(log(runif(1)) < log.prob) {
    acc <- acc+1
    W.curr <- W.prop
    Z.curr <- Z.prop
    lp.curr <- lp.prop
    mean.curr <- mean.prop
  }
  
  if( i > burnin & (i-burnin)%%thin==0) {
    sim[(i-burnin)/thin,] <- Z.curr
  }
  
  h.vec[i] <- h <- max(0,h + c1.h*i^(-c2.h)*(acc/i-0.57))
  cat("Iteration",i,"out of",n.sim,"\r")
}

plot(h.vec,type="l")

acf.plot <- acf(sim[,1],plot=FALSE)
plot(acf.plot$lag,acf.plot$acf,type="l",xlab="lag",ylab="autocorrelation",
     ylim=c(-0.1,1),main="Autocorrelogram of the simulated samples")
for(i in 2:ncol(sim)) {
  acf.plot <- acf(sim[,i],plot=FALSE)
  lines(acf.plot$lag,acf.plot$acf)
}
abline(h=0,lty="dashed",col=2)


log.integrand <- function(Z,val,F.mat) {
  eta <- as.numeric(val$mu+F.mat%*%Z)
  
  prob <- 1/(1+exp(-eta))
  
  llik <- rep(NA,n)
  
  for(i in 1:n) {
    if(y[i]==0) {
      llik[i] <- sum(log(1-prob[i]+(0:(units.m[i]-1))*val$delta))-
        sum(log(1+(0:(units.m[i]-1))*val$delta))
    } else if(y[i]==units.m[i]) {
      llik[i] <- sum(log(prob[i]+(0:(y[i]-1))*val$delta))-
        sum(log(1+(0:(units.m[i]-1))*val$delta))
    } else {
      llik[i] <- sum(log(prob[i]+(0:(y[i]-1))*val$delta))+
        sum(log(1-prob[i]+(0:(units.m[i]-y[i]-1))*val$delta))-
        sum(log(1+(0:(units.m[i]-1))*val$delta))
    }
  }  
  
  -0.5*((2*m)*log(val$sigma2)+sum(Z^2)/val$sigma2)+
  sum(llik)
}

compute.log.f <- function(sub.par,F.mat) {
  beta <- sub.par[1:p];
  val <- list()
  val$sigma2 <- exp(sub.par[p+1])
  val$mu <- as.numeric(D%*%beta)
  val$delta <- exp(sub.par[p+2])
  sapply(1:(dim(sim)[1]),function(i) log.integrand(sim[i,],val,F.mat))
}

p <- ncol(D)
log.f.tilde <- compute.log.f(c(beta0,log(sigma2.0),log(delta0)),F.mat0)
par0 <- c(c(beta0,log(sigma2.0),log(delta0)),log(phi0))

MC.log.lik <- function(par) {
  phi <- exp(par[p+3])
  sub.par <- par[1:(p+2)]
  Omega <- quad.points/phi
  F.mat <- cbind(cos(X%*%t(Omega)),sin(X%*%t(Omega)))/sqrt(m)
  log(mean(exp(compute.log.f(sub.par,F.mat)-log.f.tilde)))
}

estim <- 
nlminb(par0,
       function(x) -MC.log.lik(x),
       control=list(trace=1))

library(splancs)
poly <- X[chull(X),]
grid.pred <- gridpts(poly, xs = 0.1, ys = 0.1)

n.pred <- nrow(grid.pred)
D.pred <- cbind(1,rep(n.pred))
phi.hat <- exp(estim$par[p+3])
mu.pred.hat <- as.numeric(D.pred%*%estim$par[1:p])
Omega <- quad.points/phi.hat
F.mat.pred <- cbind(cos(grid.pred%*%t(Omega)),
                    sin(grid.pred%*%t(Omega)))
n.samples <- (n.sim-burnin)/thin
eta.samples <- sapply(1:n.samples,function(i) mu.pred.hat+F.mat.pred%*%sim[i,])
prev.samples <- 1/(1+exp(-eta.samples))

prev.hat <- apply(prev.samples,1,mean)

r <- rasterFromXYZ(cbind(grid.pred,prev.hat))

plot(r)
