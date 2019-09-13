rm(list=ls())
library(PrevMap)
library(mvtnorm)
library(numDeriv)
data(loaloa)

y <- loaloa$NO_INF
units.m <- loaloa$NO_EXAM

X <- as.matrix(loaloa[,c("LONGITUDE","LATITUDE")])
x=X
#x=scale(X)
z=y
zn=units.m
n=nrow(x)

K=1000 # Number of features
master_omega<-rmvt(K,df=1.5)
#length scale   0.9407122  intercept   -2.556573  delta   12.67649  lambda   2.567847  

p <- 1L # Number of covariates

beta0 <-  -0.429973
sigma2.0 <- exp(0.807347)# exp(0.277666) # mixed model with iid noise, compute emirical variogram, sigma^2 y = beta0 + idd
theta0 <-   exp(-0.531703) #
phi0 <- exp(-1.51897) # emirical variogram

library(tensorflow)
library(tidyr)
tf$reset_default_graph()
x_= tf$placeholder(shape=shape(n,2L),tf$float32) # 
D = tf$placeholder(shape=shape(NULL,p),tf$float32)
y = tf$placeholder(shape=shape(n,1L),tf$float32,name='testing_x') # response
yn = tf$placeholder(shape=shape(n,1L),tf$float32,name='testing_x') # response


phi0.tf = tf$constant(phi0,tf$float32) # emirical variogram
theta0.tf=tf$constant(theta0,dtype=tf$float32) 		   
beta0.tf <- tf$constant(beta0,shape=c(p,1L),dtype=tf$float32) 
sigma2.0.tf = tf$constant(sigma2.0,dtype=tf$float32)

omega <- tf$constant(master_omega,dtype=tf$float32) # ridge regression coefficients covariates
omega=tf$divide(omega,phi0.tf)
projected = tf$matmul(x_,tf$transpose(omega))
cosp<-tf$cos(projected)
sinp<-tf$sin(projected)
scale = tf$constant(sqrt(1/K),dtype=tf$float32,name='scale') # constant scale parameter for RFF
PHI.mat<- tf$multiply(tf$concat(list(sinp,cosp),1L),scale)
Z <- tf$Variable(tf$random_normal(shape(2*K,1L),dtype=tf$float32,stddev=0.1,mean=0)) 

mu0 <- tf$matmul(D,beta0.tf)

prob = tf$sigmoid(mu0+tf$matmul(PHI.mat,Z))

a.beta <- tf$divide(prob,theta0.tf)
b.beta <- tf$divide(tf$subtract(tf$constant(1.0,dtype=tf$float32),prob),theta0.tf);
tf_lbeta <- function(a, b) { 
  tf$lgamma(a) + tf$lgamma(b) - tf$lgamma(a + b) 
}
betabinomial = tf_lbeta(y + a.beta, yn - y + b.beta) - tf_lbeta(a.beta, b.beta)

mae=tf$reduce_mean(tf$abs((y/yn)-prob))
likelihood= tf$reduce_sum(betabinomial,0L) - tf$constant(0.5,dtype=tf$float32)*tf$reduce_sum(tf$square(Z),0L)/sigma2.0.tf

hessian = tf$squeeze(tf$hessians(tf$negative(likelihood),Z))
Sigma_tilde = tf$linalg$inv(hessian)
Sigma_sroot = tf$transpose(tf$linalg$cholesky(Sigma_tilde))
A = tf$linalg$inv(Sigma_sroot)
Sigma_W_inv <- tf$matmul(tf$transpose(Sigma_sroot),Sigma_sroot)/sigma2.0.tf
mu_W <- -tf$matmul(A,Z) 

train_step = tf$train$AdamOptimizer(0.001)$minimize(tf$negative(likelihood))							 

myconfig=tf$ConfigProto(log_device_placement=FALSE)
sess = tf$Session(config=myconfig)
init = tf$global_variables_initializer()
sess$run(init) # initialise session


epochs=4000
batch_size=8
library(cvTools)
store=rep(NA,epochs)
for(i in 1:epochs){
  #		wh<-cvFolds(n,K=round(n/batch_size),type='random') # cross validation folds
  #		for(j in 1:round(n/batch_size)){
  #			value<-sess$run(list(train_step), feed_dict=dict(x_=x[wh$which==j,],y=cbind(z[wh$which==j]),yn=cbind(zn[wh$which==j])))
  #		}
  value<-sess$run(list(train_step), feed_dict=dict(x_=x,y=cbind(z),yn=cbind(zn),
                                                   D=cbind(rep(1,n))))
  value<-sess$run(list(mae,likelihood), feed_dict=dict(x_=x,y=cbind(z),yn=cbind(zn),
                                                       D=cbind(rep(1,n))))
  store[i]=value[[2]]
  if(i %% 1e1 == 0){
    cat("Epoch",i," MAE ",value[[1]]," With likelihood", value[[2]],"\r")
  }
}

par(mfrow=c(1,2))	
pred<-sess$run(prob, feed_dict=dict(x_=x,y=cbind(z),yn=cbind(zn),
                                    D=cbind(rep(1,n))))
#################################################################################################################################
# objects needed for sampling and later on Z, Sigma_W_inv, Sigma_sroot, mu_W, omega,delta,intercept
object1 = sess$run(Z, feed_dict=dict(x_=x,y=cbind(z),yn=cbind(zn)))
object2 = sess$run(Sigma_W_inv, feed_dict=dict(x_=x,y=cbind(z),yn=cbind(zn),
                                               D=cbind(rep(1,n))))
object3 = sess$run(Sigma_sroot, feed_dict=dict(x_=x,y=cbind(z),yn=cbind(zn),
                                               D=cbind(rep(1,n))))
object4 = sess$run(mu_W, feed_dict=dict(x_=x,y=cbind(z),yn=cbind(zn),
                                        D=cbind(rep(1,n))))
object5 = sess$run(PHI.mat, feed_dict=dict(x_=x,y=cbind(z),yn=cbind(zn)))

tf$reset_default_graph()
tf_lbeta <- function(a, b){ tf$lgamma(a) + tf$lgamma(b) - tf$lgamma(a + b) }
D=tf$placeholder(shape=shape(NULL,p),tf$float32)
x_= tf$placeholder(shape=shape(n,2L),tf$float32) # covariates
y = tf$placeholder(shape=shape(n,1L),tf$float32,name='testing_x') # response
yn = tf$placeholder(shape=shape(n,1L),tf$float32,name='testing_x') # response
W = tf$placeholder(shape=shape(2*K,1L),tf$float32,name='testing_x') # response
omega <- tf$constant(master_omega,dtype=tf$float32) # ridge regression coefficients covariates

Z0=tf$constant(object1,dtype=tf$float32)
Sigma_W_inv=tf$constant(object2,dtype=tf$float32)
Sigma_sroot=tf$constant(object3,dtype=tf$float32)
mu_W=tf$constant(object4,dtype=tf$float32)
PHI.mat=tf$constant(object5,dtype=tf$float32)

phi0.tf = tf$constant(phi0,tf$float32) # emirical variogram
theta0.tf=tf$constant(theta0,dtype=tf$float32) 		   
beta0.tf <- tf$constant(beta0,shape=c(p,1L),dtype=tf$float32) 
sigma2.0.tf = tf$constant(sigma2.0,dtype=tf$float32)
mu0=tf$matmul(D,beta0.tf)

Z = tf$matmul(Sigma_sroot,W) + Z0	
diff_w <- W-mu_W
prob <- tf$sigmoid(mu0+tf$matmul(PHI.mat,Z))
a.beta <- tf$divide(prob,theta0.tf)
b.beta <- tf$divide(tf$subtract(tf$constant(1.0,dtype=tf$float32),prob),theta0)
betabinomial= tf_lbeta(y + a.beta, yn - y + b.beta) - tf_lbeta(a.beta, b.beta)
log_prob = tf$constant(-0.5,dtype=tf$float32) * tf$matmul(tf$transpose(diff_w),tf$matmul(Sigma_W_inv,diff_w)) +  tf$reduce_sum(betabinomial)
log_prob_grad = tf$gradients(log_prob,W)[[1]]
results = list(log_prob,log_prob_grad)

sess = tf$Session()
init = tf$global_variables_initializer()
sess$run(init) # initialise session



h <- 1.65/((2*K)^(1/6))
n.sim <- 12000
burnin <- 2000
thin <- 20
c1.h <- 0.001
c2.h <- 0.0001
W.curr <- rep(0,2*K)

initial = sess$run(results, feed_dict=dict(x_=x,y=cbind(z),yn=cbind(zn),
                                           D=cbind(rep(1,n)),W=cbind(W.curr)))
mean.curr <- as.numeric(W.curr + (h^2/2)*initial[[2]])
lp.curr <- initial[[1]]
acc <- 0
sim <- matrix(NA,nrow=(n.sim-burnin)/thin,ncol=2*K)
h.vec <- rep(NA,n.sim)
for(i in 1:n.sim) {
  W.prop <- mean.curr+h*rnorm(2*K)
  calc = sess$run(results, feed_dict=dict(x_=x,y=cbind(z),yn=cbind(zn),
                                          D=cbind(rep(1,n)),W=cbind(W.prop)))
  mean.prop <- as.numeric(W.prop + (h^2/2)*calc[[2]])
  lp.prop <- calc[[1]]
  
  dprop.curr <- -sum((W.prop-mean.curr)^2)/(2*(h^2))
  dprop.prop <- -sum((W.curr-mean.prop)^2)/(2*(h^2))
  
  log.prob <- lp.prop+dprop.prop-lp.curr-dprop.curr
  
  if(log(runif(1)) < log.prob) {
    acc <- acc+1
    W.curr <- W.prop
    lp.curr <- lp.prop
    mean.curr <- mean.prop
  }
  
  if( i > burnin & (i-burnin)%%thin==0) {
    sim[(i-burnin)/thin,] <- sess$run(Z, feed_dict=dict(x_=x,y=cbind(z),yn=cbind(zn),W=cbind(W.prop)))
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

####
log.integrand <- function(Z,val,PHI.mat) {
  eta <- as.numeric(val$mu+PHI.mat%*%Z)
  
  prob <- 1/(1+exp(-eta))
  
  llik <- rep(NA,n)
  
  for(i in 1:n) {
    if(y[i]==0) {
      llik[i] <- sum(log(1-prob[i]+(0:(units.m[i]-1))*val$theta))-
        sum(log(1+(0:(units.m[i]-1))*val$theta))
    } else if(y[i]==units.m[i]) {
      llik[i] <- sum(log(prob[i]+(0:(y[i]-1))*val$theta))-
        sum(log(1+(0:(units.m[i]-1))*val$theta))
    } else {
      llik[i] <- sum(log(prob[i]+(0:(y[i]-1))*val$theta))+
        sum(log(1-prob[i]+(0:(units.m[i]-y[i]-1))*val$theta))-
        sum(log(1+(0:(units.m[i]-1))*val$theta))
    }
  }  
  
  -0.5*((2*K)*log(val$sigma2)+sum(Z^2)/val$sigma2)+
    sum(llik)
}

compute.log.f <- function(sub.par,PHI.mat) {
  beta <- sub.par[1:p];
  val <- list()
  val$sigma2 <- exp(sub.par[p+1])
  val$mu <- as.numeric(D%*%beta)
  val$theta <- exp(sub.par[p+2])
  sapply(1:(dim(sim)[1]),function(i) log.integrand(sim[i,],val,PHI.mat))
}

D <- cbind(rep(1,n))
p <- ncol(D)
units.m <- loaloa$NO_EXAM
y <- loaloa$NO_INF

quad.points <- rmvt(K,df=1.5)
Omega0 <- quad.points/phi0
PHI.mat0 <- cbind(cos(X%*%t(Omega0)),sin(X%*%t(Omega0)))/sqrt(K)

log.f.tilde <- compute.log.f(c(beta0,log(sigma2.0),log(theta0)),PHI.mat0)
par0 <- c(c(beta0,log(sigma2.0),log(theta0)),log(phi0))

MC.log.lik <- function(par) {
  phi <- exp(par[p+3])
  sub.par <- par[1:(p+2)]
  Omega <- quad.points/phi
  PHI.mat <- cbind(cos(X%*%t(Omega)),sin(X%*%t(Omega)))/sqrt(K)
  log(mean(exp(compute.log.f(sub.par,PHI.mat)-log.f.tilde)))
}

estim <- 
  nlminb(par0,
         function(x) -MC.log.lik(x),
         control=list(trace=1))

library(splancs)
poly <- X[chull(X),]
grid.pred <- gridpts(poly, xs = 0.05, ys = 0.05)

lp = sess$run(prediction, feed_dict=dict(x_=as.matrix(loaloa[,c("LONGITUDE","LATITUDE")]),
                                         y=cbind(z),yn=cbind(zn),
                                         D=cbind(rep(1,n))))
r <- rasterFromXYZ(cbind(grid.pred,lp))
par(mfrow=c(1,1),mar=c(2,2,2,2))
plot(r)

