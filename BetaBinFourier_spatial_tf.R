rm(list=ls())

library(tensorflow)
library(PrevMap)
library(mvtnorm)

sess <- tf$Session()
data("loaloa")

### Global variables
beta0 <- -2.47388
sigma2.0 <- exp(0.277666)
delta0 <- exp(-2.31985)
phi0 <- exp(0.00277671)

n <- nrow(loaloa)
m <- 100L

X <- as.matrix(loaloa[,c("LONGITUDE","LATITUDE")])

tf_y <- tf$placeholder(tf$float32,shape = c(n,1L))
tf_units.m <- tf$placeholder(tf$float32,shape = c(n,1L))

Z.halton <- rmvt(m,df=0.5)
quad.points <- rmvt(m,df=0.5)
Omega0 <- quad.points/phi0
tf_F.mat0 <- tf$placeholder(tf$float32,shape = c(n,m))

n <- nrow(loaloa)
D <- cbind(rep(1,n))
Omega0 <- quad.points/phi0

tf_mu0 <- tf$placeholder(tf$float32,shape = c(n,1L))
tf_y <- tf$placeholder(tf$float32,shape = c(n,1L))

### Variables
tf_Z <- tf$placeholder(tf$float32,shape = c(m,1L))

### Operations 

tf_llik <- function(y,units.m,prob,delta0) {
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
}

integrand <- function (Z,y,units.m,mu0,F.mat0,delta0) {
  eta <- mu0+F.mat0%*%Z
  prob <- 1/(1+exp(-eta))
  llik <- tf_llik(y,units.m,prob,delta0)
  -0.5*sum(Z^2)/sigma2.0+sum(llik)
}

# Computation
obj <- -integrand(tf_Z,tf_y,tf_units.m,tf_mu0,tf_F.mat0)
train_op = tf$train$AdamOptimizer(0.5)$minimize(obj)

# Initialise global variables
feed_dict = dict(tf_y = loaloa$NO_INF,
                 tf_mu0 = as.numeric(D%*%beta0),
                 tf_F.mat0 = cbind(cos(X%*%t(Omega0)),sin(X%*%t(Omega0)))/sqrt(m),
                 tf_units.m = loaloa$NO_EXAM)
# Optimization

sess$run(tf$global_variables_initializer())
old_loss = 10e10
loss = 0
tol = 1e-6
step = 0
while (abs(old_loss - loss)/abs(old_loss) > tol) {
  step = step + 1
  old_loss = loss
  out = sess$run(c(train_op, obj), feed_dict)
  loss = out[[2]]
  cat(step, sess$run(obj, feed_dict),":", 
      sess$run(tf_beta), sess$run(tf_sigma2),
      sess$run(tf_phi), sess$run(tf_nu2), "\n")
}


F.star <- compute_F.mat(tf_phi,tf_grid.pred,tf_Z.halton)
F.hat <- compute_F.mat(tf_phi,tf_X,tf_Z.halton)
C <- tf_sigma2*tf$matmul(F.star,tf$transpose(F.hat))

m <- tf$cast(tf$shape(F.hat)[2], tf$float32)
A <- tf$matmul(tf$transpose(F.hat),F.hat)/tf_nu2+
  tf$eye(m,dtype=tf$float32)
L <- tf$cholesky(A)

tf_mu_pred <- tf$matmul(tf_D_pred,tf_beta)
tf_mu <- tf$matmul(tf_D,tf_beta)
y.trans <- tf$matmul(tf$transpose(F.hat),tf_y-tf_mu)
z <- tf$matrix_triangular_solve(L, y.trans, lower = TRUE)
z <- tf$matrix_triangular_solve(
  tf$transpose(L), z, lower = FALSE)

z.pred <- (tf_y-tf_mu)/(tf_nu2*tf_sigma2)-tf$matmul(F.hat,z/(tf$square(tf_nu2)*tf_sigma2)
)
tf_logit.pred <- tf_mu_pred+tf$matmul(C,z.pred)

library(raster)
r <- rasterFromXYZ(
  cbind(grid.pred,sess$run(
    tf$cast(1,tf$float32)/
      (tf$cast(1,tf$float32)+tf$exp(-tf_logit.pred)),feed_dict)))
plot(r)
