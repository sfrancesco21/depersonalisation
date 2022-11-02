#Functions for the simulation script. Mathematical justification can be found 
#in Bishop (2006).

#REFERENCES
#
#Bishop, C. M. (2006). Machine learning and pattern recognition. 
#Information science and statistics. Springer, Heidelberg.



#Responsibilities estimation for higher level hidden state (safety vs danger)
#
#Inputs:
#-s: current approximate posterior means over low level hidden states
#-phi: Fixed prior over higher level hidden states (safety vs danger)
#-mu: prior means over low level hidden states
#-sigma_phi: prior variances over low level hidden states
#-tau: current approximate posterior precisions over low level hidden states
#
#Outputs:
#-z: responsibilities for higher level hidden state (safety vs danger)
Estep_z = function(s,phi,mu,sigma_phi,tau) {
  rho = matrix(0,1,length(phi))
  for (k in 1:length(phi)){
    ln_rho = log(phi[k])-sum(((s-mu[k,])^2+(1/tau))/(2*sigma_phi))
    rho[k] = exp(ln_rho)
  }
  z = rho/sum(rho)
  return(z)
} 

#Responsibilities estimation for policies
#
#Inputs:
#-obs: observations
#-alpha: parameters of the Dirichlect prior over policies
#-s: current approximate posterior means over low level hidden states
#-sigma_s: prior variances over obs
#-tau: current approximate posterior precision over low level hidden states
#
#Outputs:
#-c: responsibilities for policies
Estep_c = function(obs,alpha,s,sigma_s,theta,tau){
  rho = matrix(0,1,length(alpha))
  for (m in 1:length(alpha)){
    ln_rho = digamma(alpha[m])-digamma(sum(alpha)) +
      sum(-0.5*log(sigma_s*theta[m,])- 
            (((obs-s)^2+(1/tau))/(2*theta[m,]*sigma_s)))
    rho[m] = exp(ln_rho)
  }
  c = rho/sum(rho)
  return(c)
}

#Gradient ascent function to maximise VFE
#
#Inputs:
#-obs: observations
#-s: current approximate posterior means over low level hidden states
#-theta: matrix with policy effects
#-sigma_s: prior variances over obs
#-sigma_phi: prior variances over low level hidden states
#-c: responsibilities for policies
#-z: responsibilities for higher level hidden state (safety vs danger)
#
#Outputs:
#-s: approximate posterior means over low level hidden states
#-tau: approximate posterior precisions over low level hidden states
gradient_ascent = function(obs,s,theta,mu,sigma_s,sigma_phi,c,z){
  diff = 11
  csize = length(c)
  zsize = length(z)
  tau_s = 1/sigma_s
  tau_phi = 1/sigma_phi
  old = matrix(0,1,2*length(s))
  while(max(abs(diff))!=0.0000){
    ds = c%*%(repmat(obs-s,csize,1)/(repmat(sigma_s,csize,1)*theta))- 
      z%*%(((repmat(s,zsize,1)-mu))/repmat(sigma_phi,zsize,1))
    d2s = -c%*%(repmat(tau_s,csize,1)/theta)- 
      z%*%(repmat(tau_phi,zsize,1))
    s = s-ds/d2s
    tau = -d2s
    diff = old-cbind(s,tau)
    diff = round(diff, digits=4)
    old = cbind(s,tau)
  }
  out = list(s,tau)
  return(out)
}

#Parameter update after responsibilities estimation
#
#Inputs:
#-obs: observations
#-s: current approximate posterior means over low level hidden states
#-z: responsibilities for higher level hidden state (safety vs danger)
#-c: responsibilities for policies
#-mu: prior means over low level hidden states
#-sigma_phi: prior variances over low level hidden states
#-sigma_s: prior variances over obs
#-theta: matrix with policy effects
#alpha_prior: parameters of the Dirichlect prior over policies
#
#Outputs:
#pi: Mixing components over policies
#alpha_post: parameters of the approximate Dirichlect posterior over policies
#-s: approximate posterior means over low level hidden states
#-tau: approximate posterior precisions over low level hidden states
Mstep = function(obs,s,z,c,mu,sigma_phi,sigma_s,theta,alpha_prior){
  alpha_post = alpha_prior+c
  pi = alpha_post/sum(alpha_post)
  out = gradient_ascent(obs,s,theta,mu,sigma_s,sigma_phi,c,z)
  s = out[[1]]
  tau = out[[2]]
  par = list(pi,alpha_post,s,tau)
  return(par)
}

#Habit formation function. 
#
#Inputs:
#-obs_train: repeated observations
#-mu: prior means over low level hidden states
#-sigma_phi: prior variances over low level hidden states
#-sigma_s: prior variances over obs
#-theta: matrix with policy effects
#-alpha_prior: parameters of the initial Dirichlect prior over policies
#
#Outputs:
#mixcom: Evolution of mixing components (pi) over time
training = function(obs_train,mu,sigma_phi,sigma_s,theta,alpha_prior){
  mixcom = matrix(0,dim(obs_train)[1],length(alpha_prior))
  pi = alpha_prior/sum(alpha_prior)
  tau = 1/sigma_s
  for (t in 1:dim(obs_train)[1]){
    mixcom[t,] = alpha_prior/sum(alpha_prior)
    s = matrix(20,1,dim(obs_train)[2])
    tau = 1/sigma_s
    old = matrix(12,1,28)
    diff = 99
    alpha_post = alpha_prior
    while (max(abs(diff)) != 0.000000){
      z = Estep_z(s,phi,mu,sigma_phi,tau)
      c = Estep_c(obs_train[t,],alpha_post,s,sigma_s,theta,tau)
      par = Mstep(obs_train[t,],s,z,c,mu,sigma_phi,sigma_s,theta,alpha_prior)
      pi = par[[1]]
      alpha_post = par[[2]]
      s = par[[3]]
      tau = par[[4]]
      new = cbind(s,z,c,tau,pi)
      diff = new-old
      diff = round(diff, digits = 6)
      old = new
    }
    alpha_prior = alpha_post
  }
  return(mixcom)
}

#Simulation of DPD episode
#
#Inputs:
#-obs_test: observations (time-series)
#-mu: prior means over low level hidden states
#-sigma_phi: prior variances over low level hidden states
#-sigma_s: prior variances over obs
#-theta: matrix with policy effects
#-alpha_prior: parameters of the Dirichlect prior over policies
#
#Outputs:
#-states_mu: approximate posterior means over low level hidden states (time-series)
#-action: responsibilities over policies (time-series) 
#-danger: responsibilities over high level hidden states (safety vs danger, time-series)
#-states_tau: approximate posterior precisions over low level hidden states (time-series)
episode = function(obs_test,mu,sigma_phi,sigma_s,theta,alpha_prior){
  states_mu = matrix(0,dim(obs_test)[1],dim(obs_test)[2])
  action = matrix(0,dim(obs_test)[1],length(alpha_prior))
  danger = matrix(0,dim(obs_test)[1],dim(mu)[1])
  states_tau = matrix(0,dim(obs_test)[1],dim(obs_test)[2])
  pi = alpha_prior/sum(alpha_prior)
  s = obs_test[1,]
  tau = 1/sigma_s
  for (t in 1:dim(obs_test)[1]){
    #s = obs_test[t,]
    old = matrix(12,1,28)
    diff = 99
    while (max(abs(diff)) != 0.000000){
      z = Estep_z(s,phi,mu,sigma_phi,tau)
      c = Estep_c(obs_test[t,],alpha_prior,s,sigma_s,theta,tau)
      par = Mstep(obs_test[t,],s,z,c,mu,sigma_phi,sigma_s,theta,alpha_prior)
      s = par[[3]]
      tau = par[[4]]
      new = cbind(s,z,c,tau,pi)
      diff = new-old
      diff = round(diff, digits = 6)
      old = new
    }
    states_mu[t,] = s
    action[t,] = c
    danger[t,] = z
    states_tau[t,] = tau
  }
  time_sequence = list(states_mu,action,danger,states_tau)
  return(time_sequence)
}