rm(list = ls())  #Clean up

# Get functions and load libraries
source("functions.R")
library(pracma)
library(mvtnorm)
library(tidyverse)

# Set variables
phi = matrix(c(0.99,0.01),1,2) #Preference about being safe vs in danger (FIXED)
alpha_prior = matrix(c(100,100,1),1,3) #Initial parameters of pi
pi = alpha_prior/sum(alpha_prior) #Habits (updatable Dirichlet distribution)
sigma_s = matrix(25,1,10) #Variance of the prior on observations
sigma_phi = matrix(100,1,10) #Variance of the prior on hidden states 
theta = matrix(1,3,10) #Initialise theta matrix (effect of policies)
theta[3,1:7] = 100 #Set the effect of dissociative policies
mu = t(cbind(matrix(20,10,1),matrix(80,10,1))) #Modes of the prior over hidden states
panic_stim = cbind(matrix(80,1,7),matrix(20,1,3)) #Observations 
obs_train = repmat(panic_stim,1000,1) #Repeated observations

#This simulates habit formation
mixcom = training(obs_train,mu,sigma_phi,sigma_s,theta,alpha_prior) 

mixcom = data.frame(cbind(mixcom[,1]+mixcom[,2],mixcom[,3]))

#Plot habit formation (evolution of pi)
ggplot(data = mixcom) + 
  geom_line(mapping = aes(x = 1:dim(mixcom)[1],y = X1, 
                          color = "Other policies"),size=1)+ 
  geom_line(mapping = aes(x = 1:dim(mixcom)[1],y = X2, 
                          color = "Dissociation"),size=1)+ 
  ggtitle("Habit evolution") + xlab("Number of DPD episodes") +
  ylab("Mixture component") + labs(color = "Policy")


#DPD episode simulation
int_test = c(4*5:20,matrix(80,1,50),80-0.25*(1:240),matrix(20,1,50)) #interoceptive observations
ext_test = matrix(20,length(int_test),3) #exteroceptive observations
obs_test = cbind(t(repmat(int_test,7,1)),ext_test)
#obs_test = t(repmat(int_test,10,1))

#This is the last episode to simulate 
max_ep = 50

#Initialise data matrices
duration = matrix(0,max_ep,1)
threshold = matrix(0,max_ep,1)
ep = 0:(max_ep-1)

#Loop that simulates DPD episodes with different priors over policies (pi).
#It shows that as one experiences more and more DPD episodes, the easier it 
#will be to trigger a new one
for (j in 1:max_ep){

n_past_episodes = j-1 

alpha_prior = matrix(c(100,100,n_past_episodes+1),1,3) #Set habits
time_sequence = episode(obs_test,mu,sigma_phi,sigma_s,theta,alpha_prior) #This simulated a DPD episode

#get posterior probabilities over dissociation policy
prob_dissociation = time_sequence[[2]][,3]
#Get the number of time points in which the probability of dissociating is >50%
duration[j] = sum(prob_dissociation > 0.5)
#Get the point in which the agent starts dissociating.
#We only consider the first 150 time points for simplicity, as data exploration
#revealed that dissociation always started before, and finished after, the 150th
#time point.
threshold_idx = sum(prob_dissociation[1:150]<0.5)+1
threshold[j] = obs_test[threshold_idx,1]
}

plot_data = data.frame(cbind(obs_test[,1],time_sequence[[1]][,1],time_sequence[[2]][,3], 
                  time_sequence[[4]][,1]))

#Here we adjust the values to be realistic heart rates. 
#Only these two fields of plot_data need to be actually plotted
#Both are time series (each row is a point in time)

m=140/60 #slope
q=60-(140/3) #intercept

plot_data$HR_obs = (plot_data[,1]*m)+q #Real HR (observations)
plot_data$HR_hs = (plot_data[,2]*m)+q #Inferred HR (inferred hidden state)
threshold_HR = (threshold*m)+q

plot2 = data.frame(cbind(threshold_HR, duration)) #dataframe for plotting duration and threshold


#Threshold plot
ggplot(data = plot2) + 
  geom_line(mapping = aes(x = ep ,y = X1),size=1)+
  ggtitle("Threshold for triggering dissociation as a function fo number of past episodes")+ xlab("Number of past episodes")+ 
  ylab("Heart Rate (Hz)")+ 
  labs(color = " ")+
  theme_classic() +
  theme(plot.title = element_text(hjust = 0.5), text = element_text(size = 16))


#DPD episode duration plot
ggplot(data = plot2) + 
  geom_line(mapping = aes(x = ep ,y = X2),size=1)+
  ggtitle("DPD episode duration as a function fo number of past episodes")+ xlab("Number of past episodes")+ 
  ylab("Time (a.u.)")+ 
  labs(color = " ")+
  theme_classic() +
  theme(plot.title = element_text(hjust = 0.5), text = element_text(size = 16))


#Episode plot (whatever episode was last simulated)
ggplot(data = plot_data) + 
  geom_line(mapping = aes(x = 1:dim(plot_data)[1],y = HR_obs, 
                          color = "Observations"),size=1)+ 
  geom_line(mapping = aes(x = 1:dim(plot_data)[1],y = HR_hs, 
                          color = "Inferred hidden states"),size=1)+ 
  ggtitle("Dissociation episode")+ xlab("Time")+ 
  ylab("Heart rate (Hz)")+ 
  labs(color = " ")+
  theme_classic() +
  theme(plot.title = element_text(hjust = 0.5), text = element_text(size = 16))

