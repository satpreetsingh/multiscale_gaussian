#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 01:02:27 2018

@author: Josue Nassar
"""

import torch
from torch import FloatTensor
from torch.autograd import Variable
import numpy as np
import scipy.io as sio
import gradients as grad
import pickle

#Load in synthetic states from matlab
mat_contents=sio.loadmat('synthetic_states.mat')
states=np.matrix(mat_contents['x'])
T=states[1,:].size #Number of data points
dim=states[:,1].size #Dimension of latent state

mat_contents=sio.loadmat('Theta.mat')
theta=mat_contents['Theta']
theta=theta.ravel()
theta=theta.reshape(2,3,6)

mat_contents=sio.loadmat('Cov.mat')
Cov=mat_contents['Cov']
Cov=Cov.ravel()
Cov=Cov.reshape(2,2,6)

mat_contents=sio.loadmat('mu.mat')
true_mu=mat_contents['mu']
true_mu=true_mu.ravel()
true_mu=true_mu.reshape(2,6)
Q=np.matrix(0.5*np.eye(dim))

M=30 #Number of samples drawn from N(0,I) at each iteration
max_iter=1 #Maximum number of iterations SGVB will run for 

#Parameters of ADAM
a=0.2
beta1=0.9
beta2=0.999

#Set up prior for parents
mu_prior=np.matrix(np.zeros(dim)).T
Cov_prior=np.matrix(10*np.eye(dim))

NumParents=2 #Number of parents
NumKids=4 #Number of kids. Each parents has 2 kids

#Inital estimate of location and scale parameter of variational distribution of parents
alpha_est=[]
alpha_est.append(np.random.multivariate_normal([0,0],Cov_prior,1).T)
alpha_est.append(np.random.multivariate_normal([0,0],Cov_prior,1).T)
alpha_est.append(np.random.multivariate_normal(alpha_est[0].ravel(),Cov_prior,1).T)
alpha_est.append(np.random.multivariate_normal(alpha_est[0].ravel(),Cov_prior,1).T)
alpha_est.append(np.random.multivariate_normal(alpha_est[1].ravel(),Cov_prior,1).T)
alpha_est.append(np.random.multivariate_normal(alpha_est[1].ravel(),Cov_prior,1).T)
alpha_est=np.matrix(np.asarray(alpha_est)).T

sigma_est=[]
sigma_est.append(np.sqrt(5))
sigma_est.append(np.sqrt(5))
sigma_est.append(np.sqrt(2))
sigma_est.append(np.sqrt(2))
sigma_est.append(np.sqrt(2))
sigma_est.append(np.sqrt(2))
sigma_est=np.matrix(np.asarray(sigma_est))

m_alpha=np.zeros((2,6))
m_sigma=np.zeros(6)
v_alpha=np.zeros((2,6))
v_sigma=np.zeros(6)

iter=0
while iter<max_iter: # TODO: write this as a for loop! (you can break)
    iter+=1
    print(iter)
    #Draw M samples from N(0,I)
    e=np.random.randn(dim,M,6)
    
    #Create list to store estimates of gradient of variational distribution wrt alpha and R
    gradients_alpha1=[]
    gradients_alpha2=[]
    gradients_alpha3=[]
    gradients_alpha4=[]
    gradients_alpha5=[]
    gradients_alpha6=[]
    
    gradients_s1=[]
    gradients_s2=[]
    gradients_s3=[]
    gradients_s4=[]
    gradients_s5=[]
    gradients_s6=[]
    
    for m in range(0,M):
        alpha0=np.matrix(alpha_est[:,0])
        alpha1=np.matrix(alpha_est[:,1])
        alpha2=np.matrix(alpha_est[:,2])
        alpha3=np.matrix(alpha_est[:,3])
        alpha4=np.matrix(alpha_est[:,4])
        alpha5=np.matrix(alpha_est[:,5])
        
        sigma0=np.matrix(sigma_est[:,0])
        sigma1=np.matrix(sigma_est[:,1])
        sigma2=np.matrix(sigma_est[:,2])
        sigma3=np.matrix(sigma_est[:,3])
        sigma4=np.matrix(sigma_est[:,4])
        sigma5=np.matrix(sigma_est[:,5])
        
        e0=np.matrix(e[:,m,0]).T
        e1=np.matrix(e[:,m,1]).T
        e2=np.matrix(e[:,m,2]).T
        e3=np.matrix(e[:,m,3]).T
        e4=np.matrix(e[:,m,4]).T
        e5=np.matrix(e[:,m,5]).T
        
        #Compute gradients from prior
        gp1,sp1,gk1,sk1,gk2,sk2=grad.prior_gradient(alpha0,sigma0,e0,alpha2,sigma2,e2,alpha3,sigma3,e3,mu_prior,Cov_prior)
        gp2,sp2,gk3,sk3,gk4,sk4=grad.prior_gradient(alpha1,sigma1,e1,alpha4,sigma4,e4,alpha5,sigma5,e5,mu_prior,Cov_prior)
        
        #Compute gradients from variational distribution
        var_a_p1,var_s_p1=grad.variational_gradient(alpha0,sigma0,e0)
        var_a_p2,var_s_p2=grad.variational_gradient(alpha1,sigma1,e1)
        var_a_k1,var_s_k1=grad.variational_gradient(alpha2,sigma2,e2)
        var_a_k2,var_s_k2=grad.variational_gradient(alpha3,sigma3,e3)
        var_a_k3,var_s_k3=grad.variational_gradient(alpha4,sigma4,e4)
        var_a_k4,var_s_k4=grad.variational_gradient(alpha5,sigma5,e5)
        
        like_a0=0
        like_a1=0
        like_a2=0
        like_a3=0
        like_a4=0
        like_a5=0
        
        like_s0=0
        like_s1=0
        like_s2=0
        like_s3=0
        like_s4=0
        like_s5=0
        
        for t in range(0,T-1):
            x_prev=np.matrix(states[:,t])
            x_curr=np.matrix(states[:,t+1])
            u=x_prev
            u=np.concatenate((u,[[1]]),axis=0)
            #Compute gradients from Likelihood
            la1,la2,la3,la4,la5,la6,ls1,ls2,ls3,ls4,ls5,ls6=grad.like_gradient(alpha0,alpha1,alpha2,alpha3,alpha4,alpha5,sigma0,sigma1,sigma2,sigma3,sigma4,sigma5,e0,e1,e2,e3,e4,e5,x_prev,u,x_curr,Cov,theta,Q)
            like_a0+=la1
            like_a1+=la2
            like_a2+=la3
            like_a3+=la4
            like_a4+=la5
            like_a5+=la6
            
            like_s0+=ls1
            like_s1+=ls2
            like_s2+=ls3
            like_s3+=ls4
            like_s4+=ls5
            like_s5+=ls6
        
        gradients_alpha1.append(like_a0+gp1-var_a_p1)
        gradients_alpha2.append(like_a1+gp2-var_a_p2)
        gradients_alpha3.append(like_a2+gk1-var_a_k1)
        gradients_alpha4.append(like_a3+gk2-var_a_k2)
        gradients_alpha5.append(like_a4+gk3-var_a_k3)
        gradients_alpha6.append(like_a5+gk4-var_a_k4)
                            
        gradients_s1.append(like_s0+sp1-var_s_p1)
        gradients_s2.append(like_s1+sp2-var_s_p2)
        gradients_s3.append(like_s2+sk1-var_s_k1)
        gradients_s4.append(like_s3+sk2-var_s_k2)
        gradients_s5.append(like_s4+sk3-var_s_k3)
        gradients_s6.append(like_s5+sk4-var_s_k4)
        
       
        
    
    gradient_alpha=[]
    #########################################
    temp=0
    for g in gradients_alpha1:
        temp+=g/M
    gradient_alpha.append(np.matrix(temp))
    
    #########################################
    temp=0
    for g in gradients_alpha2:
        temp+=g/M
    gradient_alpha.append(np.matrix(temp))
    
    #########################################
    temp=0
    for g in gradients_alpha3:
        temp+=g/M
    gradient_alpha.append(np.matrix(temp))
    
    #########################################
    temp=0
    for g in gradients_alpha4:
        temp+=g/M
    gradient_alpha.append(np.matrix(temp))
    
    #########################################
    temp=0
    for g in gradients_alpha5:
        temp+=g/M
    gradient_alpha.append(np.matrix(temp))
    
    #########################################
    temp=0
    for g in gradients_alpha6:
        temp+=g/M
    gradient_alpha.append(np.matrix(temp))
  ###################################################################
#####################################################################  
    
    gradient_s=[]
    #########################################
    temp=0
    for g in gradients_s1:
        temp+=g/M
    gradient_s.append(np.matrix(temp))
    
    #########################################
    temp=0
    for g in gradients_s2:
        temp+=g/M
    gradient_s.append(np.matrix(temp))
    
    #########################################
    temp=0
    for g in gradients_s3:
        temp+=g/M
    gradient_s.append(np.matrix(temp))
    
    #########################################
    temp=0
    for g in gradients_s4:
        temp+=g/M
    gradient_s.append(np.matrix(temp))
    
    #########################################
    temp=0
    for g in gradients_s5:
        temp+=g/M
    gradient_s.append(np.matrix(temp))
    
    #########################################
    temp=0
    for g in gradients_s6:
        temp+=g/M
    gradient_s.append(np.matrix(temp))
    ###################################################################
#####################################################################  
      
    gradient_alpha=np.matrix(np.asarray(gradient_alpha)).T
    gradient_s=np.matrix(np.asarray(gradient_s))
    
    m_alpha=beta1*m_alpha+(1-beta1)*gradient_alpha
    m_alpha_hat=m_alpha/(1-beta1**iter)
    v_alpha=beta2*v_alpha+(1-beta2)*np.power(gradient_alpha,2)
    v_alpha_hat=v_alpha/(1-beta2**iter)
    
    m_sigma=beta1*m_sigma+(1-beta1)*gradient_s
    m_sigma_hat=m_sigma/(1-beta1**iter)
    v_sigma=beta2*v_sigma+(1-beta2)*np.power(gradient_s,2)
    v_s_hat=v_sigma/(1-beta2**iter)
    
    alpha_past=alpha_est
    
    alpha_est=alpha_est+a*np.divide(m_alpha_hat,np.sqrt(v_alpha_hat)+1e-8)
    sigma_est=sigma_est+a*np.divide(m_sigma_hat,np.sqrt(v_s_hat)+1e-8)
    
    print(sigma_est)
    print(alpha_est)
    
    count=0
    for n in range(0,6):
        diff=alpha_est[:,n]-alpha_past[:,n]
        dist=np.sqrt(diff.T*diff)
        if dist<=a:
            count+=1
    
    if count==6:
        break

with open('objs.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([alpha_est,sigma_est], f)
