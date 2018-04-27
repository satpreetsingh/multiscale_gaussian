#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 01:02:27 2018

@author: Josue Nassar
Running Stochastic Gradient Variational Bayes to test how accuratcely we can determine 
the location of the posited gaussians. For the parents, they will follow 
mu_p~N(alpha_p,s_p^2*I) (i.e. an isotropic gaussian) Will optimize alpha and s.
To ensure non-negativity constraint, will project s onto the positive reals which corresponds
to taking the absolute value i.e. s_t=abs(s_+eta*g)

The kids will be conditioned on the parents
mu_k|mu_p~N(mu_k+beta_p,s_k^2*I) where beta_p is an offset. projection will also 
be done on s_k
So we will learn alpha_p's and 
"""
import torch
from torch import FloatTensor
from torch.autograd import Variable
import numpy as np
import scipy.io as sio
import gradients as grad

#Load in synthetic states from matlab
mat_contents=sio.loadmat('synthetic_states.mat')
states=np.matrix(mat_contents['x'])
states=states[:,0:1000]
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

M=1 #Number of samples drawn from N(0,I) at each iteration
max_iter=1000 #Maximum number of iterations SGVB will run for 

#Parameters of ADAM
a=0.1
beta1=0.9
beta2=0.999

#Set up prior for parents
mu_prior=np.zeros(dim)
Cov_prior=10*np.eye(dim)
Cov_k=2*np.eye(dim) #Covariance matrix for prior N(mu_k|mu_p)
NumParents=2 #Number of parents
NumKids=4 #Number of kids
PerParent=int(NumKids/NumParents) #Each parents will have equal number of children
NumFam=NumParents #Number of families
"""
alpha and theta will follow this convention
[P1, Kids of P1, P2 kids of P2, etc ]
"""
#Beta is also encoded in this array
alpha_est=np.random.multivariate_normal(mu_prior,Cov_prior,NumParents).T
beta_est=np.random.multivariate_normal(mu_prior,0.5*np.eye(dim),NumKids).T
sigma_est=np.sqrt(10)*np.ones(NumParents)
nu_est=np.sqrt(10)*np.ones(NumKids)
##Create a matrix that holds families. 
#fam_alpha=np.zeros((dim,1+PerParent,NumFam))
#
#point=0 #Use to keep track of position for the for loop below
#for p in range(0,NumParents):
#    fam_alpha[:,0,p]=alpha_est[:,p]
#    fam_alpha[:,1:1+PerParent,p]=beta_est[:,point:point+PerParent]
#    point+=PerParent
#fam_sigma=np.reshape(sigma_est,(NumFam,PerParent+1))    

#ADAM parameters for alpha 
m_alpha=np.zeros((dim,NumParents))
v_alpha=np.zeros((dim,NumParents))

#ADAM parameters for beta
m_beta=np.zeros((dim,NumKids))
v_beta=np.zeros((dim,NumKids))

#ADAM parameters for sigma
m_sigma=np.zeros((dim,NumParents))
v_sigma=np.zeros((dim,NumParents))

#ADAM parameters for nu
m_nu=np.zeros((dim,NumKids))
v_nu=np.zeros((dim,NumKids))

iter=0
while iter<max_iter: # TODO: write this as a for loop! (you can break)
    iter+=1
    print(iter)
    #Draw M samples from N(0,I)
    e_p=np.random.randn(dim,NumParents,M)#Samples for parents
    e_k=np.random.randn(dim,NumKids,M) #Samples for kids
    
    #Initalize matrix to store gradients of alpha,beta, sigma
    gradients_alpha=np.zeros((dim,NumParents,M))
    gradients_beta=np.zeros((dim,NumKids,M))
    gradients_sigma=np.zeros(NumParents)
    gradients_nu=np.zeros(NumKids)
    
    for m in range(0,M):
        
        #Compute gradients of variational distribution
        #(NOTE: Variational distribution gives gradient 0 for alpha and beta)
        grad_var_sigma=-dim*np.power(sigma_est,-1)
        grad_var_nu=-dim*np.power(nu_est,-1)
        
        
        #Compute gradients of prior distribution
        grad_prior_alpha=np.zeros((dim,NumParents))
        grad_prior_sigma=np.zeros(NumParents)
        grad_prior_nu=np.zeros(NumKids)
        point=0 #Used to keep track of location in matrix
        for p in range(0,NumParents):
            #NOTE: The prior distribution only gives non-zero gradients to location parameters 
            #of parents
            grad_prior_alpha[:,p]=np.array(-0.5*np.matrix(Cov_prior).I*np.matrix(alpha_est[:,p]\
                            +sigma_est[point]*e_p[:,p,m]-mu_prior).T).ravel()
            grad_prior_sigma[point]=np.array(np.matrix(grad_prior_alpha[:,p])*\
                            np.matrix(e_p[:,p,m]).T).ravel()
        
        for k in range(0,NumKids):
            grad_prior_nu[k]=np.array(-nu_est[k]*np.matrix(e_k[:,k,m])*np.matrix(Cov_k).I*np.matrix(e_k[:,k,m]).T).ravel()
                

        
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
    print(gradient_alpha)
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
    
    all_alpha_est[:,:,iter]=alpha_est
    all_sigma_est[iter,:]=sigma_est
    
    count=0
    for n in range(0,6):
        diff=alpha_est[:,n]-alpha_past[:,n]
        dist=np.sqrt(diff.T*diff)
        if dist<=0.1:
            count+=1
    
    if count==6:
        break



