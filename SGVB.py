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





M=3 #Number of samples drawn from N(0,I) at each iteration
max_iter=1000 #Maximum number of iterations SGVB will run for 

#Parameters of ADAM
a=0.08
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


#Extract LDS that corresponds to Parents
theta_p=theta[:,:,0:NumParents]
theta_k=theta[:,:,NumParents:NumParents+NumKids]

IC_p=Cov[:,:,0:NumParents]
IC_k=Cov[:,:,NumParents:NumParents+NumKids]

for p in range(0,NumParents):
    IC_p[:,:,p]=np.matrix(IC_p[:,:,p]).I
for k in range(0,NumKids):
    IC_k[:,:,k]=np.matrix(IC_k[:,:,k]).I

IQ=np.matrix(Q).I


"""
alpha and theta will follow this convention
[P1, Kids of P1, P2 kids of P2, etc ]
"""
#Beta is also encoded in this array
alpha_est=np.random.multivariate_normal(mu_prior,Cov_prior,NumParents).T
beta_est=np.random.multivariate_normal(mu_prior,0.5*np.eye(dim),NumKids).T
sigma_est=np.sqrt(10)*np.ones(NumParents)
nu_est=np.sqrt(10)*np.ones(NumKids)
   

#ADAM parameters for alpha 
m_alpha=np.zeros((dim,NumParents))
v_alpha=np.zeros((dim,NumParents))

#ADAM parameters for beta
m_beta=np.zeros((dim,NumKids))
v_beta=np.zeros((dim,NumKids))

#ADAM parameters for sigma
m_sigma=np.zeros((NumParents))
v_sigma=np.zeros((NumParents))

#ADAM parameters for nu
m_nu=np.zeros((NumKids))
v_nu=np.zeros((NumKids))

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
    gradients_sigma=np.zeros((M,NumParents))
    gradients_nu=np.zeros((M,NumKids))
    
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
        
        #Take gradient of likelihood
        grad_like_alpha=np.zeros((dim,NumParents))
        grad_like_sigma=np.zeros((1,NumParents))
        grad_like_beta=np.zeros((dim,NumKids))
        grad_like_nu=np.zeros((1,NumKids))
        
        
        for t in range(0,2001):
            x_prev=np.matrix(states[:,t])
            x_curr=np.matrix(states[:,t+1])
            u=x_prev
            u=np.concatenate((u,[[1]]),axis=0)
            #Compute gradients from Likelihood
            alpha_grad,beta_grad, sigma_grad, nu_grad=grad.like_gradient_mu(dim,alpha_est,beta_est,sigma_est,nu_est,e_p[:,:,m],\
                                                                        e_k[:,:,m],IC_p,IC_k,theta_p,theta_k,x_prev,\
                                                                        x_curr,u,IQ,NumParents,NumKids,PerParent)
        
            grad_like_alpha+=np.matrix(alpha_grad.numpy())
            grad_like_sigma+=np.matrix(sigma_grad.numpy())
            grad_like_beta+=np.matrix(beta_grad.numpy())
            grad_like_nu+=np.matrix(nu_grad.numpy())
        
        
        #Samples from gradient of ELBO
        gradients_alpha[:,:,m]=grad_like_alpha+grad_prior_alpha
        gradients_sigma[m,:]=grad_like_sigma+grad_prior_sigma-grad_var_sigma
        gradients_beta[:,:,m]=grad_like_beta
        gradients_nu[m,:]=grad_like_nu+grad_prior_nu-grad_var_nu
        
    
    gradient_alpha=np.mean(gradients_alpha,axis=2)
    gradient_sigma=np.mean(gradients_sigma,axis=0)
    gradient_beta=np.mean(gradients_beta,axis=2)
    gradient_nu=np.mean(gradients_nu,axis=0)
    
    
    
    m_alpha=beta1*m_alpha+(1-beta1)*gradient_alpha
    m_alpha_hat=m_alpha/(1-beta1**iter)
    v_alpha=beta2*v_alpha+(1-beta2)*np.power(gradient_alpha,2)
    v_alpha_hat=v_alpha/(1-beta2**iter)
    
    m_sigma=beta1*m_sigma+(1-beta1)*gradient_sigma
    m_sigma_hat=m_sigma/(1-beta1**iter)
    v_sigma=beta2*v_sigma+(1-beta2)*np.power(gradient_sigma,2)
    v_sigma_hat=v_sigma/(1-beta2**iter)
    
    m_beta=beta1*m_beta+(1-beta1)*gradient_beta
    m_beta_hat=m_beta/(1-beta1**iter)
    v_beta=beta2*v_beta+(1-beta2)*np.power(gradient_beta,2)
    v_beta_hat=v_beta/(1-beta2**iter)
    
    m_nu=beta1*m_nu+(1-beta1)*gradient_nu
    m_nu_hat=m_nu/(1-beta1**iter)
    v_nu=beta2*v_nu+(1-beta2)*np.power(gradient_nu,2)
    v_nu_hat=v_nu/(1-beta2**iter)
    
    
    
    alpha_past=alpha_est
    sigma_past=sigma_est
    beta_past=beta_est
    nu_past=nu_est
    
    alpha_est=alpha_past+a*np.divide(m_alpha_hat,np.sqrt(v_alpha_hat)+1e-8)
    sigma_est=sigma_past+a*np.divide(m_sigma_hat,np.sqrt(v_sigma_hat)+1e-8)
    beta_est=beta_past+a*np.divide(m_beta_hat,np.sqrt(v_beta_hat)+1e-8)
    nu_est=nu_past+a*np.divide(m_nu_hat,np.sqrt(v_nu_hat)+1e-8)
    
    print(alpha_est)
    print(beta_est)

    
    count=0
    for n in range(0,NumParents):
        diff=alpha_est[:,n]-alpha_past[:,n]
        dist=np.sqrt(diff.T*diff)
        if dist<=0.07:
            count+=1
    
    for n in range(0,NumKids):
        diff=beta_est[:,n]-beta_past[:,n]
        dist=np.sqrt(diff.T*diff)
        if dist<=0.07:
            count+=1
    
    if count==6:
        break
    
np.save(alpha_est,'alpha_parents')
np.save(beta_est,'beta_kids')



