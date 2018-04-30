#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 03:25:17 2018

@author: user
"""

import torch
from torch import FloatTensor
from torch.autograd import Variable
import numpy as np

def like_gradient_mu(dim,alpha,beta,sigma,nu,e_p,e_k,Inv_Cov_p,Inv_Cov_k,LDS_p,LDS_k,x_prev,x_curr,u_prev,IQ,NumParents,NumKids,PerParent):
    
    #Variables we don't need gradient of 
    IC_p=Variable(torch.from_numpy(Inv_Cov_p).float()) #Covariance matrices of parents
    IC_k=Variable(torch.from_numpy(Inv_Cov_k).float()) #Covariance matrices of kids
    Theta_p=Variable(torch.from_numpy(LDS_p).float()) #LDS of parents
    Theta_k=Variable(torch.from_numpy(LDS_k).float()) #LDS of kids
    IQ=Variable(torch.from_numpy(IQ).float()) #Covariance matrix of state noise
    x_p=Variable(torch.from_numpy(x_prev).float()) #x_{t-1}
    x_c=Variable(torch.from_numpy(x_curr).float()) #x_t
    u=Variable(torch.from_numpy(u_prev).float()) #u=[x_{t-1};1]
    ep=Variable(torch.from_numpy(np.matrix(e_p)).float())
    ek=Variable(torch.from_numpy(np.matrix(e_k)).float())
    
    #Variables that we do need the gradients of 
    a=Variable(torch.from_numpy(np.matrix(alpha)).float(),requires_grad=True)
    a.retain_grad()
    b=Variable(torch.from_numpy(np.matrix(beta)).float(),requires_grad=True)
    b.retain_grad()
    s=Variable(torch.from_numpy(np.matrix(sigma)).float(),requires_grad=True)
    s.retain_grad()
    n=Variable(torch.from_numpy(np.matrix(nu)).float(),requires_grad=True)
    n.retain_grad()
    
    #Used to store weights
    weights_parents=Variable(torch.from_numpy(np.matrix(np.zeros(NumParents)).T).float())
    weights_kids=Variable(torch.from_numpy(np.matrix(np.zeros(NumKids)).T).float())
    x_temp_parents=Variable(torch.from_numpy(np.matrix(np.zeros((dim,NumParents)))).float())
    x_temp_kids=Variable(torch.from_numpy(np.matrix(np.zeros((dim,NumKids)))).float())


    #Weigh Clusters
    for p in range(0,NumParents):
        weights_parents[p,0]=torch.exp( torch.matmul( torch.matmul( -(x_p[:,0]-a[:,p]-s[:,p]*ep[:,p]).unsqueeze(0),IC_p[:,:,p]) ,(x_p[:,0]-a[:,p]-s[:,p]*ep[:,p]))   )+1e-50
        x_temp_parents[:,p]=(torch.exp( torch.matmul( torch.matmul( -(x_p[:,0]-a[:,p]-s[:,p]*ep[:,p]).unsqueeze(0),IC_p[:,:,p]) ,(x_p[:,0]-a[:,p]-s[:,p]*ep[:,p]))   )+1e-50)*torch.matmul(Theta_p[:,:,p],u)
        
    for k in range(0,NumKids):
        p=int(k/PerParent)
        weights_kids[k,0]=torch.exp(torch.matmul(torch.matmul(-(x_p[:,0]-a[:,p]-s[:,p]*ep[:,p]-b[:,k]-n[:,k]*ek[:,k]).unsqueeze(0),IC_k[:,:,k]),(x_p[:,0]-a[:,p]-s[:,p]*ep[:,p]-b[:,k]-n[:,k]*ek[:,k])))+1e-50
        x_temp_kids[:,k]=(torch.exp(torch.matmul(torch.matmul(-(x_p[:,0]-a[:,p]-s[:,p]*ep[:,p]-b[:,k]-n[:,k]*ek[:,k]).unsqueeze(0),IC_k[:,:,k]),(x_p[:,0]-a[:,p]-s[:,p]*ep[:,p]-b[:,k]-n[:,k]*ek[:,k])))+1e-50)*torch.matmul(Theta_k[:,:,k],u)
    

    #Normalize
    Z=torch.sum(weights_parents)+torch.sum(weights_kids)
    
    x_p_clone=x_temp_parents.clone()
    x_k_clone=x_temp_kids.clone()
    x_unnorm=torch.sum(x_p_clone,1)+torch.sum(x_k_clone.clone(),1)
    x_norm=x_unnorm.clone()/Z
    
    y=x_norm.clone().unsqueeze(1)
    z=x_c-y
    F=torch.matmul(torch.matmul((z).transpose(0,1),IQ),(z))
    
    L=-0.5*F

    L.backward()
    return a.grad.data,b.grad.data,s.grad.data,n.grad.data
    
    
def like_gradient_cov(dim,lambda_p,lambda_k,e_p,e_k,mu_p,mu_k,LDS_p,LDS_k,x_prev,x_curr,u_prev,IQ,NumParents,NumKids,PerParent):
    
    #Variables we don't need gradient of 
    mp=Variable(torch.from_numpy(mu_p).float()) #Covariance matrices of parents
    mk=Variable(torch.from_numpy(mu_k).float()) #Covariance matrices of kids
    Theta_p=Variable(torch.from_numpy(LDS_p).float()) #LDS of parents
    Theta_k=Variable(torch.from_numpy(LDS_k).float()) #LDS of kids
    IQ=Variable(torch.from_numpy(IQ).float()) #Covariance matrix of state noise
    x_p=Variable(torch.from_numpy(x_prev).float()) #x_{t-1}
    x_c=Variable(torch.from_numpy(x_curr).float()) #x_t
    u=Variable(torch.from_numpy(u_prev).float()) #u=[x_{t-1};1]
    ep=Variable(torch.from_numpy(np.matrix(e_p)).float())
    ek=Variable(torch.from_numpy(np.matrix(e_k)).float())
    
    #Variables that we do need the gradients of 
    lp=Variable(torch.from_numpy(np.matrix(lambda_p)).float(),requires_grad=True)
    lp.retain_grad()
    lk=Variable(torch.from_numpy(np.matrix(lambda_k)).float(),requires_grad=True)
    lk.retain_grad()
    
    #Used to store weights
    weights_parents=Variable(torch.from_numpy(np.matrix(np.zeros(NumParents)).T).float())
    weights_kids=Variable(torch.from_numpy(np.matrix(np.zeros(NumKids)).T).float())
    x_temp_parents=Variable(torch.from_numpy(np.matrix(np.zeros((dim,NumParents)))).float())
    x_temp_kids=Variable(torch.from_numpy(np.matrix(np.zeros((dim,NumKids)))).float())
    

    #Weigh Clusters
    for p in range(0,NumParents):
        weights_parents[p,0]=torch.exp(torch.matmul(-0.5*(x_p[:,0]-mp[:,p]).unsqueeze(0),(x_p[:,0]-mp[:,p]))/(-ep[:,p]*torch.log(1+torch.exp(lp[:,p])))  )+1e-50
        x_temp_parents[:,p]=(torch.exp(torch.matmul(-0.5*(x_p[:,0]-mp[:,p]).unsqueeze(0),(x_p[:,0]-mp[:,p]))/(-ep[:,p]*torch.log(1+torch.exp(lp[:,p])))  )+1e-50)*torch.matmul(Theta_p[:,:,p],u)
        
    for k in range(0,NumKids):
        p=int(k/PerParent)
        weights_kids[k,0]=torch.exp(torch.matmul(-0.5*(x_p[:,0]-mk[:,k]).unsqueeze(0),(x_p[:,0]-mk[:,k]))/(-ek[:,k]*torch.log(1+torch.exp(lk[:,k])))  )+1e-50
        x_temp_kids[:,k]=(torch.exp(torch.matmul(-0.5*(x_p[:,0]-mk[:,k]).unsqueeze(0),(x_p[:,0]-mk[:,k]))/(-ek[:,k]*torch.log(1+torch.exp(lk[:,k])) )  )+1e-50)*torch.matmul(Theta_k[:,:,k],u)
    

    #Normalize
    Z=torch.sum(weights_parents)+torch.sum(weights_kids)
    
    x_p_clone=x_temp_parents.clone()
    x_k_clone=x_temp_kids.clone()
    x_unnorm=torch.sum(x_p_clone,1)+torch.sum(x_k_clone.clone(),1)
    x_norm=x_unnorm.clone()/Z
    
    y=x_norm.clone().unsqueeze(1)
    z=x_c-y
    F=torch.matmul(torch.matmul((z).transpose(0,1),IQ),(z))
    
    L=-0.5*F

    L.backward()
    return lp.grad.data,lk.grad.data