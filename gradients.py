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

#def variational_gradient_mu(alpha_est,phi_est,samp,dim):
#    alpha=Variable(torch.from_numpy(alpha_est).float(),requires_grad=True)
#    alpha.retain_grad()
#    phi=Variable(torch.from_numpy(phi_est).float(),requires_grad=True)
#    phi.retain_grad()
#    e=Variable(torch.from_numpy(samp).float())
#    
#    w1=(alpha+phi*e-alpha)
#    
#    w2=w1.transpose(0,1)
#    w3=phi*phi
#    arg=torch.matmul(w2,w1)
#    arg_3=arg/w3
#    arg_2=-0.5*arg_3
#    arg_1=-dim*0.5*torch.log(2*np.pi*phi*phi)
#    L=arg_1-arg_2
#    L.backward()
#    return alpha.grad.data, phi.grad.data
#
#def variational_gradient_cov(beta_est,theta_est,samp):
#    beta=Variable(torch.from_numpy(beta_est).float(),requires_grad=True)
#    beta.retain_grad()
#    theta=Variable(torch.from_numpy(theta_est).float(),requires_grad=True)
#    theta.retain_grad()
#    z=Variable(torch.from_numpy(samp).float())
#    
#    w1=torch.log(theta)
#    w2=torch.sqrt(theta)*z
#    L=-beta-w2-0.5*w1
#    L.backward()
#    return beta.grad.data,theta.grad.data
#
#    
#
#def prior_gradient_mu(alpha_p,R_p,es_p,alpha_k1,R_k1,es_k1,alpha_k2,R_k2,es_k2,mu_prior,Cov_prior):
#    a_p=Variable(torch.from_numpy(alpha_p).float(),requires_grad=True)
#    a_p.retain_grad()
#    s_p=Variable(torch.from_numpy(R_p).float(),requires_grad=True)
#    s_p.retain_grad()
#    e_p=Variable(torch.from_numpy(es_p).float())
#    
#    a_k1=Variable(torch.from_numpy(alpha_k1).float(),requires_grad=True)
#    a_k1.retain_grad()
#    s_k1=Variable(torch.from_numpy(R_k1).float(),requires_grad=True)
#    s_k1.retain_grad()
#    e_k1=Variable(torch.from_numpy(es_k1).float())
#    
#    a_k2=Variable(torch.from_numpy(alpha_k2).float(),requires_grad=True)
#    a_k2.retain_grad()
#    s_k2=Variable(torch.from_numpy(R_k2).float(),requires_grad=True)
#    s_k2.retain_grad()
#    e_k2=Variable(torch.from_numpy(es_k2).float())
#    
#    mu=Variable(torch.from_numpy(mu_prior).float())
#    prior_P=Variable(torch.from_numpy(Cov_prior.I).float())
#    like_P=Variable(torch.from_numpy(np.eye(2)).float())
#    
#    w1=a_p+s_p*e_p-mu
#    w2=w1.transpose(0,1)
#    w3=torch.matmul(w2,-0.5*prior_P)
#    arg1=torch.matmul(w3,w1)
#    
#    w4=a_k1+s_k1*e_k1-a_p-s_p*e_p
#    w5=w4.transpose(0,1)
#    w6=torch.matmul(w5,-0.5*like_P)
#    arg2=torch.matmul(w6,w4)
#    
#    w7=a_k2+s_k2*e_k2-a_p-s_p*e_p
#    w8=w7.transpose(0,1)
#    w9=torch.matmul(w8,-0.5*like_P)
#    arg3=torch.matmul(w9,w7)
#    
#    L=arg1+arg2+arg3
#    L.backward()
#    return a_p.grad.data,s_p.grad.data,a_k1.grad.data,s_k1.grad.data,a_k2.grad.data,s_k2.grad.data


def like_gradient(alpha1,alpha2,alpha3,alpha4,alpha5,alpha6,sigma1,sigma2,sigma3,sigma4,sigma5,sigma6,ep1,ep2,ep3,ep4,ep5,ep6,x_t,u_t,y,Cov,theta,Q):
    a1=Variable(torch.from_numpy(alpha1).float(),requires_grad=True)
    a1.retain_grad()
    a2=Variable(torch.from_numpy(alpha2).float(),requires_grad=True)
    a2.retain_grad()
    a3=Variable(torch.from_numpy(alpha3).float(),requires_grad=True)
    a3.retain_grad()
    a4=Variable(torch.from_numpy(alpha4).float(),requires_grad=True)
    a4.retain_grad()
    a5=Variable(torch.from_numpy(alpha5).float(),requires_grad=True)
    a5.retain_grad()
    a6=Variable(torch.from_numpy(alpha6).float(),requires_grad=True)
    a6.retain_grad()
    
    s1=Variable(torch.from_numpy(sigma1).float(),requires_grad=True)
    s1.retain_grad()
    s2=Variable(torch.from_numpy(sigma2).float(),requires_grad=True)
    s2.retain_grad()
    s3=Variable(torch.from_numpy(sigma3).float(),requires_grad=True)
    s3.retain_grad()
    s4=Variable(torch.from_numpy(sigma4).float(),requires_grad=True)
    s4.retain_grad()
    s5=Variable(torch.from_numpy(sigma5).float(),requires_grad=True)
    s5.retain_grad()
    s6=Variable(torch.from_numpy(sigma6).float(),requires_grad=True)
    s6.retain_grad()
    
    e1=Variable(torch.from_numpy(ep1).float())
    e2=Variable(torch.from_numpy(ep2).float())
    e3=Variable(torch.from_numpy(ep3).float())
    e4=Variable(torch.from_numpy(ep4).float())
    e5=Variable(torch.from_numpy(ep5).float())
    e6=Variable(torch.from_numpy(ep6).float())

    
    x=Variable(torch.from_numpy(x_t).float())
    u=Variable(torch.from_numpy(u_t).float())
    z=Variable(torch.from_numpy(y).float())
    
    T1=Variable(torch.from_numpy(theta[:,:,0]).float())
    T2=Variable(torch.from_numpy(theta[:,:,1]).float())
    T3=Variable(torch.from_numpy(theta[:,:,2]).float())
    T4=Variable(torch.from_numpy(theta[:,:,3]).float())
    T5=Variable(torch.from_numpy(theta[:,:,4]).float())
    T6=Variable(torch.from_numpy(theta[:,:,5]).float())
    
    L1=Variable(torch.from_numpy(np.matrix(Cov[:,:,0]).I).float())
    L2=Variable(torch.from_numpy(np.matrix(Cov[:,:,1]).I).float())
    L3=Variable(torch.from_numpy(np.matrix(Cov[:,:,2]).I).float())
    L4=Variable(torch.from_numpy(np.matrix(Cov[:,:,3]).I).float())
    L5=Variable(torch.from_numpy(np.matrix(Cov[:,:,4]).I).float())
    L6=Variable(torch.from_numpy(np.matrix(Cov[:,:,5]).I).float())
    
    state_L=Variable(torch.from_numpy(Q.I).float())
    
    temp1=x-a1-s1*e1
    temp2=temp1.transpose(0,1)
    temp3=torch.matmul(temp2,-L1)
    temp4=torch.matmul(temp3,temp1)
    w1=torch.exp(temp4)+1e-20
    
    temp5=x-a2-s2*e2
    temp6=temp5.transpose(0,1)
    temp7=torch.matmul(temp6,-L2)
    temp8=torch.matmul(temp7,temp5)
    w2=torch.exp(temp8)+1e-20
    
    temp9=x-a3-s3*e3
    temp10=temp9.transpose(0,1)
    temp11=torch.matmul(temp10,-L3)
    temp12=torch.matmul(temp11,temp9)
    w3=torch.exp(temp12)+1e-20
    
    temp13=x-a4-s4*e4
    temp14=temp13.transpose(0,1)
    temp15=torch.matmul(temp14,-L4)
    temp16=torch.matmul(temp15,temp13)
    w4=torch.exp(temp16)+1e-20
    
    temp17=x-a5-s5*e5
    temp18=temp17.transpose(0,1)
    temp19=torch.matmul(temp18,-L5)
    temp20=torch.matmul(temp19,temp17)
    w5=torch.exp(temp20)+1e-20
    
    temp21=x-a6-s6*e6
    temp22=temp21.transpose(0,1)
    temp23=torch.matmul(temp22,-L6)
    temp24=torch.matmul(temp23,temp21)
    w6=torch.exp(temp24)+1e-20
    
    Z=w1+w2+w3+w4+w5+w6
    
    w1_norm=w1/Z
    w2_norm=w2/Z
    w3_norm=w3/Z
    w4_norm=w4/Z
    w5_norm=w5/Z
    w6_norm=w6/Z
    
    x1=w1_norm*torch.matmul(T1,u)
    x2=w2_norm*torch.matmul(T2,u)
    x3=w3_norm*torch.matmul(T3,u)
    x4=w4_norm*torch.matmul(T4,u)
    x5=w5_norm*torch.matmul(T5,u)
    x6=w6_norm*torch.matmul(T6,u)
    
    m=x1+x2+x3+x4+x5+x6
    
    arg1=z-m
    arg2=arg1.transpose(0,1)
    arg3=torch.matmul(arg2,-0.5*state_L)
    L=torch.matmul(arg3,arg1)
    L.backward()
    
    return a1.grad.data,a2.grad.data,a3.grad.data,a4.grad.data,a5.grad.data,a6.grad.data,s1.grad.data,s2.grad.data,s3.grad.data,s4.grad.data,s5.grad.data,s6.grad.data

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
    
    
    