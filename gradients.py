#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 01:43:58 2018

@author: user
"""

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

def variational_gradient(alpha_est,R_est,samp):
    a=Variable(torch.from_numpy(alpha_est).cuda().float(),requires_grad=True)
    a.retain_grad()
    R=Variable(torch.from_numpy(R_est).cuda().float(),requires_grad=True)
    R.retain_grad()
    e=Variable(torch.from_numpy(samp).cuda().float())
    
    w1=(a+R*e-a)
    
    w2=w1.transpose(0,1)
    w3=R*R
    arg=torch.matmul(w2,w1).cuda()
    arg_3=arg/w3
    arg_2=-0.5*arg_3
    arg_1=-torch.log(2*np.pi*R*R).cuda()
    L=arg_1-arg_2
    L.backward()
    return a.grad.data, R.grad.data

def prior_gradient(alpha_p,R_p,es_p,alpha_k1,R_k1,es_k1,alpha_k2,R_k2,es_k2,mu_prior,Cov_prior):
    a_p=Variable(torch.from_numpy(alpha_p).cuda().float(),requires_grad=True)
    a_p.retain_grad()
    s_p=Variable(torch.from_numpy(R_p).cuda().float(),requires_grad=True)
    s_p.retain_grad()
    e_p=Variable(torch.from_numpy(es_p).cuda().float())
    
    a_k1=Variable(torch.from_numpy(alpha_k1).cuda().float(),requires_grad=True)
    a_k1.retain_grad()
    s_k1=Variable(torch.from_numpy(R_k1).cuda().float(),requires_grad=True)
    s_k1.retain_grad()
    e_k1=Variable(torch.from_numpy(es_k1).cuda().float())
    
    a_k2=Variable(torch.from_numpy(alpha_k2).cuda().float(),requires_grad=True)
    a_k2.retain_grad()
    s_k2=Variable(torch.from_numpy(R_k2).cuda().float(),requires_grad=True)
    s_k2.retain_grad()
    e_k2=Variable(torch.from_numpy(es_k2).cuda().float())
    
    mu=Variable(torch.from_numpy(mu_prior).cuda().float())
    prior_P=Variable(torch.from_numpy(Cov_prior.I).cuda().float())
    like_P=Variable(torch.from_numpy(np.eye(2)).cuda().float())
    
    w1=a_p+s_p*e_p-mu
    w2=w1.transpose(0,1)
    w3=torch.matmul(w2,-0.5*prior_P).cuda()
    arg1=torch.matmul(w3,w1).cuda()
    
    w4=a_k1+s_k1*e_k1-a_p-s_p*e_p
    w5=w4.transpose(0,1)
    w6=torch.matmul(w5,-0.5*like_P).cuda()
    arg2=torch.matmul(w6,w4).cuda()
    
    w7=a_k2+s_k2*e_k2-a_p-s_p*e_p
    w8=w7.transpose(0,1)
    w9=torch.matmul(w8,-0.5*like_P).cuda()
    arg3=torch.matmul(w9,w7).cuda()
    
    L=arg1+arg2+arg3
    L.backward()
    return a_p.grad.data,s_p.grad.data,a_k1.grad.data,s_k1.grad.data,a_k2.grad.data,s_k2.grad.data

def like_gradient(alpha1,alpha2,alpha3,alpha4,alpha5,alpha6,sigma1,sigma2,sigma3,sigma4,sigma5,sigma6,ep1,ep2,ep3,ep4,ep5,ep6,x_t,u_t,y,Cov,theta,Q):
    a1=Variable(torch.from_numpy(alpha1).cuda().float(),requires_grad=True)
    a1.retain_grad()
    a2=Variable(torch.from_numpy(alpha2).cuda().float(),requires_grad=True)
    a2.retain_grad()
    a3=Variable(torch.from_numpy(alpha3).cuda().float(),requires_grad=True)
    a3.retain_grad()
    a4=Variable(torch.from_numpy(alpha4).cuda().float(),requires_grad=True)
    a4.retain_grad()
    a5=Variable(torch.from_numpy(alpha5).cuda().float(),requires_grad=True)
    a5.retain_grad()
    a6=Variable(torch.from_numpy(alpha6).cuda().float(),requires_grad=True)
    a6.retain_grad()
    
    s1=Variable(torch.from_numpy(sigma1).cuda().float(),requires_grad=True)
    s1.retain_grad()
    s2=Variable(torch.from_numpy(sigma2).cuda().float(),requires_grad=True)
    s2.retain_grad()
    s3=Variable(torch.from_numpy(sigma3).cuda().float(),requires_grad=True)
    s3.retain_grad()
    s4=Variable(torch.from_numpy(sigma4).cuda().float(),requires_grad=True)
    s4.retain_grad()
    s5=Variable(torch.from_numpy(sigma5).cuda().float(),requires_grad=True)
    s5.retain_grad()
    s6=Variable(torch.from_numpy(sigma6).cuda().float(),requires_grad=True)
    s6.retain_grad()
    
    e1=Variable(torch.from_numpy(ep1).cuda().float())
    e2=Variable(torch.from_numpy(ep2).cuda().float())
    e3=Variable(torch.from_numpy(ep3).cuda().float())
    e4=Variable(torch.from_numpy(ep4).cuda().float())
    e5=Variable(torch.from_numpy(ep5).cuda().float())
    e6=Variable(torch.from_numpy(ep6).cuda().float())

    
    x=Variable(torch.from_numpy(x_t).cuda().float())
    u=Variable(torch.from_numpy(u_t).cuda().float())
    z=Variable(torch.from_numpy(y).cuda().float())
    
    T1=Variable(torch.from_numpy(theta[:,:,0]).cuda().float())
    T2=Variable(torch.from_numpy(theta[:,:,1]).cuda().float())
    T3=Variable(torch.from_numpy(theta[:,:,2]).cuda().float())
    T4=Variable(torch.from_numpy(theta[:,:,3]).cuda().float())
    T5=Variable(torch.from_numpy(theta[:,:,4]).cuda().float())
    T6=Variable(torch.from_numpy(theta[:,:,5]).cuda().float())
    
    L1=Variable(torch.from_numpy(np.matrix(Cov[:,:,0]).I).cuda().float())
    L2=Variable(torch.from_numpy(np.matrix(Cov[:,:,1]).I).cuda().float())
    L3=Variable(torch.from_numpy(np.matrix(Cov[:,:,2]).I).cuda().float())
    L4=Variable(torch.from_numpy(np.matrix(Cov[:,:,3]).I).cuda().float())
    L5=Variable(torch.from_numpy(np.matrix(Cov[:,:,4]).I).cuda().float())
    L6=Variable(torch.from_numpy(np.matrix(Cov[:,:,5]).I).cuda().float())
    
    state_L=Variable(torch.from_numpy(Q.I).cuda().float())
    
    temp1=x-a1-s1*e1
    temp2=temp1.transpose(0,1)
    temp3=torch.matmul(temp2,-L1).cuda()
    temp4=torch.matmul(temp3,temp1).cuda()
    w1=torch.exp(temp4).cuda()
    
    temp5=x-a2-s2*e2
    temp6=temp5.transpose(0,1)
    temp7=torch.matmul(temp6,-L2).cuda()
    temp8=torch.matmul(temp7,temp5).cuda()
    w2=torch.exp(temp8).cuda()
    
    temp9=x-a3-s3*e3
    temp10=temp9.transpose(0,1)
    temp11=torch.matmul(temp10,-L3).cuda()
    temp12=torch.matmul(temp11,temp9).cuda()
    w3=torch.exp(temp12).cuda()
    
    temp13=x-a4-s4*e4
    temp14=temp13.transpose(0,1)
    temp15=torch.matmul(temp14,-L4).cuda()
    temp16=torch.matmul(temp15,temp13).cuda()
    w4=torch.exp(temp16).cuda()
    
    temp17=x-a5-s5*e5
    temp18=temp17.transpose(0,1)
    temp19=torch.matmul(temp18,-L5).cuda()
    temp20=torch.matmul(temp19,temp17).cuda()
    w5=torch.exp(temp20).cuda()
    
    temp21=x-a6-s6*e6
    temp22=temp21.transpose(0,1)
    temp23=torch.matmul(temp22,-L6).cuda()
    temp24=torch.matmul(temp23,temp21).cuda()
    w6=torch.exp(temp24).cuda()
    
    Z=w1+w2+w3+w4+w5+w6
    
    w1_norm=w1/Z
    w2_norm=w2/Z
    w3_norm=w3/Z
    w4_norm=w4/Z
    w5_norm=w5/Z
    w6_norm=w6/Z
    
    x1=w1_norm*torch.matmul(T1,u).cuda()
    x2=w2_norm*torch.matmul(T2,u).cuda()
    x3=w3_norm*torch.matmul(T3,u).cuda()
    x4=w4_norm*torch.matmul(T4,u).cuda()
    x5=w5_norm*torch.matmul(T5,u).cuda()
    x6=w6_norm*torch.matmul(T6,u).cuda()
    
    m=x1+x2+x3+x4+x5+x6
    
    arg1=z-m
    arg2=arg1.transpose(0,1)
    arg3=torch.matmul(arg2,-0.5*state_L).cuda()
    L=torch.matmul(arg3,arg1).cuda()
    L.backward()
    
    return a1.grad.data,a2.grad.data,a3.grad.data,a4.grad.data,a5.grad.data,a6.grad.data,s1.grad.data,s2.grad.data,s3.grad.data,s4.grad.data,s5.grad.data,s6.grad.data
    
    
    
    