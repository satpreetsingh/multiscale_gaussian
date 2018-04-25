# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 16:23:38 2018

@author: josue
"""

#Contains parameters for closed form conditional posteriors

import numpy as np

#Conjugate posterior of Linear Dynamics of a posited gaussian
def LDS_posterior(M,V,Q,x,Theta,index,mu,Cov,NN):
    """
    A Matrix Normal is parameterized by MN(M,Q,V)
    M,V= Prior parameters of matrix normal
    x= latent states
    Theta= linear dynamics
    index=Which Cluster are we estimating
    mu, Cov= mean, covariance of clusters
    NN=Number of Nodes (gaussians)
    """
    T=x[0,:].size-1
    Psi=0
    Sigma=0
    
    for t in range(0,T):
        #First compute weights
        w=np.zeros(NN)
        for jj in range(0,NN):
#            print(np.exp(-(x[:,t]-np.matrix(mu[:,jj]).T ).T*np.matrix(Cov[:,:,jj]).I*(x[:,t]-np.matrix(mu[:,jj]).T )))
            w[jj]=np.exp(-(x[:,t]-np.matrix(mu[:,jj]).T ).T*np.matrix(Cov[:,:,jj]).I*(x[:,t]-np.matrix(mu[:,jj]).T ))+1e-50
        w_norm=w/np.sum(w)
        
        
        Psi+=w_norm[index]*x[:,t+1]*np.concatenate((x[:,t],[[1]]),axis=0).T
        
        for jj in range(0,NN):
            if jj!=index:
                Psi-=w_norm[index]*w_norm[jj]*np.matrix(Theta[:,:,jj])*np.concatenate((x[:,t],[[1]]),axis=0)*np.concatenate((x[:,t],[[1]]),axis=0).T
        
        Sigma+=w_norm[index]**2*np.concatenate((x[:,t],[[1]]),axis=0)*np.concatenate((x[:,t],[[1]]),axis=0).T
    print(Psi)
    Psi_bar=np.matrix(Psi+M*V)
    Sigma_bar=np.matrix(Sigma+V)
    
    Est=Psi_bar*Sigma_bar.I
    L=np.matrix(Est-Theta[:,:,index])
    dist=np.matrix(L.T*L)
    dist=np.trace(dist)
    print(dist)
    return Est,dist
    