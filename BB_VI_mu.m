% Black-Box VI method for mean vectors of the clusters
function mu=BB_VI_mu(mu_prior,Sigma_prior,sample_mu_tree,cov_tree,ld_tree,x,K,layers,N)
%mu_prior,Sigma_prior= hyperparameters for prior on layer 1 of the model
%sample_mu_tree= A tree structure that contains the current estimates of the means of the clusters
%cov_tree= A tree structure that contains the current estimates of the covariances of the clusters
%ld_tree= A tree strructure that contains the current estimates of the linear dynamics associated with each cluster
%x= latent states
%K= number of children per parent
%layers= depth of the tree
%N= number of nodes in tree
    error=1;
    mu_est=cell{N,1};
    for jj=2:N
        mu_est=sample_mu_tree.get(2)(1);
    end
    while error>0.01
        
    end


end