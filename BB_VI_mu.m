% Black-Box VI method for mean vectors of the clusters
function mu_curr_var=BB_VI_mu(sample_mu_tree,cov_tree,ld_tree,x,N,S,Q,max_iter)
%mu_prior,Sigma_prior= hyperparameters for prior on layer 1 of the model
%sample_mu_tree= A tree structure that contains the current estimates of the means of the clusters
%cov_tree= A tree structure that contains the current estimates of the covariances of the clusters
%ld_tree= A tree strructure that contains the current estimates of the linear dynamics associated with each cluster
%x= latent states
%N= number of nodes in tree
%S= number of samples taken from variational distribution
%Q= state covariance
%max_iter= maximum number of iterations algorithm will run
    error=1;
    dim=length(x(:,1));
    mu_curr_var=cell(N,1); %will be used to store current estimate of the means of the gaussians
    mu_samp=zeros(dim,S,N); %Store samples drawn from variational distributions 
    G=cell(N,1); %Used to store learning rates. Will follow AdaGrad
    eta=0.5;
    %Extract inital estimate of mu from tree
    for jj=2:N+1
        temp=sample_mu_tree.get(jj);
        mu_curr_var{jj-1}=temp{1};
        G{jj-1}=ones(dim,1)/dim; 
    end
    clear temp;
    G=zeros(dim,1); %Will use the AdaGrad adaptation of the learning rate
    
    mu_prev_var=mu_curr_var; %l2norm(mu_prev-mu_curr) will be used for stopping condition
    iter=1;
    while iter<max_iter
       %Sample from variational distributions
       for z=1:N
           mu_samp(:,:,z)=mvnrnd(mu_prev_var{z}',cov_tree.get(z+1),S)';
       end
       
       %Update parameters of variational distribution
       for z=1:N
           %Compute gradient
           g=BB_VI_gradient_estimate(sample_mu_tree,ld_tree,cov_tree,x,mu_samp,z,Q,N,S,mu_prev_var);
           %Sum of outer products of gradients over iterations
           G=G+diag(g*g');
           %Compute learning rate
           rho=eta*sqrt(G);
           %Update
           mu_curr_var{z}=mu_prev_var{z}+rho.*g;
       end
       
       mu_curr=[];
       mu_prev=[];
       for z=1:N
           mu_curr=[mu_curr;mu_curr_var{z}];
           mu_prev=[mu_prev;mu_prev_var{z}];
       end
       error=sqrt((mu_curr-mu_prev)'*(mu_curr-mu_prev));
       if error<=0.1
           break;
       end
       iter=iter+1;
       mu_prev_var=mu_curr_var;
    end


end