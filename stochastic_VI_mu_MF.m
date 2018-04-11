%This function is the first building block of learning all the parameters in the model 
%Variotional distributions will follows q(x)~N(alpha,RR'). The goal is to
%learn alpha and R so that we can minimize the KLD between the variational
%distributions and the posterior (aka the target). Will use the Mean field
%theory approach
function [alpha_curr_var,R_curr_var]= stochastic_VI_mu_MF(sample_mu_tree,cov_tree,ld_tree,x,N,S,Q,max_iter)
%mu_prior,Sigma_prior= hyperparameters for prior on layer 1 of the model
%sample_mu_tree= A tree structure that contains the current estimates of the means of the clusters
%cov_tree= A tree structure that contains the current estimates of the covariances of the clusters
%ld_tree= A tree structure that contains the current estimates of the linear dynamics associated with each cluster
%x= latent states
%P= number of first generation parents
%N= number of nodes in tree
%S= number of samples taken from variational distribution
%Q= state covariance
%max_iter= maximum number of iterations algorithm will run

    error=1;
    dim=length(x(:,1));
    alpha_curr_var=cell(N,1); %will be used to store current estimate of the mean of the variational distribution
    R_curr_var=cell(N,1); %Used to store the current estimate of R (square root of Sigma) of the variational distribution
    G=cell(N,1); %Used to store learning rates. Will follow AdaGrad
    eta=0.5;
    %Extract inital estimate of mu from tree
    for jj=2:N+1
        temp=sample_mu_tree.get(jj);
        alpha_curr_var{jj-1}=temp{1};
        R_curr_var{jj-1}=5*eye(dim);
        G{jj-1}=ones(dim,1)/dim; 
    end
    clear temp;
    G=zeros(dim,1); %Will use the AdaGrad adaptation of the learning rate
    
    alpha_prev_var=alpha_curr_var; %l2norm(mu_prev-mu_curr) will be used for stopping condition
    R_prev_var=R_curr_var; %Frobnorm(mu_prev-mu_curr) will be used for stopping condition
    iter=1;
    
    epsilon=zeros(dim,S,N); %Will be used to store values drawn from N(O,I)
    
    while iter<max_iter
        %Draw S samples from N(0,I)
        for z=1:N
            epsilon(:,:,z)=randn(dim,S);
        end
        
        %Loop over all nodes in tree
        for z=1:N
            %Compute estimate of gradient
            [alpha_grad_est,R_grad_est]=gradient_est_MF(alpha_prev_var,R_prev_var,sample_mu_tree,cov_tree,ld_tree,z,x,N,S,Q,epsilon);
            alpha_grad(:,z)=alpha_grad_est;
            R_grad(:,:,z)=R_grad_est;  
        end
        
        %Update parameters
        rho=1/iter;
        for z=1:N
            alpha_curr_var{z}=alpha_curr_var{z}+rho*alpha_grad(:,z);
            R_curr_var{z}=R_curr_var{z}+rho*R_grad(:,:,z);
        end
        iter=iter+1;
    end
end