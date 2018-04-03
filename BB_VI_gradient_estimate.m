%Will be used to calculate estimate of gradient in Black Box Variational Inference (II) 
function grad_est= BB_VI_gradient_estimate(sample_mu_tree,ld_tree,cov_tree,x,mu_samp,index,Q,N,S,mu_var)
%sample_mu_tree= A tree structure that contains the current estimates of the means of the clusters
%cov_tree= A tree structure that contains the current estimates of the covariances of the clusters
%ld_tree= A tree strructure that contains the current estimates of the linear dynamics associated with each cluster
%x= latent states
%mu_samp= samples from variational distribution
%index= indicates what variational distribution (node) we are on
%Q= covairnace of state noise
%N= number of nodes in tree
%S= number of samples drawn from variational distirbution
%mu_var= mean of variational distribution
%cov_var= covariance of variational distribution

    T=length(x(1,:)); %Length of data
    dim=length(x(:,1)); %dimenison of latent space
    %Find Markov Blanket of current node i
    parent=sample_mu_tree.getparent(index+1);
    kids=sample_mu_tree.getchildren(index+1);
%     markov_blanket=[parent,kids];
    
   %Compute f and h
    for s=1:S
        logLike=0;
        %Compute log-likelihood
        for t=2:T
            w=0;
            x_temp=0;
            for n=1:N
                alpha=x(:,t-1)-mu_samp(:,s,n);
                w(n)=exp(-alpha'*inv(cov_tree.get(n+1))*alpha);
                Omega=ld_tree.get(n+1);
                x_temp=x_temp+w(n)*Omega*[x(:,t-1);1];
            end
            x_temp=x_temp/sum(w);
            logLike=logLike+log(mvnpdf(x(:,t)',x_temp',Q)+realmin);
        end
        
        %Compute prior markov blanket
        %Compute prior asociated with parent
        if parent==1
            logPrior=log(mvnpdf(mu_samp(:,s,index)',sample_mu_tree.get(parent)',cov_tree.get(parent)));
        else
            logPrior=log(mvnpdf(mu_samp(:,s,index)',mu_samp(:,s,parent-1)',cov_tree.get(parent)));
        end
        
        %Compute prior associated with children
        for j=1:length(kids)
            logPrior=logPrior+log(mvnpdf(mu_samp(:,s,kids(j)-1)',mu_samp(:,s,index)',cov_tree.get(index+1)));
        end
        
        %Combine likelihood and prior to get joint
        logJoint=logPrior+logLike;
        
        %Compute log of variational distribution
        logVar= log(mvnpdf(mu_samp(:,s,index)',mu_var{index}',cov_tree.get(index+1)));
        
        %Compute gradient of log of variational distribution
        h(:,s)=inv(cov_tree.get(index+1))*(mu_samp(:,s,index)-mu_var{index});
        
        %Compute f
        f(:,s)=h(:,s)*(logJoint-logVar);
    end
    
    %Compute a
    num=0;
    denom=0;
    for d=1:dim
        cov_est=mean((f(d,:)-mean(f(d,:))).*(h(d,:)-mean(h(d,:))));
        num=num+cov_est;
        
        denom=denom+var(h(d,:),1);
    end
    a=num/denom;
    
    grad_est=mean(f-a*h,2);
end