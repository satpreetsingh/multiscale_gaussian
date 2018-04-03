%Compute the estimate of the gradient for the parents using the reparamterization trick
function R_grad_est=gradient_estimate_children(alpha_var,R_var,sample_mu_tree,cov_tree,ld_tree,index,x,P,N,S,Q,epsilon)
%alpha_var=Cell structure that contains the current estimate of alpha of
%R_var= Cell structure that contains the current estimate of R
%sample_mu_tree= A tree structure that contains the current estimates of the means of the clusters
%cov_tree= A tree structure that contains the current estimates of the covariances of the clusters
%ld_tree= A tree strructure that contains the current estimates of the linear dynamics associated with each cluster
%index= indicates what variational distribution (node) we are on
%x= latent states
%P= number of first generation parents
%Q= covairnace of state noise
%N= number of nodes in tree
%S= number of samples drawn from variational distirbution
%Q= Covariance of state noise
%epsilon= samples drawn from N(0,I). Will be used in estimation of gradient
grad_f_R=@(x,j,PI,e,v) 2*inv(cov_tree.get(index+1))*(x-alpha_var{PI}-...
    R_var{j}*e-R_var{PI}*v)*e';



    T=length(x(1,:)); %Length of data
    dim=length(x(:,1)); %dimenison of latent space

    %Iterate over all samples
    for s=1:S
        
        %Compute contribution from variational distribution
        grad_var_R=inv(R_var{index}'*R_var{index})*R_var{index}';
        
        %Compute contribution from prior distrbution
        grad_prior_R=grad_var_R;

        %Compute contribution from log-likelihood
        grad_like_R=0;
        for t=2:T
            x_temp=zeros(dim,N);
            w=0;
            grad_g_R=0;
            grad_Z_R=0;
            for n=1:N
                if n<=P
                    b=x(:,t-1)-alpha_var{n}-R_var{n}*epsilon(:,s,n);
                else
                    PI=sample_mu_tree.getparent(n+1)-1;
                    b=x(:,t-1)-alpha_var{PI}-R_var{PI}*epsilon(:,s,PI)-R_var{n}*epsilon(:,s,n);
                end
                w(n)=exp(-b'*inv(cov_tree.get(n+1))*b);
                x_temp(:,n)=w(n)*ld_tree.get(n+1)*[x(:,t-1);1];
                
                if n==index
                    grad_Z_R=grad_Z_R+grad_f_R(x(:,t-1),n,PI,epsilon(:,s,index),epsilon(:,s,PI));
                end
            end
            %Normalize w
            Z=sum(w); %Normalizing constant
            x_temp=x_temp/Z;
            w=w/Z;
            
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            for n=1:N
                if n==index
                    PI=sample_mu_tree.getparent(n+1)-1;
                    grad_g_R=grad_g_R+x_temp(:,n)'*inv(Q)*x(:,t)*...
                        grad_f_R(x(:,t-1),n,PI,epsilon(:,s,index),epsilon(:,s,PI));
                end
            end
            
            grad_g_R=grad_g_R/Z-grad_Z_R*(sum(x_temp,2)'*inv(Q)*x(:,t))/(Z^2);
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    

            grad_h_R=0;
            for ii=1:N
                for jj=1:N
                    cond_ii= (ii==index);
                    cond_jj= (jj==index);
                    PI=sample_mu_tree.getparent(n+1)-1;
                    grad_h_R=grad_h_R-Z^(-1)*w(ii)*w(jj)*x_temp(:,ii)'*...
                        inv(Q)*x_temp(jj)*(grad_f_R(x(:,t-1),ii,PI,epsilon(:,s,index),epsilon(:,s,PI))+...
                        grad_f_R(x(:,t-1),jj,PI,epsilon(:,s,index),epsilon(:,s,PI))+...
                        grad_Z_R/Z);
                end
            end
            
            grad_like_R=grad_like_R+grad_g_R-0.5*grad_h_R;
            
            R_grad(:,:,s)=-grad_var_R+grad_like_R+grad_prior_R;
        end
    end

    R_grad_est=mean(R_grad,2);
end