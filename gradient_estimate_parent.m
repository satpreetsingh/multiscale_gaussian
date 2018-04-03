%Compute the estimate of the gradient for the parents using the reparamterization trick
function [alpha_grad_est,R_grad_est]=gradient_estimate_parent(alpha_var,R_var,sample_mu_tree,cov_tree,ld_tree,index,x,P,N,S,Q,epsilon)
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
grad_f_alpha=@(x,j,indic,e,v) 2*inv(cov_tree.get(index+1))*(x-alpha_var{index}-...
    indic*R_var{j}*v-R_var{index}*e);
grad_f_R=@(x,j,indic,e,v) 2*inv(cov_tree.get(index+1))*(x-alpha_var{index}-...
    indic*R_var{j}*v-R_var{index}*e)*e';



    T=length(x(1,:)); %Length of data
    dim=length(x(:,1)); %dimenison of latent space
    %Find Markov Blanket of current node i
    kids=sample_mu_tree.getchildren(index+1)-1;
    %Iterate over all samples
    for s=1:S
        %Compute contribution from prior distrbution
        grad_prior_alpha=inv(cov_tree.get(1))*(-alpha_var{index}-...
            R_var{index}*epsilon(:,s,index)+sample_mu_tree.get(1));
        
        grad_prior_R=grad_prior_alpha*epsilon(:,s,index)';
        
        %Computer contribution from variational distribution
        grad_var_alpha=0;
        grad_var_R=inv(R_var{index}'*R_var{index})*R_var{index}';
        

        %Compute contribution from log-likelihood
        grad_like_alpha=0;
        grad_like_R=0;
        for t=2:T
            x_temp=zeros(dim,N);
            w=0;
            grad_g_alpha=0;
            grad_g_R=0;
            grad_Z_alpha=0;
            grad_Z_R=0;
            for n=1:N
                PI=sample_mu_tree.getparent(n+1)-1;
                if n<=P
                    b=x(:,t-1)-alpha_var{n}-R_var{n}*epsilon(:,s,n);
                else
                    
                    b=x(:,t-1)-alpha_var{PI}-R_var{PI}*epsilon(:,s,PI)-R_var{n}*epsilon(:,s,n);
                end
                w(n)=exp(-b'*inv(cov_tree.get(n+1))*b);
                x_temp(:,n)=w(n)*ld_tree.get(n+1)*[x(:,t-1);1];
                if ismember(n,kids) || PI==index
                    grad_Z_alpha=grad_Z_alpha+grad_f_alpha(x(:,t-1),n,n>P,epsilon(:,s,index),epsilon(:,s,n));
                    grad_Z_R=grad_Z_R+grad_f_R(x(:,t-1),n,n>P,epsilon(:,s,index),epsilon(:,s,n));
                end
            end
            %Normalize w
            Z=sum(w); %Normalizing constant
            x_temp=x_temp/Z;
            w=w/Z;
            
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            for n=1:N
                PI=sample_mu_tree.getparent(n+1)-1;
                if n==index || PI==index
                    grad_g_alpha=grad_g_alpha+x_temp(:,n)'*inv(Q)*x(:,t)*...
                            grad_f_alpha(x(:,t-1),n,n>P,epsilon(:,s,index),epsilon(:,s,n));
                    grad_g_R=grad_g_R+x_temp(:,n)'*inv(Q)*x(:,t)*...
                        grad_f_R(x(:,t-1),n,n>P,epsilon(:,s,index),epsilon(:,s,n));
                end
            end
            
            grad_g_alpha=grad_g_alpha/Z-grad_Z_alpha*sum(x_temp,2)'*inv(Q)*x(:,t)/(Z^2);
            grad_g_R=grad_g_R/Z-grad_Z_R*(sum(x_temp,2)'*inv(Q)*x(:,t))/(Z^2);
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
                
            grad_h_alpha=0;
            grad_h_R=0;
            for ii=1:N
                for jj=1:N
                    PI_ii=sample_mu_tree.getparent(ii+1)-1;
                    PI_jj=sample_mu_tree.getparent(jj+1)-1;
                    cond_ii= (ii==index) || (PI_ii==index);
                    cond_jj= (jj==index) || (PI_jj==index);
                    grad_h_alpha=grad_h_alpha-Z^(-1)*w(ii)*w(jj)*x_temp(:,ii)'*...
                        inv(Q)*x_temp(jj)*(cond_ii*grad_f_alpha(x(:,t-1),ii,ii>P,epsilon(:,s,index),epsilon(:,s,ii))+...
                        cond_jj*grad_f_alpha(x(:,t-1),jj,jj>P,epsilon(:,s,index),epsilon(:,s,jj))+...
                        grad_Z_alpha/Z);
                    
                    grad_h_R=grad_h_R-Z^(-1)*w(ii)*w(jj)*x_temp(:,ii)'*...
                        inv(Q)*x_temp(jj)*(cond_ii*grad_f_R(x(:,t-1),ii,ii>P,epsilon(:,s,index),epsilon(:,s,ii))+...
                        cond_jj*grad_f_R(x(:,t-1),jj,jj>P,epsilon(:,s,index),epsilon(:,s,jj))+...
                        grad_Z_R/Z);
                end
            end
            
            grad_like_alpha=grad_like_alpha+grad_g_alpha-0.5*grad_h_alpha;
            grad_like_R=grad_like_R+grad_g_R-0.5*grad_h_R;
            
            alpha_grad(:,s)=-grad_var_alpha+grad_like_alpha+grad_prior_alpha;
            R_grad(:,:,s)=-grad_var_R+grad_like_R+grad_prior_R;
        end
    end
    
    alpha_grad_est=mean(alpha_grad,2);
    R_grad_est=mean(R_grad,2);
end