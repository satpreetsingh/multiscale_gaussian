%Compute the estimate of the gradient for the parents using the reparamterization trick
function [alpha_grad_est,R_grad_est]=gradient_est_MF(alpha_var,R_var,sample_mu_tree,cov_tree,ld_tree,index,x,N,S,Q,epsilon)
%alpha_var=Cell structure that contains the current estimate of alpha of
%R_var= Cell structure that contains the current estimate of R
%sample_mu_tree= A tree structure that contains the current estimates of the means of the clusters
%cov_tree= A tree structure that contains the current estimates of the covariances of the clusters
%ld_tree= A tree strructure that contains the current estimates of the linear dynamics associated with each cluster
%index= indicates what variational distribution (node) we are on
%x= latent states
%Q= covairnace of state noise
%N= number of nodes in tree
%S= number of samples drawn from variational distirbution
%Q= Covariance of state noise
%epsilon= samples drawn from N(0,I). Will be used in estimation of gradient

weight_calc= @(x,alpha,R,e) exp(-(x-alpha-R*e)'*inv(R*R')*(x-alpha-R*e)); 


    T=length(x(1,:)); %Length of data
    dim=length(x(:,1)); %dimenison of latent space
    %Find out who the parent is
    parent=sample_mu_tree.getparent(index+1);
    %Find out it's kids
    kids=sample_mu_tree.getchildren(index+1);
    
    %Iterate over all samples
    for s=1:S
        %If the parent is equal to 1 then it is a first generation node
        if parent==1
        %Compute contribution from parent
            grad_prior_alpha=-inv(cov_tree.get(1))*(alpha_var{index}+...
                R_var{index}*epsilon(:,s,index)-sample_mu_tree.get(1));
            grad_prior_R=grad_prior_alpha*epsilon(:,s,index)';
        else
            grad_prior_alpha=-inv(cov_tree.get(parent))*(alpha_var{index}+...
                R_var{index}*epsilon(:,s,index)-alpha_var{parent-1}-...
                R_var{parent-1}*epsilon(:,s,parent-1));
            grad_prior_R=grad_prior_alpha*epsilon(:,s,index)';
        end
        
        %Compute contribution from kids
        for kid=1:length(kids)
            beta=alpha_var{kids(kid)-1};
            L=R_var{kids(kid)-1};
            v=epsilon(:,s,kids(kid)-1);
            grad_kid_alpha=inv(cov_tree.get(index+1))*(beta+L*v-...
                alpha_var{index}-R_var{index}*epsilon(:,s,index));
            grad_kid_R=grad_kid_alpha*epsilon(:,s,index)';

            grad_prior_alpha=grad_prior_alpha+grad_kid_alpha;
            grad_prior_R=grad_prior_R+grad_kid_R;

        end
       
        
        
        %Compute contribution from variational distribution
        grad_var_alpha=0;
        grad_var_R=-inv(R_var{index}*R_var{index}')*R_var{index};
        

        %Compute contribution from log-likelihood
        grad_like_alpha=0;
        grad_like_R=0;
        for t=2:T
            x_temp=zeros(dim,N);
            w=0;
            grad_w_alpha=0;
            grad_w_R=0;
            for n=1:N
                %Calculate weights
                w(n)=weight_calc(x(:,t-1),alpha_var{index},R_var{index},epsilon(:,s,index));
                %Calculate contribtuion to system dynamics from cluster n
                x_temp(:,n)=w(n)*ld_tree.get(n+1)*[x(:,t-1);1];        
            end
            %Normalize w
            Z=sum(w); %Normalizing constant
            
            %Calculate weight normalized sum
            m_t=sum(x_temp,2)/Z;
            
            %Compute gradient of w (gradient of w is equal t gradient of Z
            %because of MF approach
            b=x(:,t-1)-alpha_var{index}-R_var{index}*epsilon(:,s,index);
            
            grad_w_alpha=2*w(index)*inv(R_var{index}*R_var{index}')*b;
            grad_w_R=grad_w_alpha*epsilon(:,s,index)'+grad_w_alpha*(b'*...
                inv(R_var{index}*R_var{index}')*R_var{index});
            
            grad_h_alpha=-Z^(-2)*(m_t'*inv(Q)*x(:,t)-0.5*m_t'*inv(Q)*m_t)*grad_w_alpha;
            grad_h_R=-Z^(-2)*(m_t'*inv(Q)*x(:,t)-0.5*m_t'*inv(Q)*m_t)*grad_w_R;
  
            grad_g_alpha=grad_w_alpha*([x(:,t-1);1]'*ld_tree.get(index+1)'*inv(Q)*x(:,t));
            grad_g_alpha=grad_g_alpha-2*(x_temp(:,index)'*inv(Q)*ld_tree.get(index+1)*[x(:,t-1);1])*grad_w_alpha;
            temp=x_temp;
            temp(:,index)=[];
            grad_g_alpha=grad_g_alpha-2*(sum(temp,2)'*inv(Q)*ld_tree.get(index+1)*[x(:,t-1);1])*grad_w_alpha;
            grad_g_alpha=grad_g_alpha/Z;
            
            grad_g_R=grad_w_R*([x(:,t-1);1]'*ld_tree.get(index+1)'*inv(Q)*x(:,t));
            grad_g_R=grad_g_R-2*(x_temp(:,index)'*inv(Q)*ld_tree.get(index+1)*[x(:,t-1);1])*grad_w_R;
            temp=x_temp;
            temp(:,index)=[];
            grad_g_R=grad_g_R-2*(sum(temp,2)'*inv(Q)*ld_tree.get(index+1)*[x(:,t-1);1])*grad_w_R;
            grad_g_R=grad_g_R/Z;
            
            %Compute the likelihood 
            grad_like_alpha=grad_like_alpha+grad_g_alpha+grad_h_alpha;
            grad_like_R=grad_like_R+grad_g_R+grad_h_R;
        end
        alpha_grad(:,s)=-grad_var_alpha+grad_like_alpha+grad_prior_alpha;
        R_grad(:,:,s)=-grad_var_R+grad_like_R+grad_prior_R;
    end
    
    alpha_grad_est=mean(alpha_grad,2);
    R_grad_est=mean(R_grad,3);
end