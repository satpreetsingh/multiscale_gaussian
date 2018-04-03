%Test case to see to make sure reparamterization trick
clear all;close all;

N=100; %Number of data points drawn 
mu_true=5*ones(2,1); 
data_Cov=2*eye(2); %Covariance of likelihood
mu_prior=-3*ones(2,1); %mean of prior
prior_Cov= 4*eye(2); %Covariance of prior

posterior_cov=inv(N*inv(data_Cov)+inv(prior_Cov));
posterior_mu=(posterior_cov)*(N*inv(data_Cov)*mean(data,2)+inv(prior_Cov)*mu_prior);
%Generate data
data=mvnrnd(mu_true',data_Cov,N)';

%Start of Stochastic VI
T=500; %Number of iterations VI will run for 
alpha_est=[-3;-4]; %Initalize value for mu 
R_est=10*eye(2);
M=100; %Number of samples drawn from e~N(0,I)

%let q(mu)~N(alpha,RR'). Find alpha and R using SGD
for t=1:T
    %Draw M samples from N(0,I)
    e=randn(2,M);
    
    %Compute estimate of gradient 
    alpha_grad=zeros(2,M);
    R_grad=zeros(2,2,M);
    
    for m=1:M
        %Compute likelihood term
        alpha_like=0;
        R_like=0;
        for n=1:N
            temp=inv(data_Cov)*(data(:,n)-alpha_est-R_est*e(:,m));
            alpha_like=alpha_like+temp;
            R_like=R_like+temp*e(:,m)';
        end
        
        %Compute prior term
        temp=inv(prior_Cov)*(-alpha_est-R_est*e(:,m)+mu_prior);
        alpha_prior=temp;
        R_prior=temp*e(:,m)';
        
        %Compute variational term
        alpha_var=zeros(2,1);
        R_var=-inv(R_est'*R_est)*R_est';
        
        %Compute sample gradient
        alpha_grad(:,m)=-alpha_var+alpha_like+alpha_prior;
        R_grad(:,:,m)=-R_var+R_like+R_prior;
    end
    
    %Estimate of gradient
    alpha_grad=mean(alpha_grad,2);
    R_grad=mean(R_grad,3);
    
    %Update estimate of alpha and R
    rho=1/t;
    alpha_est=alpha_est+rho*alpha_grad;
    R_est=R_est+rho*R_grad;
end