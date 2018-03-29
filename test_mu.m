%will run series of test cases to see if learning and inference is relaible
%in proposed Gibbs/VI scheme. In this example, only thing unknown are the
%mean vectors of each of the clusters
clear all;close all;
load('synthetic_states.mat');
load('synthetic_obs.mat');
load('synthetic_mu_tree.mat');
load('synthetic_cov_tree.mat');
load('synthetic_ld_tree.mat');
addpath('@tree');

layers=2; %Number of layers used in model
K=2; %Number of children each parent node has
mu_prior=[0;0];
Sigma_prior=3*eye(2);

%Calculate number of gaussians used in model
num_of_nodes=0;
for ii=1:layers
    num_of_nodes=num_of_nodes+K^(ii);
end

% Create cell structure that will store pointers to nodes
mu_nodes=cell(layers,1);
start=2;
for ii=1:layers
    fin=K^ii-1;
    mu_nodes{ii}=start:start+fin;
    start=start+fin+1;
end


sample_mu_tree=tree('root');
for ii=1:layers
    mu_index=1;
    t_index=1;
   for j=1:K^ii
       
       if ii==1
           c=cell(2,1);
           S=cov_tree.get(mu_nodes{ii}(j));
           c{1}=mvnrnd(mu_prior,S)';
           [sample_mu_tree, ~]=sample_mu_tree.addnode(1,c);
       else
           m=sample_mu_tree.get(mu_nodes{ii-1}(mu_index));
           S=cov_tree.get(mu_nodes{ii}(j));
           c=cell(2,1);
           c{1}=mvnrnd(m{1}',S);
           [sample_mu_tree, ~]=sample_mu_tree.addnode(mu_nodes{ii-1}(mu_index),c);
           t_index=t_index+1;
           if t_index>K
               t_index=1;
               mu_index=mu_index+1;
           end
       end
   end
end
clear t_index; clear mu_index;clear start; clear fin;clear m;clear S;clear c;
disp(sample_mu_tree.tostring)
%Will run Metropolis-Hastings on the conditionals where the proposal is
%constructed using Black Box VI 
M=1000; %Number of samples accepted
burnIn=1000; %Length of burn in period for markov chain

