%will run series of test cases to see if learning and inference is relaible
%in proposed Gibbs/VI scheme. In this example, only thing unknown are the
%mean vectors of each of the clusters

clear all;close all;
load('synthetic_states.mat');
load('synthetic_obs.mat');
load('synthetic_mu_tree.mat');
load('synthetic_cov_tree.mat');
load('synthetic_ld_tree.mat');

%Will run Metropolis-Hastings on the conditionals where the proposal is
%constructed using Black Box VI 
M=1000; %Number of samples accepted
burnIn=1000; %Length of burn in period for markov chain
