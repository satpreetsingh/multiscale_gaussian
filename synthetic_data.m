%Generate data to test inference and learning in multi-scale gaussian example
%All vectors are column vectors unless stated otheriwse
clear all;close all;
addpath('@tree');

%Create a tree for storing the mean associated with each cluster
mu_tree=tree('root');
%Level 1
[mu_tree, mu_node1]=mu_tree.addnode(1,[-3;-3]);
[mu_tree, mu_node2]=mu_tree.addnode(1,[3;3]);
%Level 2
[mu_tree, mu_node3]=mu_tree.addnode(mu_node1,[-4; 0]);
[mu_tree, mu_node4]=mu_tree.addnode(mu_node1, [0; -4]);
[mu_tree, mu_node5]=mu_tree.addnode(mu_node2,[0; 2]);
[mu_tree, mu_node6]=mu_tree.addnode(mu_node2, [3; 0]);

nodes=[mu_node1,mu_node2,mu_node3,mu_node4,mu_node5,mu_node6];
%Create a tree for storing the covarinace associated with each cluster
cov_tree=tree('root');
[cov_tree, cov_node1]=cov_tree.addnode(1,3*eye(2));
[cov_tree, cov_node2]=cov_tree.addnode(1,4*eye(2));
%Level 2
[cov_tree, cov_node3]=cov_tree.addnode(cov_node1,eye(2));
[cov_tree, cov_node4]=cov_tree.addnode(cov_node1,eye(2));
[cov_tree, cov_node5]=cov_tree.addnode(cov_node2,0.7*eye(2));
[cov_tree, cov_node6]=cov_tree.addnode(cov_node2, 0.5*eye(2));

%Create a tree for storing the linear dynamics of each cluster
ld_tree=tree('root');
[ld_tree, ld_node1]=ld_tree.addnode(1,[cos(pi/15),-sin(pi/15),0;sin(pi/15),cos(pi/15),0]);
[ld_tree, ld_node2]=ld_tree.addnode(1,[cos(-pi/5),-sin(-pi/5),0;sin(-pi/5),cos(-pi/5),0]);

% [ld_tree, ld_node1]=ld_tree.addnode(1,[0,0,.5;0,0,0.5]);
% [ld_tree, ld_node2]=ld_tree.addnode(1,[0,0,-.5;0,0,-0.5]);
%Level 2
[ld_tree, ld_node3]=ld_tree.addnode(ld_node1,[1,0,-0.5;0,1,-0.5]);
[ld_tree, ld_node4]=ld_tree.addnode(ld_node1, [1,0,0.5;0,1,0.5]);
[ld_tree, ld_node5]=ld_tree.addnode(ld_node2,[1,0,2;0,1,2]);
[ld_tree, ld_node6]=ld_tree.addnode(ld_node2, [1,0,-0.5;0,1,0.5]);

% [ld_tree, ld_node3]=ld_tree.addnode(ld_node1,2*[cos(pi/10),-sin(pi/10),0;sin(pi/10),cos(pi/10),0]);
% [ld_tree, ld_node4]=ld_tree.addnode(ld_node1, 2*[cos(-pi/10),-sin(-pi/10),0;sin(-pi/10),cos(-pi/10),0]);
% [ld_tree, ld_node5]=ld_tree.addnode(ld_node2,2*[cos(pi/10),-sin(pi/10),0;sin(pi/10),cos(pi/10),0]);
% [ld_tree, ld_node6]=ld_tree.addnode(ld_node2, 2*[cos(-pi/10),-sin(-pi/10),0;sin(-pi/10),cos(-pi/10),0]);

%Set starting point
x(:,1)=[-5;1];
y(:,1)=x(:,1);
T=1500; %Length of data
scatter(x(1,1),x(2,1),'b');hold on;
for t=2:T
    %Compute Malahanobis distance between data point and clusters
    x_temp=0;
    w=0;
    for ii=1:6
%         w(ii)=mvnpdf(x(:,t-1)',mu_tree.get(nodes(ii))',cov_tree.get(nodes(ii)));
        w(ii)=exp(-(x(:,t-1)-mu_tree.get(nodes(ii)))'*inv(cov_tree.get(nodes(ii)))*(x(:,t-1)-mu_tree.get(nodes(ii))));
        x_temp=x_temp+w(ii)*ld_tree.get(nodes(ii))*[x(:,t-1);1];
    end
    w_norm=w/sum(w);
    
    x_temp=0;
    for ii=1:6
        x_temp=x_temp+w_norm(ii)*ld_tree.get(nodes(ii))*[x(:,t-1);1];
    end
%     x_temp=x_temp/sum(w);
    x(:,t)=x_temp+0*randn(2,1);
    scatter(x(1,t),x(2,t),'b');hold on;
    y(:,t-1)=x(:,t)+0.05*randn(2,1);
end


                                                                                                                           
x1 = -8:.2:8; x2 = -8:.2:8;
[X1,X2] = meshgrid(x1,x2);
for ii=1:6
    F =sqrt(det(2*pi*cov_tree.get(nodes(ii))))*mvnpdf([X1(:) X2(:)],mu_tree.get(nodes(ii))',cov_tree.get(nodes(ii)));
    F = reshape(F,length(x2),length(x1));
    figure(1);contour(x1,x2,F);hold on;
end

save('synthetic_states.mat','x');
save('synthetic_obs.mat','y');
save('synthetic_mu_tree.mat','mu_tree');
save('synthetic_cov_tree.mat','cov_tree');
save('synthetic_ld_tree.mat','ld_tree');
