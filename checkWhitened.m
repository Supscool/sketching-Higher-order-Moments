clear all
clc;

for j = 1:10

documents = 10000; vocab = 5; topic = 4;
tSparsity = 0.25; dSparsity = 0.05;
L = 10; iteration = 20; num = 20;
sketchSize1 = 100; sketchSize2 = 500; sketchSize3 = 1000;

[samples,M2,M3,mu,w,wM3,W_true,tM3] = genDoc(documents,vocab,topic,tSparsity,dSparsity);
% [M3t,M3d] = tensorPowerIteration(documents,vocab,samples,M2,topic,L,iteration);
[swM3,sM2,sM3] = sketchedTPI(documents,vocab,samples,topic,L,iteration,sketchSize1);
a_true = W_true'*mu;

for i = 1:topic
    temp3 = tmprod(wM3,a_true(:,i)',3);
    temp2 = tmprod(temp3,a_true(:,i)',2);
    t(i) = tmprod(temp2,a_true(:,i)',1);
    temp3 = tmprod(swM3,a_true(:,i)',3);
    temp2 = tmprod(temp3,a_true(:,i)',2);
    s(i) = tmprod(temp2,a_true(:,i)',1);
end

r(j) = frob(sM3)/frob(M3);


end