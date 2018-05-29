function [swM3,sM2,sM3] = sketchedTPI(documents, vocab, x_sample, topic, L, iteration, sketchSize)
tic
for i = 1:sketchSize
    for j = 1:documents
        x = rand();
        if x <= 2/3
            sketchMatrix(i,j) = -1/(2*sketchSize)^(1/3);
        else
            sketchMatrix(i,j) = 2/(2*sketchSize)^(1/3);
        end
    end
end

for i = 1:sketchSize
    for j = 1:documents
        x = rand();
        if x <= 2/3
            sMatrix(i,j) = -1/(2*sketchSize)^(1/3)*(1/sketchSize)^(1/3);
        else
            sMatrix(i,j) = 2/(2*sketchSize)^(1/3)*(1/sketchSize)^(1/3);
        end
    end
end

x1_samples = x_sample(:,:,1);
x2_samples = x_sample(:,:,2);
x3_samples = x_sample(:,:,3);
x4_samples = x_sample(:,:,4);
x5_samples = x_sample(:,:,5);

sx1_samples = sketchMatrix*x1_samples;
mx1_samples = sMatrix*x1_samples;
clear x1_samples
sx2_samples = sketchMatrix*x2_samples;
mx2_samples = sMatrix*x2_samples;
clear x2_samples
sx3_samples = sketchMatrix*x3_samples;
mx3_samples = sMatrix*x3_samples;
clear x3_samples
sx4_samples = sketchMatrix*x4_samples;
mx4_samples = sMatrix*x4_samples;
clear x4_samples
sx5_samples = sketchMatrix*x5_samples;
mx5_samples = sMatrix*x5_samples;
clear x5_samples
clear sketchMatrix


% fprintf('The second order moment: ');
%%% one way of computing Whiteinig Matrix: W = U D^(-0.5)
%%% U \in R^{d \times k} is the matrix of orthonormal eigenvectors of M2
%%% D \in R^{k \times k} is the diagonal matrix of positive eigenvalues of M2

sM2 = zeros(vocab);
for i = 1:sketchSize
    sM2 = sM2 + sx1_samples(i,:)'*sx2_samples(i,:)+sx2_samples(i,:)'*sx1_samples(i,:)+...
        sx1_samples(i,:)'*sx3_samples(i,:)+sx3_samples(i,:)'*sx1_samples(i,:)+...
        sx1_samples(i,:)'*sx4_samples(i,:)+sx4_samples(i,:)'*sx1_samples(i,:)+...
        sx1_samples(i,:)'*sx5_samples(i,:)+sx5_samples(i,:)'*sx1_samples(i,:)+...
        sx2_samples(i,:)'*sx3_samples(i,:)+sx3_samples(i,:)'*sx2_samples(i,:)+...
        sx2_samples(i,:)'*sx4_samples(i,:)+sx4_samples(i,:)'*sx2_samples(i,:)+...
        sx2_samples(i,:)'*sx5_samples(i,:)+sx5_samples(i,:)'*sx2_samples(i,:)+...
        sx3_samples(i,:)'*sx4_samples(i,:)+sx4_samples(i,:)'*sx3_samples(i,:)+...
        sx3_samples(i,:)'*sx5_samples(i,:)+sx5_samples(i,:)'*sx3_samples(i,:)+...
        sx4_samples(i,:)'*sx5_samples(i,:)+sx5_samples(i,:)'*sx4_samples(i,:);
end
sM2 = sM2/(20*documents)*1/(2/(2)^(2/3)*(sketchSize^(1/3)));

% fprintf('The second order moment singular values:'); Lw;
[sUw, sLw, sVw] = svd(sM2);
% sLw = sLw*1/(2/(2)^(2/3)*(sketchSize^(1/3)));
sW = sVw(:,1:topic)* sqrt(pinv(sLw(1:topic,1:topic)));
%%%
%%%                                       n
%%% Third Order Moment of raw data M3 = \sum x1(t,:) \otimes x2(t,:) \otimes x3(t,:) 
%%%                                      t=1
%%% (3a) Whiten the data
%%%


sy1 = sx1_samples * sW; %%% W^T*x1
sy2 = sx2_samples * sW; %%% W^T*x2
sy3 = sx3_samples * sW; %%% W^T*x3
sy4 = sx4_samples * sW; %%% W^T*x2
sy5 = sx5_samples * sW; %%% W^T*x3

%%%                                       n
%%% (4) Moments: Third Order Moment T = \sum y1(t,:) \otimes y2(t,:) \otimes y3(t,:)
%%%                                      t=1
%%% Which is equal to T = M3(W,W,W)
sM3d = zeros(topic,topic,topic);
sM3 = zeros(vocab,vocab,vocab);
for t = 1 : sketchSize
    sM3d = sM3d + outprod(sy1(t,:),sy2(t,:),sy3(t,:)) + outprod(sy1(t,:),sy3(t,:),sy2(t,:))...
        + outprod(sy2(t,:),sy1(t,:),sy3(t,:))+outprod(sy2(t,:),sy3(t,:),sy1(t,:))...
        +outprod(sy3(t,:),sy1(t,:),sy2(t,:))+outprod(sy3(t,:),sy2(t,:),sy1(t,:))+...
        outprod(sy1(t,:),sy2(t,:),sy4(t,:)) + outprod(sy1(t,:),sy4(t,:),sy2(t,:))...
        + outprod(sy2(t,:),sy1(t,:),sy4(t,:))+outprod(sy2(t,:),sy4(t,:),sy1(t,:))...
        +outprod(sy4(t,:),sy1(t,:),sy2(t,:))+outprod(sy4(t,:),sy2(t,:),sy1(t,:))+...
        outprod(sy1(t,:),sy2(t,:),sy5(t,:)) + outprod(sy1(t,:),sy5(t,:),sy2(t,:))...
        + outprod(sy2(t,:),sy1(t,:),sy5(t,:))+outprod(sy2(t,:),sy5(t,:),sy1(t,:))...
        +outprod(sy5(t,:),sy1(t,:),sy2(t,:))+outprod(sy5(t,:),sy2(t,:),sy1(t,:))+...
        outprod(sy1(t,:),sy4(t,:),sy3(t,:)) + outprod(sy1(t,:),sy3(t,:),sy4(t,:))...
        + outprod(sy4(t,:),sy1(t,:),sy3(t,:))+outprod(sy4(t,:),sy3(t,:),sy1(t,:))...
        +outprod(sy3(t,:),sy1(t,:),sy4(t,:))+outprod(sy3(t,:),sy4(t,:),sy1(t,:))+...
        outprod(sy1(t,:),sy5(t,:),sy3(t,:)) + outprod(sy1(t,:),sy3(t,:),sy5(t,:))...
        + outprod(sy5(t,:),sy1(t,:),sy3(t,:))+outprod(sy5(t,:),sy3(t,:),sy1(t,:))...
        +outprod(sy3(t,:),sy1(t,:),sy5(t,:))+outprod(sy3(t,:),sy5(t,:),sy1(t,:))+...
        outprod(sy1(t,:),sy4(t,:),sy5(t,:)) + outprod(sy1(t,:),sy5(t,:),sy4(t,:))...
        + outprod(sy4(t,:),sy1(t,:),sy5(t,:))+outprod(sy4(t,:),sy5(t,:),sy1(t,:))...
        +outprod(sy5(t,:),sy1(t,:),sy4(t,:))+outprod(sy5(t,:),sy4(t,:),sy1(t,:))+...
        outprod(sy4(t,:),sy2(t,:),sy3(t,:)) + outprod(sy4(t,:),sy3(t,:),sy2(t,:))...
        + outprod(sy2(t,:),sy4(t,:),sy3(t,:))+outprod(sy2(t,:),sy3(t,:),sy4(t,:))...
        +outprod(sy3(t,:),sy4(t,:),sy2(t,:))+outprod(sy3(t,:),sy2(t,:),sy4(t,:))+...
        outprod(sy5(t,:),sy2(t,:),sy3(t,:)) + outprod(sy5(t,:),sy3(t,:),sy2(t,:))...
        + outprod(sy2(t,:),sy5(t,:),sy3(t,:))+outprod(sy2(t,:),sy3(t,:),sy5(t,:))...
        +outprod(sy3(t,:),sy5(t,:),sy2(t,:))+outprod(sy3(t,:),sy2(t,:),sy5(t,:))+...
        outprod(sy5(t,:),sy2(t,:),sy4(t,:)) + outprod(sy5(t,:),sy4(t,:),sy2(t,:))...
        + outprod(sy2(t,:),sy5(t,:),sy4(t,:))+outprod(sy2(t,:),sy4(t,:),sy5(t,:))...
        +outprod(sy4(t,:),sy5(t,:),sy2(t,:))+outprod(sy4(t,:),sy2(t,:),sy5(t,:))+...
        outprod(sy5(t,:),sy4(t,:),sy3(t,:)) + outprod(sy5(t,:),sy3(t,:),sy4(t,:))...
        + outprod(sy4(t,:),sy5(t,:),sy3(t,:))+outprod(sy4(t,:),sy3(t,:),sy5(t,:))...
        +outprod(sy3(t,:),sy5(t,:),sy4(t,:))+outprod(sy3(t,:),sy4(t,:),sy5(t,:));
    
    sM3 = sM3 + outprod(sx1_samples(t,:),sx2_samples(t,:),sx3_samples(t,:)) + outprod(sx1_samples(t,:),sx3_samples(t,:),sx2_samples(t,:))...
        + outprod(sx2_samples(t,:),sx1_samples(t,:),sx3_samples(t,:))+outprod(sx2_samples(t,:),sx3_samples(t,:),sx1_samples(t,:))...
        +outprod(sx3_samples(t,:),sx1_samples(t,:),sx2_samples(t,:))+outprod(sx3_samples(t,:),sx2_samples(t,:),sx1_samples(t,:))+...
        outprod(sx1_samples(t,:),sx2_samples(t,:),sx4_samples(t,:)) + outprod(sx1_samples(t,:),sx4_samples(t,:),sx2_samples(t,:))...
        + outprod(sx2_samples(t,:),sx1_samples(t,:),sx4_samples(t,:))+outprod(sx2_samples(t,:),sx4_samples(t,:),sx1_samples(t,:))...
        +outprod(sx4_samples(t,:),sx1_samples(t,:),sx2_samples(t,:))+outprod(sx4_samples(t,:),sx2_samples(t,:),sx1_samples(t,:))+...
        outprod(sx1_samples(t,:),sx2_samples(t,:),sx5_samples(t,:)) + outprod(sx1_samples(t,:),sx5_samples(t,:),sx2_samples(t,:))...
        + outprod(sx2_samples(t,:),sx1_samples(t,:),sx5_samples(t,:))+outprod(sx2_samples(t,:),sx5_samples(t,:),sx1_samples(t,:))...
        +outprod(sx5_samples(t,:),sx1_samples(t,:),sx2_samples(t,:))+outprod(sx5_samples(t,:),sx2_samples(t,:),sx1_samples(t,:))+...
        outprod(sx1_samples(t,:),sx4_samples(t,:),sx3_samples(t,:)) + outprod(sx1_samples(t,:),sx3_samples(t,:),sx4_samples(t,:))...
        + outprod(sx4_samples(t,:),sx1_samples(t,:),sx3_samples(t,:))+outprod(sx4_samples(t,:),sx3_samples(t,:),sx1_samples(t,:))...
        +outprod(sx3_samples(t,:),sx1_samples(t,:),sx4_samples(t,:))+outprod(sx3_samples(t,:),sx4_samples(t,:),sx1_samples(t,:))+...
        outprod(sx1_samples(t,:),sx5_samples(t,:),sx3_samples(t,:)) + outprod(sx1_samples(t,:),sx3_samples(t,:),sx5_samples(t,:))...
        + outprod(sx5_samples(t,:),sx1_samples(t,:),sx3_samples(t,:))+outprod(sx5_samples(t,:),sx3_samples(t,:),sx1_samples(t,:))...
        +outprod(sx3_samples(t,:),sx1_samples(t,:),sx5_samples(t,:))+outprod(sx3_samples(t,:),sx5_samples(t,:),sx1_samples(t,:))+...
        outprod(sx1_samples(t,:),sx4_samples(t,:),sx5_samples(t,:)) + outprod(sx1_samples(t,:),sx5_samples(t,:),sx4_samples(t,:))...
        + outprod(sx4_samples(t,:),sx1_samples(t,:),sx5_samples(t,:))+outprod(sx4_samples(t,:),sx5_samples(t,:),sx1_samples(t,:))...
        +outprod(sx5_samples(t,:),sx1_samples(t,:),sx4_samples(t,:))+outprod(sx5_samples(t,:),sx4_samples(t,:),sx1_samples(t,:))+...
        outprod(sx4_samples(t,:),sx2_samples(t,:),sx3_samples(t,:)) + outprod(sx4_samples(t,:),sx3_samples(t,:),sx2_samples(t,:))...
        + outprod(sx2_samples(t,:),sx4_samples(t,:),sx3_samples(t,:))+outprod(sx2_samples(t,:),sx3_samples(t,:),sx4_samples(t,:))...
        +outprod(sx3_samples(t,:),sx4_samples(t,:),sx2_samples(t,:))+outprod(sx3_samples(t,:),sx2_samples(t,:),sx4_samples(t,:))+...
        outprod(sx5_samples(t,:),sx2_samples(t,:),sx3_samples(t,:)) + outprod(sx5_samples(t,:),sx3_samples(t,:),sx2_samples(t,:))...
        + outprod(sx2_samples(t,:),sx5_samples(t,:),sx3_samples(t,:))+outprod(sx2_samples(t,:),sx3_samples(t,:),sx5_samples(t,:))...
        +outprod(sx3_samples(t,:),sx5_samples(t,:),sx2_samples(t,:))+outprod(sx3_samples(t,:),sx2_samples(t,:),sx5_samples(t,:))+...
        outprod(sx5_samples(t,:),sx2_samples(t,:),sx4_samples(t,:)) + outprod(sx5_samples(t,:),sx4_samples(t,:),sx2_samples(t,:))...
        + outprod(sx2_samples(t,:),sx5_samples(t,:),sx4_samples(t,:))+outprod(sx2_samples(t,:),sx4_samples(t,:),sx5_samples(t,:))...
        +outprod(sx4_samples(t,:),sx5_samples(t,:),sx2_samples(t,:))+outprod(sx4_samples(t,:),sx2_samples(t,:),sx5_samples(t,:))+...
        outprod(sx5_samples(t,:),sx4_samples(t,:),sx3_samples(t,:)) + outprod(sx5_samples(t,:),sx3_samples(t,:),sx4_samples(t,:))...
        + outprod(sx4_samples(t,:),sx5_samples(t,:),sx3_samples(t,:))+outprod(sx4_samples(t,:),sx3_samples(t,:),sx5_samples(t,:))...
        +outprod(sx3_samples(t,:),sx5_samples(t,:),sx4_samples(t,:))+outprod(sx3_samples(t,:),sx4_samples(t,:),sx5_samples(t,:));
end

% fprintf('The third order moment: ');
sM3d = sM3d / (60*documents);  %% Empirical Average 
%%%                             k
%%% (5) Power Iteration:  T = \sum \lambda_i V(:,i) \otimes V(:,i) \otimes V(:,i)
%%%                            i=1
sM3 = sM3/(60*documents);
swM3 = sM3d;

for i = 1:topic
    sV_old = rand(topic,L);
    for j = 1:L
        sV_old(:,j) = sV_old(:,j)./norm(sV_old(:,j),2);
        t = sV_old(:,j);
        for k = 1:iteration
            temp3 = tmprod(sM3d,t',3);
            temp2 = tmprod(temp3,t',2);
            t = temp2./norm(temp2,2);
            temp1 = tmprod(sM3d,t',1);
            temp2 = tmprod(temp1,t',2);
            temp3 = tmprod(temp2,t',3);
            if frob(sM3d - temp3*outprod(t,t,t)) < frob(sM3d)
                spickV(:,j) = t;
                spickLambda(j) = temp3;
            end
        end
    end
    [value, index] = max(spickLambda);
    t = spickV(:,index);
    for k = 1:iteration
        temp3 = tmprod(sM3d,t',3);
        temp2 = tmprod(temp3,t',2);
        t = temp2./norm(temp2,2);
        temp1 = tmprod(sM3d,t',1);
        temp2 = tmprod(temp1,t',2);
        temp3 = tmprod(temp2,t',3);
        if frob(sM3d - temp3*outprod(t,t,t)) < frob(sM3d)
            sV_est(:,i) = t;
            slambda_est(i) = temp3;
        end
    end
    sM3d = sM3d - slambda_est(i)*outprod(sV_est(:,i),sV_est(:,i),sV_est(:,i));
end

sA_est = pinv(sW')*(sV_est*diag(slambda_est));

for i = 1:topic
    checkSign = sA_est(:,i);
    countP = 0;
    countN = 0;
    for j = 1:vocab
        if checkSign(j,1) < 0
            countN = countN + (checkSign(j,1))^2;
        else
            countP = countP + (checkSign(j,1))^2;
        end
    end
    if countN > countP
        checkSign = checkSign.*(-1);
    end
    for j = 1:vocab
        if sign(checkSign(j,1)) < 0
            checkSign(j,1) = checkSign(j,1)*(0);
        end
    end
    checkSign = checkSign./sum(checkSign);
    sA_est(:,i) = checkSign;
end

% original = 1:topic;
% estimate= original;
% for i = 1:topic
%     dif = realmax;
%     for j = 1:topic
%         temp = norm(A_true(:,i) - sA_est(:,i),1);
%         if temp < dif && (original(i) ~= 0 && estimate(j) ~= 0)
%             dif = temp;
%             smatch(i,1) = i;
%             smatch(i,2) = j;
%             smatch(i,3) = dif;
%         end
%     end
%     original(i) = 0;
%     estimate(1,smatch(i,2)) = 0;
% end

slambda = diag(diag(slambda_est)^(-2));
slambda = slambda./sum(slambda);
toc





M3s = zeros(vocab,vocab,vocab);
for i = 1:topic
    M3s = M3s + slambda(i)*outprod(sA_est(:,i),sA_est(:,i),sA_est(:,i));    
end