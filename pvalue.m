function [M2,I,ptable,XProcessed,yProcessed] = pvalue(X,y)
% pvalue computes the p-values of the features in X and outputs XProcessed, which contains solely the statistically significant features
ptable = zeros(size(X,1),size(X,3));
melpos = intersect(find(y == 1),find(~isnan(y)));
mel = X(:,melpos,:);
nevpos = intersect(find(y == 0),find(~isnan(y)));
nev = X(:,nevpos,:);
for i_im = 1:size(X,3)
    for i_layer = 1:size(X,1)
        [~,p1] = ttest2(mel(i_layer,:,i_im), nev(i_layer,:,i_im));
        ptable(i_layer,i_im) = p1;
    end
end
[M,I] = min(ptable);
M2 = find(M <= 0.05);
I = I(M2);
XProcessed = zeros(1,size(X,2),length(M2));
for i_im = 1:length(M2)
    XProcessed(1,:,i_im) = X(I(i_im),:,M2(i_im));
end
XProcessed = squeeze(XProcessed);
yProcessed = y;
end
