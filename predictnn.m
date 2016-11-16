function [p,J] = predictnn(Theta1, Theta2, X,y,lambda)

% predictnn applies the obtained parameters for neural networks to a test set of images in order to compute the cost function

m = size(X, 1);
num_labels = size(Theta2, 1);

p = zeros(size(X, 1), 1);

h1 = sigmoid([ones(m, 1) X] * Theta1');
h2 = sigmoid([ones(m, 1) h1] * Theta2');
[~, p] = max(h2, [], 2);
p(p == 2) = 0;

temp = zeros(size(y,1),1);
temp(y == 0) = 1;
y = [y,temp]; %melanomas are 1 in column 1, nevi are 1 in column 2
J=(1./m).*sum(sum(-y.*log(h2)-(1-y).*log(1-h2)));
J = J + lambda./(2.*m).*(sum(sum(Theta1(:,2:end).^2))+sum(sum(Theta2(:,2:end).^2)));


end
