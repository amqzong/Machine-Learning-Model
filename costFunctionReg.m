function [J, grad] = costFunctionReg(theta, X, y, lambda)
% Compute cost and gradient for logistic regression with regularization

m = length(y); % number of training examples
X = [ones(m, 1), X];
h=sigmoid(X*theta);
theta(1)=0;
J=(1./m).*(sum(-y.*log(h)-(1-y).*log(1-h)))+lambda./(2.*m).*sum(theta.^2);
grad=(1./m).*(transpose(X)*(h-y))+lambda./m.*theta;

end
