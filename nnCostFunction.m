function [J, grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
                                   
% nnCostFunction computes the cost function for Neural Networks

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

m = size(X, 1);
n = size(X, 2);
         
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

a1 = [ones(m,1) X];
z2 = a1 * Theta1';
a2 = [ones(m,1) sigmoid(z2)];
z3 = a2 * Theta2';
a3 = sigmoid(z3);
h = a3;
temp = zeros(size(y,1),1);
temp(y == 0) = 1;
y = [y,temp]; %melanomas are 1 in column 1, nevi are 1 in column 2
J=(1./m).*sum(sum(-y.*log(h)-(1-y).*log(1-h)));
J = J + lambda./(2.*m).*(sum(sum(Theta1(:,2:end).^2))+sum(sum(Theta2(:,2:end).^2)));

% -------------------------------------------------------------
d3 = a3 - y;
d2 = d3 * Theta2(:,2:end) .* sigmoidGradient(z2);
delta1 = d2' * a1;
delta2 = d3' * a2;
Theta1_grad = (1./m).*delta1;
Theta2_grad = (1./m).*delta2;
Theta1(:,1) = 0;
Theta2(:,1) = 0;
Theta1 = lambda./m.*Theta1;
Theta2 = lambda./m.*Theta2;
Theta1_grad = Theta1_grad + Theta1;
Theta2_grad = Theta2_grad + Theta2;

% =========================================================================

grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
