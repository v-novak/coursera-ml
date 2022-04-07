function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

m = length(y); % number of training examples

J = 0;
grad = zeros(size(theta));

h = sigmoid(X * theta);
J = (1 / m) * sum(-y .* log(h) - (1 .- y) .* log(1 .- h)) + sumsq(theta(2:end)) * 0.5 * lambda / m;

gradMask = ones(size(theta)) * lambda / m;
gradMask(1) = 0;

grad = (sum((h - y) .* X) / m); % idk why this didn't work: + theta .* gradMask;

% so i had to do this:
for j=2:size(theta, 1)
    grad(j) += theta(j) * lambda / m;
endfor

end
