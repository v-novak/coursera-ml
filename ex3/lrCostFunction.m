function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));


h = sigmoid(X * theta);
J = (1 / m) * sum(-y .* log(h) - (1 .- y) .* log(1 .- h)) ...
    + sumsq(theta(2:end)) * 0.5 * lambda / m;
grad = sum((h - y) .* X) / m;

regterm = lambda .* theta ./ m;
regterm(1) = 0;
grad += regterm';

grad = grad';
end
