function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
    h_theta = X * theta
    delta = h_theta - y
    theta = theta - alpha * sum(X .* delta, 1)' / m
    J_history(iter) = computeCost(X, y, theta);
end

end
