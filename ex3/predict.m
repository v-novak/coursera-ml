function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

X = [ones(size(X, 1), 1) X];
h = sigmoid(X * Theta1');
h = sigmoid([ones(size(h, 1), 1) h] * Theta2');
[_, p] = max(h, [], 2);

end
