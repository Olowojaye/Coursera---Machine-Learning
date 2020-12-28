function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
    temp0 = theta(1);
    temp1 = theta(2);
    for i = 1:m
        temp0 = temp0 - ((X(i, :)*theta-y(i)))*X(i, 1)*alpha/m;
        temp1 = temp1 - ((X(i, :)*theta-y(i)))*X(i, 2)*alpha/m;
    end
    theta(1) = temp0;
    theta(2) = temp1;
    J_history(iter) = computeCost(X, y, theta);
    fprintf('Iteration stage: %f\n', iter);
    fprintf('Current cost computed: %f\n', J_history(iter));
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    % ============================================================

    % Save the cost J in every iteration 
end
   
end

