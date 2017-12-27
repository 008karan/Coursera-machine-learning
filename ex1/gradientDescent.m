function [theta, J_history] = gradientDescent(X, y, theta, alpha, iterations)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y);
 % number of training examples
J_history = zeros(iterations, 1);

for iter = 1:iterations

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
h=X*theta;
c=zeros(2,1);
a=h-y;
x1=X(:,1)'*a;
x2=X(:,2)'*a;
x3=sum(x1)*alpha;
x4=sum(x2)*alpha;
theta_change1=x3/m;
theta_change2=x4/m;

theta(1,1) = theta(1,1) - theta_change1;
theta(2,1) = theta(2,1) - theta_change2;









    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
    

end

end
