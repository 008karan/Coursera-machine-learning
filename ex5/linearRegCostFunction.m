function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values

  
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
[p,l]=size(theta);
k=lambda/m;
h = X * theta;
[m,n]= size(h);
a=h-y;

j1=a.^2;
j1=sum(j1);
j1=j1/(2*m);
t1=theta.^2;
t2=t1(2:p,:);
t2=sum(t2)*k;
t=t2/2;
J=j1+t;



grad=((X'*a)/m) + (k.*theta);
 

g=X(:,1)'*a;
grad(1,1)=g/m;











% =========================================================================

grad = grad(:);

end
