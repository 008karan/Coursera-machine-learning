function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%
h= sigmoid(X * theta);
[m,n]= size(h);
for i=1:m
    for j=1:n
       j1(i,j)= (-y(i,j)*log(h(i,j)))-((1-y(i,j))*log(1-h(i,j)));
    end
end    
j1= sum(j1);
J=j1/m;
 
a=h-y;
x1=X(:,1)'*a;
x2=X(:,2)'*a;
x3=X(:,3)'*a;

grad(1,1)=x1/m;
grad(2,1)=x2/m;
grad(3,1)=x3/m;







% =============================================================

end
