function [Jcost, grad] = costFunction(theta, X, y)
  m = length(y);
  Jcost = (1./m) * (-y' * log(sigmoid(X*theta)) - (1 - y') * ...
            log(1 - sigmoid(X*theta))); 
  grad = (1./m) *((sigmoid(X*theta)) - y)' * X;
end  