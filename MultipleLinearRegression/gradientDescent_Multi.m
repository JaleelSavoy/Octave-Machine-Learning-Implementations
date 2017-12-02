function [theta, J_Results] = gradientDescent_Multi(X,y, theta, alpha, iterations)
  
  m = length(y); % number of training examples
  J_Results = zeros(iterations, 1);

  for i = 1:iterations

      gradJ = (1/(2*m)) * 2 * (X'*X*theta - X'*y);
      theta = theta - alpha * gradJ;

      J_Results(i)    = computeCost_Multi(X, y, theta);


  end

end