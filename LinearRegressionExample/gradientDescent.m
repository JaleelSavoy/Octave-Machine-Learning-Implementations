function [theta, J_Results] = gradientDescent(X,y,theta,alpha, num_iters)
  m = length(y);
  J_Results = zeros(num_iters, 1);
  
  for iter = 1:num_iters
    theta = theta - (alpha * (1/m) * ((theta'*X')' - y)' * X)';    
  
    %store the cost J from every iterations
    J_Results(iter) = computeCost(X,y,theta);
  end
  
end