function J = computeCost(X,y,theta)
  m = length(y);     %number of training examples
  J = 0;
  
  H = (theta'*X')';  %the hypothesis function result
  S = sum((H - y).^2);
  avg_maker = 1/(2*m);       %(1/2m)
  
  
  J = avg_maker * S;  %cost function result
end