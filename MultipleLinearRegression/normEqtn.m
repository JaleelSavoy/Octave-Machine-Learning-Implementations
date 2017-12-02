function [theta] = normEqtn(X,y)
  theta = (pinv(X'*X)*(X'*y));
  
end