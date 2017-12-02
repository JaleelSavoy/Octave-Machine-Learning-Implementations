function [r_squared] = rSquared(X, y, theta)
  prediction = X*theta;
  y_mean = mean(y);
  
  SStot = sum((y - y_mean).^2);
  SSres = sum((y - prediction).^2);
  
  r_squared = (SSres/SStot);

end  