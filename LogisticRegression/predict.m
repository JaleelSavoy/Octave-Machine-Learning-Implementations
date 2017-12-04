function predictions = predict(theta, X)
  m = size(X,1);
  predictions = zeros(m,1);
  predictions = round(sigmoid(X*theta));
end  