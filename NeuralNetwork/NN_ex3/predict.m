function p = predict(Theta1, Theta2, X)
  %PREDICT: Predict the label of an input for a neural network (trained)
  %   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
  %   trained weights of a neural network (Theta1, Theta2)

  m = size(X,1);
  number_of_labels = size(Theta2, 1);

  p = zeros(m, 1);

  % Input Layer
  xWithBias = [ones(m,1),X];
  a1 = xWithBias;

  % Hidden Layer
  z2 = a1 * Theta1';
  a2 = [ones(size(z2,1),1), sigmoid(z2)];

  % Output Layer
  z3 = a2 * Theta2';
  a3 = sigmoid(z3);

  %Output
  [maxP, p] = max(a3, [], 2);

end