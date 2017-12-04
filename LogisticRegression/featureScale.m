function [X_scale, X_max, X_min] = featureScale(X);
  X_scale = X;
  X_max = zeros(1, size(X,2));
  X_min = zeros(1, size(X,2));

  %Rescaling
  X_max = max(X);
  X_min = min(X);
  indices = 1:size(X,2);
  
  for i=1:indices,
    X_scale(:,i) = (X(:,i) - X_min(i))/(X_max(i) - X_min(i));
  end
end