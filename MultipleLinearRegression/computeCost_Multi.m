function CostJ = computeCost_Multi(X,y, theta)
    m = length(y); % number of training examples

    CostJ = 0;


    CostJ = sum((X*theta - y).^2)/(2*m);




% =========================================================================

end