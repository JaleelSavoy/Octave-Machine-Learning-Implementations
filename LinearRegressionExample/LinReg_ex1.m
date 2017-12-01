%%Machine Learning - Linear Regression with Sample Salary Data (small dataset)

%%%%Initialize%%%%
clear; close all; clc

%%%%Plot the dataset%%%%
fprintf("Plotting the data...\n");
data = csvread('Salary_Data.csv');
X = data(:, 1);   %features
#X = featureScale(X);
y = data(:, 2);   %target variables
m = length(y);    %number of training examples

%%%%Plot%%%%
plotData(X, y);
fprintf("Program has been paused. Press Enter to continue...\n");
pause;

%%%%Cost Function and Gradient Descent%%%%
X = [ones(m, 1), X];
theta = zeros(2, 1); %initialize parameters matrix

iterations = 15000;  %number of gradient descent iterations
alpha = 0.0001;       %learning rate

  %Cost Function
fprintf("Computing Cost Function...\n");
%compute cost function, and then display
J = computeCost(X, y, theta);
fprintf('With theta = [0 ; 0]\nCost computed = %f\n', J);

J = computeCost(X, y, [-1 ; 2]);
fprintf('\nWith theta = [-1 ; 2]\nCost computed = %f\n', J);
fprintf("Program has been paused. Press Enter to continue...\n");
pause;

  %Gradient Descent
fprintf("Running Gradient Descent...\n");
theta = gradientDescent(X,y, theta, alpha, iterations);
%print theta to screen
fprintf("Theta found:\n");
fprintf("%f\n", theta);

%plot the linear fit
hold on; %previous plot stays visible
plot(X(:,2), X*theta, '*');
legend("Training Data", "Linear Regression");
hold off  %do not overlay any more plots on this figure

%predict values for salary for 8 and 9 years
format long;
predict1 = [1, 8] * theta;
fprintf("For Years Experience = 8, we predict a Salary of $%f\n", predict1);
predict2 = [1, 9] * theta;
fprintf("For Years Experience = 9, we predict a Salary of $%f\n", predict2);

fprintf("Program has finished. Press Enter to continue...\n");
pause;