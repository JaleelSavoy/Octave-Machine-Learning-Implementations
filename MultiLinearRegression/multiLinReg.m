%%Machine Learning - Multiple Linear Regression with Sample Salary Data (small dataset)

%%%%Initialize
clear; close all; clc

%%%%Load and Plot the dataset
fprintf("Loading the data...\n");
fprintf("Plotting the data...\n");
data = csvread('softdrink.csv');
X = data(:,1:2);   %features (time and cases)
[X X_max X_min] = featureScale(X);
y = data(:, end);   %target variable (distance)
m = length(y);    %number of training examples

%%%%Print first 5 data points
fprintf("First 5 rows of the data...\n");
display(data(1:5,:));

%Paused the program
fprintf("Program is paused. Press Enter to continue...");
pause;

%%%%Add intercept term to X by adding a column to the X matrix_type
X = [ones(m, 1) X];

%%%%Normal Equation
fprintf("Solving with normal equation...\n");

%Calculate for the parameters from the normal equation
theta = normEqtn(X,y);

%Display the results
fprintf('Theta computed: \n');
fprintf(' %f \n', theta);
fprintf('\n');

%Predict from time: 22 and distance: 10
time_prediction = (12 - X_min(:,1))/(X_max(:,1) - X_min(:,1));
case_prediction = (3 - X_min(:,2))/(X_max(:,2) - X_min(:,2));

prediction = [1,time_prediction,case_prediction]*theta;
fprintf("Prediction for a 12 minutes and 3 cases: %f\n", prediction);