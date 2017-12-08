% Machine Learning Neural Network - Handwritten Letter Recognition

% Initialize
clear; close all; clc

% Setup Parameters
input_layerSize  = 400;   %20 by 20 Input Images of Handwritten Digits
hidden_layerSize =  25;   %25 hidden units 
number_of_layers =  10;   %10 labels (1 to 10 for digits 0 to 9, with "0" mapped to 10)

% Load & Display Data
fprintf("Loading and Displaying Data...\n");
load('ex3data1.mat');
m = size(X,1);

% Select 100 Data Points (Randomly)
randSelect = randperm(size(X,1));
randSelect = (1:100);
displayData(X(randSelect, :));    %Code Reused for Coursera ML Coursera

fprintf("Press Enter to Continue...\n");

% Load Parameters
load("ex3weights.mat");

% Predict
predictions = predict(Theta1, Theta2, X);
fprintf("\n Training Set Accuracy: %f\n", mean(double(predictions ==y))*100); 
fprintf("Press Enter to Continue...\n");

% Randomly permute examples
random_permutes = randperm(m);
for i = 1:m
    fprintf('\nDisplaying Example Image\n');
    displayData(X(random_permutes(i), :));

    pred = predict(Theta1, Theta2, X(random_permutes(i),:));
    fprintf('\nNeural Network Prediction: %d (digit %d)\n', pred, mod(pred, 10));
    
    % Pause with quit option
    s = input('Enter to continue, 0 to exit:','s');
    if s == '0'
      break
    end
end
