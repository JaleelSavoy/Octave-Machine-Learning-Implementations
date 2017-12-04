% Machine Learning - Binary Classification Problem Using Logistic Regression

%%%% Initialization, Data loading, and Data Prep
clear; close all; clc

% Load Dataset
fprintf("Loading the Dataset...\n");
data = csvread("breast-cancer-wisconsin.csv");
m = length(data);

% Load Features
X = data(:, 1:end-1); %Assign to all columns except the last column (the target variable) to the X matrix

% Load Target Variable
y = data(:, end);     %Assign the last column to the y matrix/vector 

% Preview Data (First 10 Rows)
fprintf("Data Preview, Press Enter...\n");
disp(X(1:10, :));
pause;

% Remove non-essential features
fprintf("Removing Non-Essential Features...Press Enter\n");
X(:, 1) = []; %the Sample Code Number would not contribute to solving the classification problem


#{
#  Attribute                     Domain
   -- -----------------------------------------
     REMOVED  Sample code number            id number
           2. Clump Thickness               1 - 10
           3. Uniformity of Cell Size       1 - 10
           4. Uniformity of Cell Shape      1 - 10
           5. Marginal Adhesion             1 - 10
           6. Single Epithelial Cell Size   1 - 10
           7. Bare Nuclei                   1 - 10
           8. Bland Chromatin               1 - 10
           9. Normal Nucleoli               1 - 10
          10. Mitoses                       1 - 10
          11. Class:cc                        (2 for benign, 4 for malignant)
   
   Since all features are on a scale of one to ten, there is no real reason for feature scale
    but I will anyway.
   Also, change y to have either 0 (benign) or 1 (malignant)
#}

[X, X_max, X_min] = featureScale(X);
y = binaryFeatureConvert(y, 4);

% Setup data matrix by adding a column of ones (intercept term) to the X matrix
[m_X, n_X] = size(X);
X = [ones(m,1), X];

% Split data into training and testing set
number_of_trainset = round(m*0.66);
% train sets
X_train = X(1:number_of_trainset,:);
y_train = y(1:number_of_trainset,:);
m_train = length(y_train);
% test sets
X_test  = X(number_of_trainset+1:end,:);
y_test  = y(number_of_trainset+1:end,:);
m_test = length(y_test);

%%%% Compute Cost and Gradient
%Initialize fitting parameters
init_theta = zeros(n_X +1, 1);

% Compute initial cost and gradient
[cost, grad] = costFunction(init_theta, X_train, y_train);
fprintf('At Initial Theta (zeros)...\n');
fprintf('Cost: %f\n', cost);
fprintf('Gradient: \n');
fprintf(' %f \n', grad);

%%%% Optimizing for theta using fminunc
%  Set options for fminunc
options = optimset('GradObj', 'on', 'MaxIter', 300);

%  Run fminunc to obtain the optimal theta
%  This function will return theta and the cost 
[theta, Jcost] =	fminunc(@(t)(costFunction(t, X_train, y_train)), init_theta, options);
fprintf("Finding Theta with fminunc...\n");
fprintf('Cost: %f\n', Jcost);
fprintf('theta: \n');
fprintf(' %f \n', theta);

%%%% Predict
% Evaluate Based on Train Data Fitting Performance 
predictions = predict(theta, X_train);
fprintf('Train Accuracy: %f\n', mean(double(predictions == y_train)) * 100);

% Evaluate Based on Test Data Prediction Performance 
predictions = predict(theta, X_test);
fprintf('Test Accuracy: %f\n', mean(double(predictions == y_test)) * 100);