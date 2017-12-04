%Machine Learning - Multiple Feature Linear Regression using data on concrete samples
  %Using Normal Equation and then Gradient Descent and comparing

%%%%Initialize
clear; close all; clc

%Load Data
data = csvread("Concrete_Data.csv");

%Define x and y
y = data(:,end); % 28-day Compressive Strength (Mpa) 
X = data(:,1:7); % see commment block below for colnames
m = length(y);

%Add intercept term to the X matrix, by adding a column of ones
X = [ones(m,1),X];

%Print first five data points in the data set
fprintf("First 5 rows of the dataset...\n");
display(data(1:5,:));
fprintf("Press Enter to Continue...\n");
pause; %Pause the program so user can view the data

#{

Input variables (7)(component kg in one M^3 concrete): 
  Cement	
  Slag	
  Fly ash	
  Water	
  SP	
  Coarse Aggr.	
  Fine Aggr.	

Output Variable
  SLUMP (cm)	NOT USED
  FLOW (cm)	  NOT USED
  28-day Compressive Strength (Mpa) 
  
#}

%Split data into training and testing set
number_of_trainset = round(698*0.75);
%train sets
X_train = X(1:number_of_trainset,:);
y_train = y(1:number_of_trainset,:);
m_train = length(y_train);
%test sets
X_test  = X(number_of_trainset+1:end,:);
y_test  = y(number_of_trainset+1:end,:);
m_test = length(y_test);



%%%%Method 1: Normal Equation%%%%
fprintf("Solving this linear regression problem using the Normal Equation...\n");

%calculate for the parameters from the normal equation
theta = normEqtn(X_train,y_train);

%Display the calculated theta vectorize
fprintf("Theta computed: \n");
fprintf("%f\n", theta);
predictions = X*theta;


%evaluate our model
[r_squared] = rSquared(X, y, theta);
fprintf("R-Squared: %0.4f\n", r_squared); 



%%%%Method 2: Gradient Descent%%%%
%Re-Load Data
data = csvread("Concrete_Data.csv");

%Define x and y
y = data(:,end); % 28-day Compressive Strength (Mpa) 
X = data(:,1:7); % see commment block below for colnames
[X X_max X_min] = featureScale(X);
m = length(y);

%Add intercept term to the X matrix, by adding a column of ones
X = [ones(m,1),X];

%Split data into training and testing set
number_of_trainset = round(698*0.75);
%train sets
X_train = X(1:number_of_trainset,:);
y_train = y(1:number_of_trainset,:);
m_train = length(y_train);
%test sets
X_test  = X(number_of_trainset+1:end,:);
y_test  = y(number_of_trainset+1:end,:);
m_test = length(y_test);

fprintf("Solving this linear regression problem using Gradient Descent...\n");

%Choose alpha
alpha = 0.01;

%Choose number of iterations
iterations = 555;

%Initialize theta
theta = zeros((size(X,2)),1);

%Run Gradient Descent
[theta, J_Results] = gradientDescent_Multi(X,y, theta, alpha, iterations);
xlabel("Number of iterations");
ylabel("Cost of J");

%Plot convergence
plot(1:numel(J_Results), J_Results, '-b', 'LineWidth', 2); 

%Display the calculated theta vectorize
fprintf("Theta computed: \n");
fprintf("%f\n", theta);
predictions = X*theta;


%evaluate our model
[r_squared] = rSquared(X, y, theta);
fprintf("R-Squared: %0.4f\n", r_squared); 


