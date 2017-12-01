function plotData(x,y)
  
  figure;     %open a new figure window
  
  data = csvread('Salary_Data.csv');  %load the data
  X = data(:, 1);       %separate the xs
  y = data(:, 2);           %and the ys
  m = length(y);        %number of training examples
  
  plot(X, y, "-" ,"MarkerSize", 30); %plot the data
  xlabel("Years Experience");   %x-axis label
  ylabel("Salary");             %y-axis label
  
end  