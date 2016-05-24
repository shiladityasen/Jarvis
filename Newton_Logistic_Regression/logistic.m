function f = logistic(x, theta)
%function to calculate sigmoid function
%	INPUTS: x - dxN matrix for training data; each column is a data point
%			theta - dx1 matrix for current set of weights
%
%	OUTPUTS:	f - Nx1 matrix containing the sigmoid values for each data point

inner = x'*theta;
f = (1.0)./(1.0+exp(-inner));

end