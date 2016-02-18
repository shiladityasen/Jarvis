function labels = predict(data, theta)
% function to predict labels for data points, given set of weights
%
%	INPUTS: data - Nxd data
%			theta - dx1 weights
%
%	OUTPUTS:	lables - Nx1 predicted labels with a threshold of 0.5

labels = logistic(data', theta)>0.5;

end

