function grad = gradient(data, labels, theta)
%computes the gradient for sigmoid function in logistic regression
% INPUTS:	data - Nxd matrix for the training data
%			labels - Nx1 matrix for labels (0,1)
%			theta - dx1 matrix for current set of weights
%
% OUTPUTS:	grad - Nx1 matrix for gradient values

grad = data'*(labels-logistic(data', theta));

end

