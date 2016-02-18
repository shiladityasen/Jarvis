function l = loss(data, theta, labels)
%	function to compute the total log-loss value for the dataset with current set of weights
%
%	INPUTS:	data - Nxd training data
%			theta - dx1 set of current weights
%			labels - Nx1 truth values
%
%	OUTPUTS:	l - total log-loss over dataset

f = logistic(data', theta);
l = sum(labels.*log(f) + (1-labels).*log(1-f));

end

