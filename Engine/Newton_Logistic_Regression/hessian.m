function H = hessian(data, theta)
%function for computing the Hessian for log-loss in logistic regression
%	INPUTS:	data - Nxd matrix for training data
%			theta - Nx1 matrix for current weights
%
%	OUTPUTS:	H - dxd Hessian matrix

 [N d] = size(data);
 H = zeros(d);
 
 for i=1:N
     x = data(i,:)';
     f = logistic(x,theta);
     
     h = f*(1-f) * (x*x');
     H = H-h;
 end

end

