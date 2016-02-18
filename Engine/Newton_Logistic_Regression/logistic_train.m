function weights = logistic_train(data,labels,epsilon,maxiterations)
%  [weights] = logistic_train(data,labels,epsilon,maxiterations)
%
%  code to train a logistic regression classifier
%
% INPUTS:
%	data = n x (d+1) matrix with n samples and d features, where
%			column d+1 is all ones (corresponding to the intercept term)
%	labels = n x 1 vector of class labels (taking values 0 or 1)
%	epsilon = optional argument specifying the convergence
%               criterion. Let pi be the probability predicted by the model
%				for data point i and deltai be the absolute change in this
%				probability since the last iteration. Let delta be the average
%				of the deltai?s, over all the training data points. When delta
%				becomes lower than epsilon, then halt.
%	         	(if unspecified, use a default value of 10?-5)
%	maxiterations = optional argument that specifies the
%					maximum number of iterations to execute (useful when
%					debugging in case your code is not converging correctly!)
%					(if unspecified can be set to 1000)
%
% OUTPUT:
%    weights = (d+1) x 1 vector of weights where the weights
%               correspond to the columns of "data"

if ~exist('epsilon', 'var')
    epsilon = 10e-5;
end

if ~exist('maxiterations', 'var')
    maxiterations = 1000;
end

[N d] = size(data);
prev_theta = zeros(1,d)';

x = {};
acc = {};
l = {};

for i=1:maxiterations
    new_theta = prev_theta - inv(10e-6*eye(d)+hessian(data, prev_theta))*gradient(data, labels, prev_theta);
    
    x{i} = i;
    l{i} = loss(data, new_theta, labels);
    acc{i} = N-sum(abs(predict(data,new_theta)-labels));
    
    if mean(abs(new_theta-prev_theta)) < epsilon
        break;
    end
    
    prev_theta = new_theta;
    %disp(i);
end

weights = new_theta;

% figure('name', 'loss');
% plot(cell2mat(x), cell2mat(l), 'r', 'LineWidth', 2);
% 
% figure('name', 'accuracy');
% plot(cell2mat(x), cell2mat(acc), 'r', 'LineWidth', 2);
% 
% disp(acc{i}/N);

end

