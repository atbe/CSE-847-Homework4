
function [weights] = logistic_train(data, labels, epsilon, maxiter)
%
% code to train a logistic regression classifier
%
% INPUTS:
% data = n * (d+1) matrix with n samples and d features, where
%       column d+1 is all ones (corresponding to the intercept term)
% labels = n * 1 vector of class labels (taking values 0 or 1)
% epsilon = optional argument specifying the convergence
%       criterion - if the change in the absolute difference in
%       predictions, from one iteration to the next, averaged across
%       input features, is less than epsilon, then halt
%       (if unspecified, use a default value of 1e-5)
% maxiter = optional argument that specifies the maximum number of
%       iterations to execute (useful when debugging in case your
%       code is not converging correctly!)
%       (if unspecified can be set to 1000)
%
% OUTPUT:
% weights = (d+1) * 1 vector of weights where the weights correspond to
%       the columns of "data"
%

if nargin == 2
    epsilon = 1e-5;
end
if nargin < 4
    maxiter = 1000;
end
weights = zeros(size(data,2),1);
end