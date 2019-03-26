
function [weights] = logistic_train(data, labels, epsilon, maxiter)
%
% code to train a logistic regression classifier
%
% INPUTS:
% data = n * (d+1) matrix withn samples and d features, where
% column d+1 is all ones (corresponding to the intercept term)
% labels = n * 1 vector of class labels (taking values 0 or 1)
% epsilon = optional argument specifying the convergence
% criterion - if the change in the absolute difference in
% predictions, from one iteration to the next, averaged across
% input features, is less than epsilon, then halt
% (if unspecified, use a default value of 1e-5)
% maxiter = optional argument that specifies the maximum number of
% iterations to execute (useful when debugging in case your
% code is not converging correctly!)
% (if unspecified can be set to 1000)
%
% OUTPUT:
% weights = (d+1) * 1 vector of weights where the weights correspond to
% the columns of "data"
%

if nargin == 2
    epsilon = 1e-5;
end
if nargin < 4
    maxiter = 1000;
end
weights = zeros(size(data,2),1);

iteration = 0;

while (iteration <= maxiter)
    y = sigmf(data * weights, [1 0]);
    R = diag (y .* (1 - y) );
    
    % Adding scalar to avoid matrix being close to signular
    noise = 0.01;
    R = R + noise * eye(length(R));
    z = (data * weights) - (R^(-1) * (y - labels));
    weights = (data' * R * data)^(-1) * data' * R * z;    
    y_new = sigmf(data * weights,[1 0]);
    predicition_magnitude_difference = mean(abs(y_new - y));   
    
    if (predicition_magnitude_difference < epsilon)
        break;
    end
    iteration = iteration + 1;
end

end