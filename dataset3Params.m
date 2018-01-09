function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel(Gaussian Kernel)
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. 
%   You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
values_C = [0.01 0.02 0.03 0.1 0.2 0.3 1 2 3 10 20 30];
values_sigma = [0.01 0.02 0.03 0.1 0.2 0.3 1 2 3 10 20 30];

min_error = 1000;

for i=1:length(values_C),
    local_C = values_C(i);
    for j=1:length(values_sigma),
        local_sigma = values_sigma(j);
        model = svmTrain(X, y, local_C, @(x1, x2) gaussianKernel(x1, x2, local_sigma));
        predictions = svmPredict(model, Xval);
        pred_error = mean(double(predictions ~= yval));
        if pred_error < min_error,
            %fprintf('\nPrediction error is %f with C %f and sigma %f', pred_error, local_C, local_sigma);
            min_error = pred_error;
            C = local_C;
            sigma = local_sigma;
        end
    end
    
end

fprintf('\nPrediction error is %f with C %f and sigma %f', min_error, C, sigma);

% =========================================================================

end
