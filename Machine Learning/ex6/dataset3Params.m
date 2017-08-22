function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
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
temp = [0.01, 0.02, 0.1, 0.3, 1, 3, 10, 30, 100];
i = 0;
err = zeros(9, 9);
for c = [0.01, 0.02, 0.1, 0.3, 1, 3, 10, 30, 100],
	i = i + 1; j = 0;
	for sig = [0.01, 0.02, 0.1, 0.3, 1, 3, 10, 30, 100],
		j = j + 1;
		model= svmTrain(X, y, c, @(x1, x2) gaussianKernel(x1, x2, sig)); 
		predictions = svmPredict(model, Xval);
		err(i, j) = mean(double(predictions ~= yval));
	end
end
[M,I] = min(err(:));
[I_row, I_col] = ind2sub(size(err),I);

C = temp(I_row);
sigma = temp(I_col);

% =========================================================================

end
