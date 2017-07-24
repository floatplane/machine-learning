function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%

# (y == 0) is an array where (y == 0)(n) == 1 if y(n) == 0
# (y == 1) is an array where (y == 1)(n) == 1 if y(n) == 1. In other words, (y == 1) is just the same as y.

pos_indices = find(y);
neg_indices = find(y == 0);

plot(X(pos_indices, 1), X(pos_indices, 2), 'k+', 'LineWidth', 2, 'MarkerSize', 7);
plot(X(neg_indices, 1), X(neg_indices, 2), 'ko', 'LineWidth', 2, 'MarkerSize', 7, 'MarkerFaceColor', 'y');







% =========================================================================



hold off;

end
