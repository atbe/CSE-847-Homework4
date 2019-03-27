X = load("./data.txt");
X_test = X(2001:4601,:);
y = load("./labels.txt");
y_test = y(2001:4601);

test_accuracies = [0, 0, 0, 0, 0, 0]';
train_sizes = [200, 500, 800, 1000, 1500, 2000]';
w = zeros(6, size(X,2));

for i = 1:6
    n = train_sizes(i);
    X_train = X(1:n, :);
    y_train = y(1:n);
    log_reg = logistic_train(X_train, y_train, 1e-5, 1000);
    
    y_pred = round(sigmf(X_test * log_reg,[1 0]));
    test_accuracies(i) = sum(y_test== y_pred) / numel(y_test);
end

figure;
plot(train_sizes, test_accuracies, 'o-');
title('Problem 1: Logistic Regression Experiment');
xlabel('Number of elements (n)');
ylabel('Testing Accuracy');
saveas(gcf, 'problem_1_accuracy.png');
