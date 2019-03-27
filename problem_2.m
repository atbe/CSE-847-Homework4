addpath(fullfile('SLEP-master', 'SLEP','functions','L1','L1R'));
addpath(fullfile('SLEP-master', 'SLEP','functions'));
addpath(fullfile('SLEP-master', 'SLEP','opts'));

alz_data = load('./ad_data.mat');

X_train = alz_data.X_train;
y_train = alz_data.y_train;
X_test = alz_data.X_test;
y_test = alz_data.y_test;
parameters = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1];
weights = zeros(numel(parameters),size(X_train,2));
predictions = zeros(numel(parameters), size(X_test,1));

bias_values = zeros(size(parameters));
number_of_features = zeros(size(parameters));
number_of_features_with_bias = zeros(size(parameters));
test_accuracy = zeros(size(parameters));
all_aucs = zeros(size(parameters));
for i = 1:numel(parameters)
    parameter = parameters(i);
    [w, bias] = logistic_l1_train(X_train, y_train, parameter);
    bias_values(i) = bias;
    weights(i,:) = w;
    number_of_features(i) = sum(w ~= 0);
    y_pred = (X_test * w) + bias;
    performance = classperf(y_test >= 0,y_pred >=0);
    test_accuracy(i) = performance.CorrectRate;
    [far,gar,thres,auc] = perfcurve(y_test, y_pred, 1);
    all_aucs(i) = auc;
end

figure;
plot(parameters, test_accuracy, 'o-');
title('Sparse Logistic Regression Experiment');
xlabel('l1 Parameter');
ylabel('Testing Accuracy');

