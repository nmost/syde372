clear all;
close all;

%1D Model Estimation
load('lab2_1.mat')
xrange = [0:.1:10];
true_a = makedist('Normal', 'mu', 5, 'sigma', 1);
true_b = makedist('Exponential', 'mu', 1); %mean of exponential is 1/lambda, lambda is 1

figure(1)
plot(xrange, pdf(true_a, xrange))
title('Parametric Estimate: Gaussian Distribution')
legend('True Distribution')