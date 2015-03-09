clear all;
close all;

%1D Model Estimation
load('lab2_1.mat')
xrange = [0:.1:10];

true_a = makedist('Normal', 'mu', 5, 'sigma', 1);
true_b = makedist('Exponential', 'mu', 1); %mean of exponential is 1/lambda, lambda is 1

ml_mu_a = sum(a)/length(a);
ml_sigma_a = sum(arrayfun(@squareddistance, a, ones(size(a))*ml_mu_a))... 
    /length(a);
ml_mu_b = (sum(b)/length(b));
ml_sigma_b = sum(arrayfun(@squareddistance, b, ones(size(b))*ml_mu_b))...
    /length(b);


gaussian_dist_a = makedist('Normal', 'mu', ml_mu_a, 'sigma', ml_sigma_a);
gaussian_dist_b = makedist('Normal', 'mu', ml_mu_b, 'sigma', ml_sigma_b);
exponential_dist_a = makedist('Exponential', 'mu', ml_mu_a);
exponential_dist_b = makedist('Exponential', 'mu', ml_mu_b);

figure(1)
plot(xrange, pdf(true_a, xrange), xrange, pdf(true_b, xrange),...
    xrange,pdf(gaussian_dist_a,xrange), xrange ,pdf(gaussian_dist_b, xrange));
title('Parametric Estimate: Gaussian Distribution')
legend('True a', 'True b', 'Gaussian Estimate of a', 'Gaussian Estimate of b')

figure(2)
plot(xrange, pdf(true_a, xrange), xrange, pdf(true_b, xrange),...
    xrange,pdf(exponential_dist_a,xrange), xrange ,pdf(exponential_dist_b, xrange));
title('Parametric Estimate: Exponential Distribution')
legend('True a', 'True b', 'Gaussian Estimate of a', 'Gaussian Estimate of b')