function [ pd ] = GaussianParzen1D( trainingData, range, h)
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here
len_r = length(range);
len_t = length(trainingData);
pd = zeros(1,len_r);
for i = 1:len_r
    sum1 = 0.0;
    for j = 1:len_t
        sum1 = sum1 + (1/(sqrt(2*pi)))*exp(-0.5*((range(i)-trainingData(j))...
            /h)^2)/h;
    end
    pd(i) = sum1/len_t;
end
end

