function [ pd ] = GaussianParzen2D( trainingdata, range, E )
    len_r = length(range);
    len_t = length(trainingdata);
    pd = ones(len_r);
    denom = sqrt(2*pi)*sqrt(det(E));

    for i = 1:len_r
        sum1 = 0;
        for j = 1:len_t
            diff = range(i, :) - trainingdata(j, :);
            expon = diff / E * diff' / -2;
            numer = exp(expon);
            sum1 = numer/denom;
        end
        pd(i) = sum1/len_t; %ALWAYS GIVES 1 WTF
    end
end

