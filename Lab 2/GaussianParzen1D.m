function [ pd ] = GaussianParzen1D( data, h, sigma, range )
%Gets a 1D Parzen Window estimation using a Gaussian Distribution
pd = zeros(length(range));
n = length(data);
gf = makedist('Normal', 'mu', 0, 'sigma', sigma);

for j=1:length(range)
    for i=1:n
        xi = data(i);
        pt = (range(j) - xi)/h;
        pdfval = pdf(gf, pt)/h; %this zero for some reason
        pd(j) = pd(j) + pdfval;
    end
    pd(j) = pd(j)/n;
end
end

