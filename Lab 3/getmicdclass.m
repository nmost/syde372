function [ class ] = getmicdclass( x, means, covs )
%GETMICDCLASS Outputs and integer (from 1 to 10) equal to the class
%   x: the data to be classified
%   means: the means for all classes
%   covs: the covariances for all classes
class = 0;
currmin = -1;
for i=1:10
    covmat = reshape(covs(i, :), 2, 2);
    diff = x - means(i, :)';
    val = (diff)'*inv(covmat)*(diff);
    if (currmin == -1 || val < currmin)
        class = i;
        currmin = val;
    end
end
end

