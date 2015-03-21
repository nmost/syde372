function [ means,covs ] = getmeansandcovs( feats )
means = zeros(10, 2);
covs = zeros(10, 2, 2);
%MASSIVE HACK: assumes that the features are sorted by image (which they
%are)
for i=0:9
    sind = i*16 + 1;
    eind = sind + 15;
    means(i+1, 1) = mean(feats(1, sind:eind));
    means(i+1, 2) = mean(feats(2, sind:eind));
    c = cov(feats(1:2, sind:eind)');
    covs(i+1, 1, 1) = c(1,1);
    covs(i+1, 1, 2) = c(1,2);
    covs(i+1, 2, 1) = c(2,1);
    covs(i+1, 2, 2) = c(2,2);
end
end