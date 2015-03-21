function [ classes ] = getclassifications( data, means, covs )
%GETCLASSIFICATIONS applies getmicdclass to an array
len = length(data);
classes = zeros(len,1);
for i=1:len
   classes(i) = getmicdclass(data(1:2,i),means,covs); 
end
end

