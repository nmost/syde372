function [ c ] = k_means( data, k )
%Initialize the cluster to random positions within the max values of the
%features
len = length(data)
Z = rand(k, 2)*max(reshape(data(1:2,:),[len*2, 1]))
c = zeros(len, 1);
for i=1:len
   mindist = -1;
   for cind=1:k
       dist = %compute the distance here boys
       if (mindist == -1 || dist < mindist)
          mindist = dist;
          c(i) = cind;
       end
   end
end
end

