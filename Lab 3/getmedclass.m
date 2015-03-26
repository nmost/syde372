function [class] = getmedclass(x, z, k)
    min = -1;
    class = 0;
    for i=1:k
        if (min == -1 || sqrt((x - z(:,i))'*(x-z(:,i))) < min)
            min = sqrt((x - z(:,i))'*(x-z(:,i)));
            class = i;
        end
    end
end