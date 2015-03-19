function [ class] = DiscriminantClassify(x, G)
    class = 0;
    for j=1:size(G, 1)
        if (G(j, :) == [0 0 0 0 0 0])
            break;
        end
        %if it classifies as class A and nba == 0
        if (MEDDecisionMetric(x,G(j,1:2)) < MEDDecisionMetric(x, G(j, 3:4)) && G(j,6) == 0)
            class = 1;
            break;
        %else if it classifies as class B and nab == 0
        elseif (MEDDecisionMetric(x,G(j,1:2)) >= MEDDecisionMetric(x, G(j, 3:4)) && G(j,5) == 0)
            class = 2;
            break;
        end
    end
end