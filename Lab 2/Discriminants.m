clear all;
close all;

%1D Model Estimation
load('lab2_3.mat')
xrange = 0:0.1:500;
yrange = 0:0.1:500;

aPoints = a;
bPoints = b;
j = 1;
G = zeros(200, 6);
naBm = zeros(200);
nbAm = zeros(200);

classDistances=[];


while (~isempty(aPoints) && ~isempty(bPoints))
    nba=-1;
    nab=-1;
    correctA = zeros(200);
    correctB = zeros(200);
    
    while (nba ~= 0 && nab ~= 0)
        rowA = randsample(1:1:size(aPoints, 1),1);
        rowB = randsample(1:1:size(bPoints, 1),1);

        %disp(rowA);

        za = aPoints(rowA,:);
        zb = bPoints(rowB,:);


        naa=0;
        nbb=0;
        nba=0;
        nab=0;
        for i = 1:size(aPoints, 1)
           if(MEDDecisionMetric(aPoints(i,:),za) < (MEDDecisionMetric(aPoints(i,:),zb)))
               naa = naa+1;
               correctA(naa) = i;
           else
               nab = nab+1;
               naBm(nab) = i;
           end
        end

        for i = 1:size(bPoints, 1)
           if (MEDDecisionMetric(bPoints(i,:),za) < (MEDDecisionMetric(bPoints(i,:),zb)))
               nba = nba+1;
               nbAm(nba) = i;
           else
               nbb = nbb+1;
               correctB(nbb) = i;
           end
        end
    end
    %stores each 'good' discriminant in a row of G, like this:
    % [Ax Ay Bx By nab nba]
    G(j, :) = [za zb nab nba];
    if (nab == 0)
        i = 1;
        while(correctB(i) > 0)
            %remove every point that was correctly classified from the
            %sample set (as per instructions)
            %correctB contains indexes of correctly classified points.
            %After each removal index shifts down one, hence the -(i - 1)
            bPoints(correctB(i) - (i - 1),:)=[];
            i = i + 1;
        end
    end

    if (nba == 0)
        i = 1;
        while(correctA(i) > 0)
            aPoints(correctA(i) - (i - 1),:)=[];
            i = i + 1;
        end
    end

    j = j + 1;
    disp(length(aPoints))
    disp(length(bPoints))
    disp('___')

end





