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
    nba=-10;
    nab=-10;
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
    
    G(j, :) = [za zb nab nba];
    if (nab == 0)
        i = 1;
        while(correctB(i) > 0)
            bPoints(correctB(i) - (i - 1),:)=[];
            i = i + 1;
        end
        %aPoints(rowA,:)=[];
        %disp('AYYY removing a');
        %disp(size(aPoints));
    end

    if (nba == 0)
        i = 1;
        while(correctA(i) > 0)
            aPoints(correctA(i) - (i - 1),:)=[];
            i = i + 1;
        end
        %bPoints(rowB,:)=[];
        %disp('AYYY removing b');
        %disp(size(bPoints));
    end

    j = j + 1;
    disp(length(aPoints))
    disp(length(bPoints))
    disp('___')

end



