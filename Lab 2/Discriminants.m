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

%loop until both sets of points are empty
while (~isempty(aPoints) && ~isempty(bPoints))
    nba=-1;
    nab=-1;
    correctA = zeros(200);
    correctB = zeros(200);
    
    %loop until we find a random point that perfectly classifies either A
    %or B
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
        %loop through rows of aPoints
        for i = 1:size(aPoints, 1)
           if(MEDDecisionMetric(aPoints(i,:),za) < (MEDDecisionMetric(aPoints(i,:),zb)))
               naa = naa+1;
               %store list of indexes of correctly classified points 
               %use naa to keep track of where we are in correctA (and also
               %how many correctly classified points there are)
               correctA(naa) = i;
           else
               nab = nab+1;
               %same thing but for misclassified
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
        %correctB was initialized as zeroes, loop until we reach unset
        %(zero) values
        while(correctB(i) > 0)
            %remove every point that was correctly classified from the
            %sample set (as per instructions)
            %correctB contains indexes of correctly classified points.
            %After each removal index shifts down one, hence the -(i - 1)
            %IE. we are looping through an array and removing items from it
            %at the same time
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





