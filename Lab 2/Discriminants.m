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
        %Remove all the zero rows and columns
        correctB( ~any(correctB,2), : ) = [];  %rows
        correctB( :, ~any(correctB,1) ) = [];  %columns
        for i = 1:size(correctB)
            %remove every point that was correctly classified from the
            %sample set (as per instructions)
            %correctB contains indexes of correctly classified points.
            %REPLACE the row with a zero row INSTEAD of deleting, otherwise
            %the indexes we saved will be wrong!
            bPoints(correctB(i),:)=[0,0];
        end
            %NOW we can delete all those zero rows.
            bPoints( ~any(bPoints,2), : ) = []; 

    end

    if (nba == 0)
             i = 1;
        %Remove all the zero rows and columns
        correctA( ~any(correctA,2), : ) = [];  %rows
        correctA( :, ~any(correctA,1) ) = [];  %columns
        for i = 1:size(correctA)
            aPoints(correctA(i),:)=[0,0];
        end
            %delete all those zero rows.
            aPoints( ~any(aPoints,2), : ) = [];
    end

    j = j + 1;

end

%Delete our zero rows on G
G( ~any(G,2), : ) = [];  %rows


[plotX, plotY] = meshgrid(0:1:500);

xy = [plotX(:) plotY(:)];

map = zeros(length(xy), 1);

%Iterate through every point.
for i = 1:length(xy)
    %Iterate through every G (if necessary)
     for x = 1:j;
         %{
          %Classify this point based off of the first descriminant.
           if(MEDDecisionMetric(xy(i,:),G(x,1:2))) < (MEDDecisionMetric(xy(i,:),G(x,3:4)))
               %We have classified as A.
               %Make sure NBa is zero, otherwise move to the next discriminant 
                if(G(x,6) == 0)
                    %This point is correctly classified as A
                    map(i) = 1;
                end
           else
               %We have classified as B using this discriminant.
               %Make sure NaB is zero, otherwise move to the next discriminant 
              if(G(x,5) == 0)
                    %This point is correctly classified as B
                    map(i) = 2;
              end
           end
          %If we set a value for this point, we can move to the next one.
          if (map(i) ~= 0)
              break
          end
           
          %}
         map(i) = DiscriminantClassify(xy(i, :), G);
     end
end


decisionmap = reshape(map, size(plotX));
contour(decisionmap);
hold on;
plot(a(:,1),a(:,2),'r.');
hold on;
plot(b(:,1),b(:,2),'b.');



