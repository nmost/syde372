clear all;
close all;

%1D Model Estimation
load('lab2_3.mat')
xrange = 0:0.1:500;
yrange = 0:0.1:500;

aPoints = a;
bPoints = b;
j = 1;
G = zeros(200);
naBm = zeros(200);
nbAm = zeros(200);

classDistances=[];


while (~isempty(aPoints) && ~isempty(bPoints))
rowA = randsample(1:1:length(aPoints),1);
rowB = randsample(1:1:length(bPoints),1);

disp(rowA);

za = aPoints(rowA,:);
zb = bPoints(rowB,:);


naa=0;
nbb=0;
nba=0;
nab=0;
for i = length(aPoints)
   if(MEDDecisionMetrix(aPoints(i,:),za) < (MEDDecisionMetrix(aPoints(i,:),zb)))
       naa = naa+1;
   else
       nab = nab+1;
   end
end

for i = length(bPoints)
   if (MEDDecisionMetrix(bPoints(i,:),za) < (MEDDecisionMetrix(bPoints(i,:),zb)))
       nba = nba+1;
   else
       nbb = nbb+1;
   end
end


if (nab == 0)
aPoints(rowA,:)=[];
disp('AYYY removing a');
disp(size(aPoints));
end

if (nba == 0)
bPoints(rowB,:)=[];
disp('AYYY removing b');
disp(size(bPoints));
end


end

