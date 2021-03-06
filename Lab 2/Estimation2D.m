clear all;
close all;

%1D Model Estimation
load('lab2_2.mat')

xrange = 0:0.1:500;
yrange = 0:0.1:500;

ml_mu_al = sum(al)/length(al);
ml_sigma_al = 0.0;
for i = 1:length(al)
    ml_sigma_al = ml_sigma_al + squareddistance2d(al(i,:), ml_mu_al);
end
ml_sigma_al = ml_sigma_al / length(al);

ml_mu_bl = sum(bl)/length(bl);
ml_sigma_bl = 0.0;
for i = 1:length(bl)
    ml_sigma_bl = ml_sigma_bl + squareddistance2d(bl(i,:), ml_mu_bl);
end
ml_sigma_bl = ml_sigma_bl / length(bl);

ml_mu_cl = sum(cl)/length(cl);
ml_sigma_cl = 0.0;
for i = 1:length(cl)
    ml_sigma_cl = ml_sigma_cl + squareddistance2d(cl(i,:), ml_mu_cl);
end
ml_sigma_cl = ml_sigma_cl / length(cl);

gclassify = @(x) max([GaussianPDF2D(x, ml_mu_al, ml_sigma_al);...
                        GaussianPDF2D(x, ml_mu_bl, ml_sigma_bl);...
                        GaussianPDF2D(x, ml_mu_cl, ml_sigma_cl)]);
                    
[plotX, plotY] = meshgrid(0:1:500);

xy = [plotX(:) plotY(:)];

map = zeros(length(xy), 1);
for i = 1:length(xy)
    [M, I] = gclassify(xy(i, :));
    map(i) = I;
end

decisionmap = reshape(map, size(plotX));
figure(1)
contour(decisionmap);
hold on;
plot(al(:,1),al(:,2),'r.');
hold on;
plot(bl(:,1),bl(:,2),'g.');
hold on;
plot(cl(:,1),cl(:,2),'b.');

cov = [400, 0; 0, 400]; %sigma
[range_x, range_y] = meshgrid(0:1:500);
coord_vec = [range_x(:) range_y(:)];

parzenA = GaussianParzen2D(al, coord_vec, cov);
parzenB = GaussianParzen2D(bl, coord_vec, cov);
parzenC = GaussianParzen2D(cl, coord_vec, cov);


parzmap = zeros(length(coord_vec), 1);
for i = 1:length(coord_vec)
    [M, I] = max([parzenA(i); parzenB(i); parzenC(i)]);
    parzmap(i) = I;
end

parzdecisionmap = reshape(parzmap, size(range_x));
figure(2)
contour(parzdecisionmap)
hold on;
plot(al(:,1),al(:,2),'r.')
hold on;
plot(bl(:,1),bl(:,2),'g.')
hold on;
plot(cl(:,1),cl(:,2),'b.')
