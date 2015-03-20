%% SYDE 372 Lab #3: nmost, VicV, ajwootto

load feat.mat
%1: since our feature is based on the grayscale level of each pixel
%compared to it's orthogonal neighbors, the "noisiest" images are likely to
%be the most confused; cork, cloth, stone, and pigskin

%2: the images with the most regular shapes are easier to separate from
%eachother; the face, paper, and raiffa should be easily distinguishable.

[f2means, f2covs] = getmeansandcovs(f2);
f2covs(1, :, :)