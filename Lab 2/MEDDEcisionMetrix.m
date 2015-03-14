function [ val ] = MEDDecisionMetrix( x,m )
w=-(m);
wo = (m*m');
val = (w*x')+wo;
end
