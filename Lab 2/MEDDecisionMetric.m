function [ val ] = MEDDecisionMetric( x,m )
    val = ((x-m)*(x-m)')^(1/2);
end
%w=-(m);
%wo = (m*m');
%val = (w*x')+wo;