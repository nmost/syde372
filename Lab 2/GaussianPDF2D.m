function [ p ] = GaussianPDF2D( x, u, E )
    p = 1/((2*pi)^(1/2)*det(E)^(1/2))*exp(-1/2*(x-u)/E*(x-u)');
end