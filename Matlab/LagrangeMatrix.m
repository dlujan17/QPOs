function [L,DL] = LagrangeMatrix(T,t)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% [L,DL] = LagrangeMatrix(T,t)
% 
% Calculate Lagrange Matrices
% 
% DEPENDENCIES:
% - LagrangePolyhomial.m (MATLAB function)
%
% AUTHOR: N. Baresi
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Initialization
    M  = length(t);
    L  = zeros(1,M);
    DL = zeros(1,M);

    for mm = 1:M
        [L(mm),DL(mm)] = LagrangePolynomial(T,t,mm);
    end
end