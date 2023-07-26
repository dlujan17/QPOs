function [val,dval] = LagrangePolynomial(t,tm,jj)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% [val,dval] = LagrangePolynomial(t,tm,jj)
% 
% Calculate Lagrange Basis Polynomials
% 
% DEPENDENCIES: 
% - none
% 
% AUTHOR: N. Baresi
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

M = length(tm);

assert(jj <= M,'jj index exceeds length of time mesh points!');

% Initialization
den  = 1;
val  = 1;
dval = 0;


% Compute Polynomial
for mm = 1:M
    if(mm ~= jj)
        den = den.*(tm(jj)-tm(mm));
        val = val.*(t - tm(mm));
    end
end

% Compute Polynomial Derivative
for mm = 1:M
    tmp = 1;
    if(mm ~= jj)
        for nn = 1:M
            if(nn ~= mm && nn ~= jj)
                tmp = tmp.*(t - tm(nn));
            end 
        end
        dval = dval + tmp;
    end
end

% Output val and dval
val = val./den;
dval = dval./den;
end