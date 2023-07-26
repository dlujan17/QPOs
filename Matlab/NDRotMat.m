% function Q = NDRotMat(rho,d,N)
%
% NDRotMat.m creates a matrix that rotates the coefficients from a discrete
%   Fourier transform (DFT) by the angles specified in rho.
%
% Inputs:
%   d: scalar of the number of states for the system.
%   rho: [1xn] array containing the angles in radians to rotate each
%       dimension.
%   N: [1xn] array containing the number of discretized points in each
%       dimension. Must be odd number of points in each dimension.
%
% Outputs:
%   Q: [pxp] matrix containing the rotation matrix to be applied to the DFT
%       coefficients.
%
%   p = prod(N)
%   n = length(N)
%
% Dependencies: none
%
% Author: David Lujan, david.lujan@colorado.edu

function Q = NDRotMat(d,rho,N)

ndim = length(N);
for dim = 1:ndim
%     K = 0:N(dim)-1;
    K = [0:(N(dim)-1)/2,-(N(dim)-1)/2:0];
    K(end) = [];
    if dim == 1
        Q = diag(exp(-1j*K*rho(dim)));
    else
        Q = kron(Q,diag(exp(-1j*K*rho(dim))));
    end
end

Q = kron(Q,eye(d));
end