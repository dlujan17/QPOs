% function [W,InvW,DT] = NDDFT(d,N)
%
% NDDFT.m computes an n = length(N) dimensional discrete Fourier transform and
%   its inverse in matrix form. To properly apply the matrices of the ouput
%   you MUST have the elements of your function f arranged such that the
%   first variable changes in order while the others are held constant.
%
%   example: [W,InvW,dW] = NDDFT(2,[2,3])
%           |f1(0,0)|
%           |f2(0,0)|
%           |f1(0,1)|
%           |f2(0,1)|
%           |f1(0,2)|
%       f = |f2(0,2)| ,  F = W*f,   f = InvW*F
%           |f1(1,0)|
%           |f2(1,0)|
%           |f1(1,1)|
%           |f2(1,1)|
%           |f1(1,2)|
%           |f2(1,2)|
%
% Inputs:
%   d: scalar of the dimension of your function f. d is independent of the
%       dimension of the DFT. If d = 1 then f is scalar valued and if d > 1
%       then f is vector valued.
%   N: 1xn array containing the number of points to discretize each dimension.
%       Must be an odd number of points.
%
% Outputs:
%   W: pxp matrix to convert from time domain to frequency domain
%   InvW: pxp matrix to convret from frequency domain to time domain
%   DT: pxpxn containing derivative matrices to be applied to constraints
%       for QP tori in GMOS.
%
%       p = prod(N)*d
%       n = length(N)
%
% Dependencies: none
%
% Author: David Lujan, david.lujan@colorado.edu

function [W,InvW,DT] = NDDFT(d,N)
ndim = length(N);

% Error check. Code needs odd number of points.
for i = 1:ndim
    if mod(N(i),2) == 0
        disp('Must use odd number of points')
        W = nan;
        InvW = nan;
        DT = nan;
        return
    end
end

if ndim ~= 1
    InvWr = cell(1,ndim);
    DInvWr = cell(1,ndim);
end

for dim = 1:ndim
    % Create 1-d DFT and IDFT matrices
    % complex DFT
    J = 0:N(dim)-1;
    K = [0:(N(dim)-1)/2,-(N(dim)-1)/2:-1];
    WN = exp(-1j*(J'*K)*2*pi/N(dim));
    
    % real DFT, inverse real DFT, and derivative of inverse real DFT - used
    % for DT in GMOS
    WNr = zeros(N(dim));
    WNr(1,:) = 1;
    InvWNr = zeros(N(dim));
    InvWNr(:,1) = 1/2;
    DInvWNr = zeros(N(dim));
%     DInvWrN(:,1) = 1/2;
    for k = 1:(N(dim)-1)/2 
        WNr(2*k,:) = cos(J*2*pi*k/N(dim));
        WNr(2*k+1,:) = -sin(J*2*pi*k/N(dim));
        InvWNr(:,2*k) = cos(J*2*pi*k/N(dim));
        InvWNr(:,2*k+1) = -sin(J*2*pi*k/N(dim));
        DInvWNr(:,2*k) = k.*sin(J*2*pi*k/N(dim));
        DInvWNr(:,2*k+1) = k.*cos(J*2*pi*k/N(dim));
    end
    InvWNr = 2/N(dim)*InvWNr;
    DInvWNr = -2/N(dim)*DInvWNr;
    
    % Do Kronecker product to recursively create entire DFT and real DFT
    % matrices
    if dim == 1
        W = WN;
        Wr = WNr;
    else
        W = kron(W,WN);
        Wr = kron(Wr,WNr);
    end
    
    if ndim == 1
        InvWr = InvWNr;
        DInvWr = DInvWNr;
    else
        InvWr{dim} = InvWNr;
        DInvWr{dim} = DInvWNr;
    end
        
end

InvW = 1/prod(N)*W';

W = kron(W,eye(d));
InvW = kron(InvW,eye(d));

% Compute all derivative matrices.
if ndim == 1
    DT = kron(DInvWr*Wr,eye(d));
else
    DT = zeros(prod(N)*d,prod(N)*d,ndim);
    for dim = 1:ndim % derivative number to compute
        for noper = 1:ndim % operation number to execute
            if noper == 1 % initialize B
                if dim == 1
                    B = DInvWr{1};
                else
                    B = InvWr{1};
                end
            else
                if noper == dim
                    B = kron(B,DInvWr{noper});
                else
                    B = kron(B,InvWr{noper});
                end
            end
        end
        DT(:,:,dim) = kron(B*Wr,eye(d));
    end
end

end