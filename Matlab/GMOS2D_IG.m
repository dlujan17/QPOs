% GMOS2D_IG.m generates initial guess for a GMOS shooting method
%
% [Xqp0,Wqp0,Zqp0] = GMOS2D_IG(Xpo,Tpo,Vpo,Epo,EOM,pars)
% 
% INPUTs:
% VARIABLE     TYPE         DESCRIPTION
% - Xpo        double       Periodic Orbit Initial Conditions
% - Tpo        double       Periodic Orbit Period
% - Vpo        double       Center subspace Eigenvector
% - Epo        double       Center subspace Eigenvalue
% - @EOM       function     Equations of Motion
% - pars       struct       List of Parameters
%   .d         int          No. of states
%   .GMOS
%    .ds       double       Initial displacement along center subspace
%    .n        int          No. of Multiple-shooting segments
%    .N        int          No. of GMOS solution points
%
% OUTPUTS:
% VARIABLE     TYPE                 DESCRIPTION
% Xqp0         2D double array      QP Torus initial guess
% Wqp0         1D double array      QP Torus frequencies initial guess
% Zqp0         1D double array      Approximated Family Tangent
%
% DEPENDENCIES: none
%
% AUTHOR: N. Baresi, edited by David Lujan, david.lujan@colorado.edu

function [Xqp0,Wqp0,Zqp0] = GMOS2D_IG(Xpo,Tpo,Vpo,Epo,pars)

% Problem parameters
d     = pars.d;

% GMOS parameters
ds    = pars.GMOS.ds;
n     = pars.GMOS.n;
N     = pars.GMOS.N;
D     = d*N;

% Check Inputs
assert(~isreal(Epo),'Please check input eigenvalues!')

% Reintegrate Periodic Orbit
EOM = pars.GMOS.EOM;
time   = linspace(0,Tpo,n+1); 
Ic     = [Xpo; reshape(eye(d),d*d,1)];
opt    = odeset('RelTol',3e-14,'AbsTol',1e-16);
[~,X0] = ode113(@(t,x) EOM(t,x,pars),time,Ic,opt);

% Construct Torus Skeleton
tht    = linspace(0,2*pi,N+1); tht(end) = [];
THT    = repmat(tht,d,1); 
THT    = THT(:);
U0     = zeros(D,n);
dU0    = zeros(D,n);
for ii = 1:n
    % Retrive PO data
    Xt = X0(ii,1:d)';
    Mt = reshape(X0(ii,d+1:end),d,d);
  
    % Map eigenvector forward in time
    Vt = Mt*Vpo;
    Vt = Vt/norm(Vt);
    
    % Produce initial guess at t = t(ii)
    dU0(:,ii) = (repmat(real(Vt),N,1).*cos(THT) - repmat(imag(Vt),N,1).*sin(THT));
    U0(:,ii)  = repmat(Xt,N,1) + ds*dU0(:,ii);
end

% Stroboscopic time
T0     = Tpo;                         
dT0    = 0;
w00    = 2*pi/T0;

% Rotation number
rho0     = atan2(imag(Epo),real(Epo));     
drho0    = 0;
w10    = rho0/T0;    

% Output
% QP Torus Initial Guess
Xqp0   = U0;
Wqp0   = [T0; rho0; w00; w10];

% Approximate family tangent
Zqp0   = [dU0(:); dT0; drho0]/sqrt(dU0(:)'*dU0(:)/(n*N) + dT0^2 + drho0^2); 
end 