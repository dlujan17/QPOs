% [Xqp0,Wqp0,Zqp0] = GMOS3D_IG(Xpo,Tpo,Vpo,Epo,EOM,pars)
%
% Generate Initial Guess for the GMOS multiple-shooting Method
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
% DEPENDENCIES:
% - none
%
% AUTHOR: David Lujan
% Contact: david.lujan@colorado.edu

function [Xqp0,Wqp0,Vqp0,dU0] = GMOS3D_IG(Xpo,Tpo,Vpo,Epo,pars)
% Problem parameters
d     = pars.d;

% GMOS parameters
ds    = pars.GMOS.ds;
n     = pars.GMOS.n; % # of multiple shooting segments
N     = pars.GMOS.N;
p = prod(N);
D     = d*p;
pn = p*n;
Dn = D*n;

% Dynamics
EOM = pars.GMOS.EOM;

% Check Inputs
assert(~isreal(Epo),'Please check input eigenvalues!');

% Reintegrate Periodic Orbit
time   = linspace(0,Tpo,n+1); 
Ic     = [Xpo; reshape(eye(d),d*d,1)];
opt    = odeset('RelTol',3e-14,'AbsTol',1e-16);
[~,y] = ode113(@(t,x) EOM(t,x,pars),time,Ic,opt);

% Initial Guess
if pars.GMOS.txt
    fprintf('\nGenerating Initial Guess: ')
end
tht1        = linspace(0,2*pi,N(1)+1);
tht1(end)   = [];
tht2        = linspace(0,2*pi,N(2)+1);
tht2(end)   = [];
tht1 = reshape(repmat(tht1,N(2),1),1,p);
tht2 = repmat(tht2,1,N(1));
THT         = [tht1,tht2];
THT = reshape(repmat(THT,d,1),D,2);

% Construct Torus Skeleton
U0     = zeros(D,n);
dU0    = zeros(D,n);
TanSpace = zeros(Dn+3,2);

for ii = 1:n
    
    % Retrive PO data
    Xt = y(ii,1:d)';
    PHIt = reshape(y(ii,d+1:end),d,d);
  
    % Map eigenvector forward in time
    Vt = PHIt*Vpo;
    Vt1 = Vt(:,1)/norm(Vt(:,1));
    Vt2 = Vt(:,2)/norm(Vt(:,2));

    TanSpace(D*(ii-1)+1:D*ii,1) = repmat(real(Vt1),p,1).*cos(THT(:,1)) - repmat(imag(Vt1),p,1).*sin(THT(:,1));
    TanSpace(D*(ii-1)+1:D*ii,2) = repmat(real(Vt2),p,1).*cos(THT(:,2)) - repmat(imag(Vt2),p,1).*sin(THT(:,2));
    
    % Produce initial guess at t = t(ii)
    dU0(:,ii) = ds.*(repmat(real(Vt1),p,1).*cos(THT(:,1)) - repmat(imag(Vt1),p,1).*sin(THT(:,1)))...
        + ds/5.*(repmat(real(Vt2),p,1).*cos(THT(:,2)) - repmat(imag(Vt2),p,1).*sin(THT(:,2)));
    U0(:,ii)  = repmat(Xt,p,1) + dU0(:,ii);

end

% Stroboscopic time
T0     = Tpo;                         
w00    = 2*pi/T0;

% Rotation number
rho10     = atan2(imag(Epo(1)),real(Epo(1)));     
w10    = rho10/T0; 

rho20 = atan2(imag(Epo(2)),real(Epo(2)));
w20 = rho20/T0;

% QP Torus Initial Guess
Xqp0   = U0;
Wqp0   = [T0; rho10; rho20; w00; w10; w20];
Vqp0 = TanSpace./sqrt(sum(TanSpace(1:end-3,:).^2,1)/pn + sum(TanSpace(end-2:end,:).^2,1));
if pars.GMOS.txt
    fprintf('Done!\n')
end
end 