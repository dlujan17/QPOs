function [Xqp0,Wqp0,Zqp0] = GMOS_Collocation_InitialGuess(Xpo,Tpo,Vpo,Epo,EOM,f,dfdx,pars)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% [Xqp0,Wqp0,Zqp0] = GMOS_Collocation_InitialGuess(Xpo,Tpo,Vpo,Epo,f,dfdx,pars)
%
% Generate Initial Guess for the GMOS Collocation Method
% 
% INPUTs:
% VARIABLE     TYPE         DESCRIPTION
% - Xpo        double       Periodic Orbit Initial Conditions
% - Tpo        double       Periodic Orbit Period
% - Vpo        double       Center subspace Eigenvector
% - Epo        double       Center subspace Eigenvalue
% - @EOM       function     Equations of Motion
% - @f         function     Vector field
% - @dfdx      function     Vector field 
% - pars       struct       List of Parameters
%   .d         int          No. of states
%   .GMOS
%    .ds       double       Initial displacement along center subspace
%    .N        int          No. of GMOS solution points
%    .Collocation
%     .n       int          No. of collocation segments
%     .m       int          Degree of Lagrange polynomials

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
% AUTHOR: N. Baresi
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Problem parameters
d     = pars.d;

% GMOS Collocation parameters
n     = pars.GMOS.Collocation.n;
m     = pars.GMOS.Collocation.m;
Ns    = (n*(m+1)+1);

% GMOS parameters
ds    = pars.GMOS.ds;
N     = pars.GMOS.N;
Npts  = Ns*N;


% Check Inputs
assert(isreal(Epo) == 0,'Please check input eigenvalues!');



%% Initialization %%%%
%fprintf('Initialization:\n');

% Create Time Points
t     = linspace(0,1,n+1); t(end) = [];


% Legendre Polynomial Roots
%fprintf('Compute Legendre Polynomial Roots...');
syms tau;
tm    = vpasolve(legendreP(m,tau) == 0);
tm    = double(tm);
tm    = (tm + 1)/(2*n);
tm    = [0; tm]; 
%fprintf('Done!\n');


% Create Time Vector
%fprintf('Create time vector...');
t    = repmat(t,m+1,1) + repmat(tm,1,n);
t    = [t(:); 1];
%fprintf('Done!\n');



%% Initial Guess 
%fprintf('\nGenerating Initial Guess:\n');

% Integrate Initial Guess
time   = Tpo*t;
Ic     = [Xpo; reshape(eye(d),d*d,1)];
opt    = odeset('Reltol',3e-14,'Abstol',1e-16);
[~,X0] = ode113(EOM,time,Ic,opt,f,dfdx,pars);


% Construct Torus skeleton
%fprintf('Construct Torus skeleton...');
tht    = linspace(0,2*pi,N+1); tht(end) = [];
THT    = repmat(tht,d,1); 
THT    = THT(:);
U0     = zeros(d*N,Ns);
dU0    = zeros(d*N,Ns);
for ii = 1:Ns
    Xt        = X0(ii,1:d)';
    Mt        = reshape(X0(ii,d+1:end), d, d);
    Vt        = Mt*Vpo/norm(Mt*Vpo);
    
    dU0(:,ii) = (repmat(real(Vt),N,1).*cos(THT) - repmat(imag(Vt),N,1).*sin(THT));
    U0(:,ii)  = repmat(Xt,N,1) + ds*dU0(:,ii);
end
%fprintf('Done!\n');
% U0     = repmat(X0(:,1:d)',N,1);   % for cases where monodromy matrix is not available...

P0     = Tpo;                        % Stroboscopic time 
dP0    = 0;
w10    = 2*pi/Tpo;
dw10   = 0;

p0     = atan2(imag(Epo),real(Epo));     % Rotation number
if p0 < 0
    p0 = 2*pi + p0;
end
dp0    = 0;
w20    = p0/P0;              
dw20   = 0;

dl10   = 0;
dl20   = 0;


% QP Torus Initial Guess
Xqp0   = U0;
Wqp0   = [P0; p0; w10; w20];

% Approximate family tangent
Zqp0   = [dU0(:); dP0; dp0; dw10; dw20; dl10; dl20]/sqrt(dU0(:)'*dU0(:)/Npts + dP0^2 + dp0^2); 
end % Continue below for EOM.m


