% GMOS.m computes members of a N-D quasi-periodic invariant tori
%   family using a shooting method
%
% [Xqp,Wqp,Bqp,Zqp,Sqp,Vqp] = GMOS(Xqp0,Wqp0,Zqp0,pars)
%
% INPUT:
% VARIABLE      TYPE               DESCRIPTION
% - Xqp0        2D double array    Initial guess of QP Torus
% - Wqp0        1D array           Initial guess of torus frequencies
% - Zqp0        1D array           Initial guess of family tangent 
% - 
% - pars        struct             List of Parameters
%   .d          int                number of states
%   .GMOS
%    .ds        double             initial step-length
%    .dsMax     double             maximum step-length allowed
%    .dsMin     double             minimum step-length allowed
%    .Iter      int                No. of Newton's method interations allowed
%    .Ntrial    int                No. of attempts to compute a single torus
%    .n         int                No. of multiple-shooting segments
%    .M         int                No. of node points in each segment
%    .N         int                No. of points in stroboscopic map
%    .Nmax      int                No. of quasi-periodic tori to be computed
%    .Opt       int                optimal number of Newton's iteration
%    .Plt       bool               Flag: 1 to enable plotting functions, 0 ow.
%    .txt       bool               Flag: 1 to enable text output, 0 ow.
%    .stab      bool               Flag: 1 to compute stability matrix, 0 ow.
%    .TS        bool               Flag: 1 to enable finding tangent space, 0 ow.
%    .Tol       double             Convergence Tolerance
%    .DFT       2D double array    Discrete Fourier Transform
%    .IDFT      2D double array    Inverse Discrete Fouier Transform
%    .DT        2D double array    Derivative operator
%    .EOM       function           Equations of Motion
%    .VF        function           Vector field
%    .ParameterConstraints function dTorx1 cell array containing parametric
%                                       constraint functions
%
% OUTPUT:
% VARIABLE      TYPE               DESCRIPTION
% - Xqp         3D double array    QP tori computed with the algorithm      
% - Wqp         2D double array    Frequencies of QP tori
% - Bqp         3D double array    Floquet Matrices 
% - Zqp         2D double array    Family tangent
% - Sqp         1D double array    Step-lengths
% - Vqp         2D double array    Tangent Space
%
% DEPENDENCIES: NDRotMat.m
%
% AUTHOR: David Lujan, david.lujan@colorado.edu

function [Xqp,Wqp,Bqp,Zqp,Sqp,Vqp] = GMOS(Xqp0,Wqp0,Zqp0,pars)

% Extract Parameters
d     = pars.d;
ds    = pars.GMOS.ds;     % step-length
dsMax = pars.GMOS.dsMax;  % Max step-length
dsMin = pars.GMOS.dsMin;  % Min step-length
MaxIter = pars.GMOS.Iter;   % Max no. of iterations allowed
Ntrial = pars.GMOS.Ntrial; % Max no. of attempts to compute torus
n     = pars.GMOS.n;      % No. of Segments
M     = pars.GMOS.M;      % No. of time nodes per segment
N     = pars.GMOS.N;      % No. of GMOS Points
Nmax  = pars.GMOS.Nmax;   % No. of family members to be computed
Opt   = pars.GMOS.Opt;    % No. of Optimal iterations
Plt   = pars.GMOS.Plt;    % Plot flag
txt = pars.GMOS.txt;      % text flag
stab = pars.GMOS.stab;    % stability computation flag
TS = pars.GMOS.TanSpace; % Tangent space flag
Tol = pars.GMOS.Tol;    % Tolerance for convergence
DT = pars.GMOS.DT;        % derivative operator

% Extra Parameters
p = prod(N); % number of points in strob map
D = d*p; % dimension of each segment
pn = p*n; % total # integration points
Dn = D*n; % dimension of all segments (Dn = D for single shooting)
dTor = length(N)+1; % torus dimension

% QPO coloring vector
clr = linspace(0,1,Nmax);

% Parameter Constraint Function
S = pars.GMOS.PC;

% Integration stuff
EOM = pars.GMOS.EOM; % Equations of Motion
VF = pars.GMOS.VF; % Vector Field
opt = odeset('RelTol',3e-14,'AbsTol',1e-16);
STMIC = reshape(eye(d),d^2,1);
 
% Define Initial Guess
U0 = Xqp0;
T0 = Wqp0(1);  % Stroboscopic time
rho0 = Wqp0(2:dTor); % Rotation numbers
w0 = Wqp0(dTor+1:end); % frequencies

% Family tangent
dz0    = Zqp0;
dU0    = reshape(dz0(1:Dn),D,n);
dT0    = dz0(Dn+1);
drho0  = dz0(Dn+2:end);

if txt
    if n > 1
        fprintf('GMOS %d-D (Multiple-shooting):\n',dTor)
    else
        fprintf('GMOS %d-D (Single-shooting):\n',dTor)
    end
end

% Initialize output Matrices
Xqp    = zeros(D,M*n,Nmax);
Wqp    = zeros(2*dTor,Nmax);
if stab
    Bqp    = zeros(D,D,Nmax);
else
    Bqp = [];
end
Zqp    = zeros(Dn+dTor,Nmax);
Sqp    = zeros(1,Nmax);
if TS
    Vqp    = zeros(Dn+dTor,dTor,Nmax);
else
    Vqp = [];
end

% Initialize intermediate variables
Ut = zeros(d,pn);
tmp = zeros(d,pn,M);
PHIt = zeros(d^2,pn);
rPHI = zeros(d^2*pn,1);
cPHI = zeros(d^2*pn,1);
for k = 1:pn
    idx = d*(k-1)+1:d*k;
    rPHI(d^2*(k-1)+1:d^2*k) = repmat(idx',d,1);
    cPHI(d^2*(k-1)+1:d^2*k) = reshape(repmat(idx,d,1),d^2,1);
end
[rDF,cDF,vDF] = GMOSSparseIDs(D,n,dTor);

% Convergence Indicator
conv = 0;

% Compute up to Nmax family members
for ii = 1:Nmax
    
    % Tracking index
    if txt
        fprintf('Family member No. %d, ds = %e:\n', ii, ds)
    end
    
    % Partial Derivatives for phase constraints
    t  = linspace(0,T0,n+1);
    t0 = reshape(repmat(t(1:end-1),p,1),pn,1);

    dUT = zeros(D,n,dTor);
    dUT(:,:,1) = 1/w0(1)*reshape(VF(t0,U0(:),pars),D,n);
    for kk = 1:(dTor-1)
        dUT(:,:,kk+1) = DT(:,:,kk)*U0;
        dUT(:,:,1) = dUT(:,:,1) - w0(kk+1)/w0(1)*dUT(:,:,kk+1);
    end
    
    for trial = 1:Ntrial
        % Predictor
        U  = U0 + ds*dU0;
        T  = T0 + ds*dT0;
        rho  = rho0 + ds*drho0;
        
        for iter = 1:MaxIter
            
            % Rotation Matrix
            R = NDRotMat(d,rho,N);
            R = real(pars.GMOS.IDFT*R*pars.GMOS.DFT);
            
            % Integrate Trajectories
            % Initialization
            Ut = reshape(Ut,d,pn);
            tmp = reshape(tmp,d,pn,M);
            PHIt = reshape(PHIt,d^2,pn);

            t  = linspace(0,T,n+1);
            t0 = reshape(repmat(t(1:end-1),p,1),pn,1);
            tf = reshape(repmat(t(2:end),p,1),pn,1);
            parfor kk = 1:pn
                idx = d*(kk-1)+1:d*kk;
                time  = linspace(t0(kk),tf(kk),M+1);
                IC    = [reshape(U(idx),d,1); STMIC];
                [~,y] = ode113(@(t,x) EOM(t,x,pars),time,IC,opt);

                % Store Results
                Ut(:,kk)   = y(end,1:d)';
                if M == 1
                    tmp(:,kk,:) = y(1,1:d)';
                else
                    tmp(:,kk,:) = y(1:end-1,1:d)';
                end
                PHIt(:,kk) = reshape(y(end,d+1:end),d^2,1);
            end
            Ut = reshape(Ut,D,n);
            ft = reshape(VF(tf,Ut(:),pars),D,n);
            tmp = reshape(tmp,Dn,M);
            Xt = zeros(D,M*n);
            for kk = 1:n
                Xt(:,M*(kk-1)+1:M*kk) = tmp(D*(kk-1)+1:D*kk,:);
            end

            PHIt = PHIt(:);
            PHIn = sparse(rPHI,cPHI,PHIt);
            
            % Rotate Points
            Ur   = R*Ut;
            
            % Plot Strob Map
            if(Plt)
                figure(99)
                u0 = reshape(U(:,1),d,p);
                ut = reshape(Ut(:,end),d,p);
                ur = reshape(Ur(:,end),d,p);
                plot3(u0(1,:),u0(2,:),u0(3,:),'ob')
                hold on
                plot3(ut(1,:),ut(2,:),ut(3,:),'or')
                plot3(ur(1,:),ur(2,:),ur(3,:),'og')
                plot3(u0(1,1),u0(2,1),u0(3,1),'ob','Markerfacecolor','b')
                plot3(ut(1,1),ut(2,1),ut(3,1),'or','Markerfacecolor','r')
                plot3(ur(1,1),ur(2,1),ur(3,1),'og','Markerfacecolor','g')
                axis equal
                hold off
                drawnow
            end
            
            % Error Vector
            F = zeros(Dn+(2*dTor),1);
            
            % Quasi-periodicity
            F(1:D) = Ur(:,end) - U(:,1);

            if n > 1
                % Continuity - multiple shooting
                F(D+1:Dn) = reshape(Ut(:,1:end-1),D*(n-1),1) - reshape(U(:,2:end),D*(n-1),1);
            end

            % Jacobian - Quasi-periodicity and Continuity
            for kk = 1:n 
                % Index
                idx = D*(kk-1)+1:D*kk;
                
                if n > 1
                    % Continuity - multiple shooting
                    vDF(D^2*(kk-1)+1:D^2*kk) = reshape(-eye(D),D^2,1);
                end
                if kk == 1
                    % Quasi-periodicity
                    if n > 1
                        % mutltiple shooting
                        ad_id = (2*n-1)*D^2+(n-1)*D;
                        vDF(ad_id+(1:D^2)) = reshape(R*PHIn(((n-1)*D+1):Dn,(n-1)*D+1:Dn),D^2,1);
                        
                        for jj = 1:dTor
                            ad_id = 2*n*D^2+(n-1)*D + (jj-1)*D;
                            if jj == 1
                                vDF(ad_id+(1:D)) = R*ft(:,end)/n;
                            else
                                vDF(ad_id+(1:D)) = -DT(:,:,jj-1)*Ur(:,end);
                            end
                        end
                    else
                        % single shooting
                        vDF(1:D^2) = reshape(R*PHIn-eye(D),D^2,1);
                        
                        for jj = 1:dTor
                            ad_id = D^2 + (jj-1)*D;
                            if jj == 1
                                vDF(ad_id+(1:D)) = R*ft;
                            else
                                vDF(ad_id+(1:D)) = -DT(:,:,jj-1)*Ur;
                            end
                        end
                    end
                else
                    % Continutiy - multiple shooting
                    ad_id = n*D^2;
                   vDF(ad_id+(D^2*(kk-2)+1:D^2*(kk-1))) = reshape(PHIn(D*(kk-2)+1:D*(kk-1),D*(kk-2)+1:D*(kk-1)),D^2,1);
                    
                    ad_id = (2*n-1)*D^2;
                    vDF(ad_id+(D*(kk-2)+1:D*(kk-1))) = ft(:,kk-1)/n;
                end
            end
            
            % Phase Constraints
            if n > 1
                % multiple shooting
                ad_id = 2*n*D^2 + (n+dTor-1)*D - Dn;
            else
                % single shooting
                ad_id = D^2 + (dTor-1)*D;  
            end
            for jj = 1:dTor
                if jj == 1
                    F(Dn+jj) = dot(U(:) - U0(:),reshape(dUT(:,:,1),Dn,1))/pn;
                else
                    F(Dn+jj) = dot(U(:),reshape(dUT(:,:,jj),Dn,1))/pn;
                end
                ad_id = ad_id + Dn;
                vDF(ad_id+(1:Dn)) = reshape(dUT(:,:,jj),Dn,1)/pn;
            end
            
            % Pseudo-arclength continuation
            F(Dn+dTor+1)  = dot(U(:) - U0(:),dU0(:))/pn + (T - T0)*dT0 + (rho - rho0)'*drho0 - ds;
            if n > 1
                % multiple shooting
                ad_id = 2*n*D^2 + ((dTor+1)*n+dTor-1)*D;
                vDF(ad_id+(1:Dn)) = dU0(:)/pn;

                ad_id = 2*n*D^2 + ((dTor+2)*n+dTor-1)*D;
                vDF(ad_id+(1:dTor)) = [dT0;drho0];
            else
                % single shooting
                ad_id = D^2+2*dTor*D;
                vDF(ad_id+(1:D)) = dU0/pn;

                ad_id = D^2+(2*dTor+1)*D;
                vDF(ad_id+(1:dTor)) = [dT0;drho0];
            end

            % Parameterization Constraints
            if n > 1
                endid = 2*n*D^2+((dTor+2)*n+(dTor-1))*D+dTor;
            else
                endid = D^2+(2*dTor+1)*D+dTor;
            end
            for kk = 1:(dTor-1)
                s = S{kk};
                [F((Dn+dTor+1)+kk),df,Zidx] = s([U(:);T;rho],[U0(:);T0;rho0],dz0,pars);
                ndf = length(df);
                rDF(endid+(1:ndf)) = ((Dn+dTor+1)+kk)*ones(ndf,1);
                cDF(endid+(1:ndf)) = Zidx;
                vDF(endid+(1:ndf)) = df;
                endid = endid + ndf;
            end
            
            DF = sparse(rDF,cDF,vDF);

            % Newton's Update
            z = -DF\F;
            normF = sqrt(dot(F(1:Dn),F(1:Dn))/pn + dot(F(Dn+1:end),F(Dn+1:end)));
            
            if txt
                normz = sqrt(dot(z(1:Dn),z(1:Dn))/pn + dot(z(Dn+1:end),z(Dn+1:end)));
                fprintf('|F|* = %.4e, |z|* = %.4e, arc = %.4e\n',normF, normz, F(end-dTor+1))
            end
            
            if normF < Tol
                if sum(rho>0) < (dTor-1)
                    % rotation numbers aren't correct
                    break
                end
                if TS
                    % check if there is a tangent space
                    if ii == 1
                        TanSpace = null(full(DF(1:(Dn+dTor),:)));
                    else
                        TanSpace = orth([DF(1:(Dn+dTor),:);TanSpace']\[zeros(Dn+dTor,dTor);eye(dTor)]);
                        TanSpace = TanSpace./sqrt(sum(TanSpace(1:Dn,:).^2,1)/pn + sum(TanSpace(Dn+1:end,:).^2,1));
                    end
                    if size(TanSpace,2) ~= dTor % No :(
                        % converged but not with enough precision, update
                        % prediction and iterate again
                        U = U + reshape(z(1:end-dTor),D,n);
                        T = T + z(Dn+1);
                        rho = rho + z(Dn+2:end);
                        continue % continues iterations but does not change trials
                    end
                    % There is a full tangent space! Continue with program.
                end
                
                if txt
                    fprintf('Quasi-periodic Torus has been found!\n\n')
                end
                
                % Plot
                if(Plt)
                    figure(98)
                    if(d == 4)
                        plot(Xt(1:4:end,:),Xt(2:4:end,:),'.','Color',[0,clr(ii),1-clr(ii)])
                    else
                        plot3(Xt(1:6:end,:),Xt(2:6:end,:),Xt(3:6:end,:),'.','Color',[0,clr(ii),1-clr(ii)])
                    end
                    axis equal
                    drawnow
                end
                
                % Stability
                if stab
                    Phi = eye(D);
                    for kk = 1:n
                        Phi = PHIn(D*(kk-1)+1:D*kk,D*(kk-1)+1:D*kk)*Phi;
                    end
                    B = R*Phi;
                end
                
                % Step-size Controller
                Eps = Opt/iter;
                if(Eps > 2)
                    Eps = 2.0;
                elseif(Eps < 0.5)
                    Eps = 0.5;
                end
                ds = min(Eps*ds,dsMax);

                % Compute Family Tangent
%                 idx = [1:(Dn+dTor),(Dn+dTor+2):(Dn+2*dTor)]; % skips pseudo-arclength constraint
%                 dz = [DF(idx,:);dz0']\[zeros(Dn+2*dTor-1,1);1];
                dz = [U(:)-U0(:);T-T0;rho-rho0];
                if dot(dz,dz0) < 0
                    dz = -dz;
                end
                dz0 = dz./sqrt(dot(dz(1:Dn),dz(1:Dn))/pn + dot(dz(Dn+1:end),dz(Dn+1:end)));
                dU0 = reshape(dz0(1:end-dTor),D,n);
                dT0 = dz0(Dn+1);
                drho0 = dz0(Dn+2:end);
   
                % Store Results
                Xqp(:,:,ii) = Xt;
                Wqp(:,ii)   = [T; rho; 2*pi/T; rho/T];
                if stab
                    Bqp(:,:,ii) = B;
                end
                Zqp(:,ii)   = dz0;
                Sqp(ii)     = ds;
                if TS
                    Vqp(:,:,ii) = TanSpace;
                end
                
                % Update Previous Solution
                U0   = U;
                T0   = T;
                rho0 = rho;
                w0 = [2*pi;rho0]./T0;
                conv = 1; % convergence idicator
                break % exit the iterations loop
            else
                % Update solution
                U = U + reshape(z(1:Dn),D,n);
                T = T + z(Dn+1);
                rho = rho + z(Dn+2:end);
            end
        end % End of Newton iterations
        
        % Did we converge?
        if conv == 1 % Yes!!!
            % Is our step size too small?
            if ds <= dsMin % Yes :(
                % converged but ds is too small so exit the program
                Xqp(:,:,ii+1:end) = [];
                Wqp(:,ii+1:end)   = [];
                Bqp(:,:,ii+1:end) = [];
                Zqp(:,ii+1:end)   = [];
                Sqp(ii+1:end)     = [];
                Vqp(:,:,ii+1:end) = [];
                return
            else % Step size is adequate
                conv = 0; % reset convergence indicator
                break % converged so exit trials
            end
        else % Did not converge :(
            % Is our step size too small?
            if ds <= dsMin % Yes :(
                % no convergence and ds is already too small so exit the program
                Xqp(:,:,ii:end) = [];
                Wqp(:,ii:end)   = [];
                Bqp(:,:,ii:end) = [];
                Zqp(:,ii:end)   = [];
                Sqp(ii:end)     = [];
                Vqp(:,:,ii:end) = [];
                return
            else % Step size is adequate
                % Are we on last trial?
                if trial < Ntrial % No!!!
                    % retry with smaller step size
                    ds = ds/5;
                    if txt
                        fprintf('Retrying with ds = %e:\n\n',ds)
                    end
                else % Yes :(
                    % did not converge by last trial so exit program
                    if txt
                        fprintf('QP torus could not be found!\n\n')
                    end
                    Xqp(:,:,ii:end) = [];
                    Wqp(:,ii:end)   = [];
                    Bqp(:,:,ii:end) = [];
                    Zqp(:,ii:end)   = [];
                    Sqp(ii:end)     = [];
                    Vqp(:,:,ii:end) = [];
                    return
                end
            end
        end
    end % End of trials
end % End of continuation
end