function [Xqp,Wqp,Bqp,Zqp,Sqp] = GMOS_Collocation(Xqp0,Wqp0,Zqp0,f,dfdx,J,pars)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% [Xqp,Wqp,Bqp,Zqp,Sqp] = GMOS_Collocation(Xpo,Tpo,EOM,f,dfdx,J,pars)
%
% Compute several members of a 2D quasi-periodic invariant tori family
% using the collocation version of the GMOS algorithm
%
% INPUT:
% VARIABLE      TYPE               DESCRIPTION
% - Xqp0        2D double array    Initial guess of QP Torus
% - Wqp0        1D array           Initial guess of torus frequencies
% - Zqp0        1D array           Initial guess of family tangent 
% - @f          function           Vector field
% - @dfdx       function           Partial derivatives of Vector field
% - J           2D double array    Canonical Transformation [1]
% - pars        struct             List of Parameters
%   .d          int                number of states
%   .GMOS
%    .Collocation
%     .n        int                No. of Collocation segments
%     .m        int                degree of Legendre Polynomials
%    .ds        double             initial step-length
%    .dsMax     double             maximum step-length allowed
%    .Iter      int                No. of Newton's method interations allowed
%    .N         int                No. of GMOS solution points
%    .Nmax      int                No. of quasi-periodic tori to be computed
%    .Opt       int                optimal number of Newton's iteration
%    .Plt       int                Flag: 1 to enable plotting functions, 0 ow.
%    .Tol       double             Convergence Tolerance
%
% OUTPUT:
% VARIABLE      TYPE               DESCRIPTION
% - Xqp         3D double array    QP tori computed with the algorithm      
% - Wqp         2D double array    Frequencies of QP tori
% - Bqp         3D double array    Floquet Matrices 
% - Zqp         2D double array    Family tangent
% - Sqp         1D double array    Step-lengths
%
% DEPENDENCIES:
% - LagrangeMatrix.m
% - LagrangePolynomials.m
% - FourierMatrix.m
% - RotationMatrix.m
%
% AUTHOR: N. Baresi
%
% REFERENCES:
% [1] Olikara, Z. P., "Computation of Quasi-periodic Tori and Heteroclinic
% Connections in Astrodynamics using Collocation Techniques", PhD Thesis,
% University of Colorado Boulder, 2016, Chapter 2, pp. 24-31 
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Problem parameters
d     = pars.d;

% Collocation parameters
n     = pars.GMOS.Collocation.n;
m     = pars.GMOS.Collocation.m;

Ns    = (n*(m+1)+1);

% GMOS parameters
ds    = pars.GMOS.ds;
dsMax = pars.GMOS.dsMax;
dsMin = pars.GMOS.dsMin;  % Min step-length
MaxIter  = pars.GMOS.Iter;
Ntrial = pars.GMOS.Ntrial; % Max no. of attempts to compute torus
N     = pars.GMOS.N;
Nmax  = pars.GMOS.Nmax;
Opt   = pars.GMOS.Opt;
Plt   = pars.GMOS.Plt;
Tol   = pars.GMOS.Tol;
param_const = pars.GMOS.param_const;
txt = pars.GMOS.txt;      % text flag
stab = pars.GMOS.stab;    % stability computation flag
DT = pars.GMOS.DT;        % derivative operator

D     = d*N;
Npts  = Ns*N;
clr  = linspace(0,1,Nmax);

%% Initialization %%%%
if txt
    fprintf('Initialization:\n')
end

% Create Time Points
t     = linspace(0,1,n+1); t(end) = [];

% Legendre Polynomial Roots
syms tau;
tm    = vpasolve(legendreP(m,tau) == 0);
tm    = double(tm);
tm    = (tm + 1)/(2*n);
tm    = [0; tm]; 

% Calculate Lagrange Polynomials
[L,~] = LagrangeMatrix(t(2),tm);
DL    = zeros(m,m+1);
for jj = 2:m+1
    [~,dL] = LagrangeMatrix(tm(jj),tm);
    DL(jj-1,:) = dL;
end
L1_c  = repmat(L,D,1); L1_c = L1_c(:)';

L2    = sparse(m*D,(m+1)*D);
for jj = 1:m
    idx = D*(jj-1)+1:D*jj;
    L2(idx,:) = sparse(repmat(1:D,1,m+1),1:(m+1)*D,repmat(DL(jj,:),D,1));
end


% % Quadrature Coefficients - Currently not using these, but may be useful
% % evaluation of integral constraints
% fprintf('Compute Quadrature Coefficients...');
% w  = zeros(m+1,1);
% for jj = 1:m+1
%     w(jj) = integral(@(z) LagrangePolynomial(z,tm,jj),t(1),t(2),'AbsTol',1e-16,'RelTol',3e-14);
% end
% fprintf('Done!\n');


% Create Time Vector
t    = repmat(t,m+1,1) + repmat(tm,1,n);
t    = [t(:); 1];

% Canonical Transformation
Jd  = sparse(zeros(d*N));
for ii = 1:N
    idx = d*(ii-1)+1:d*ii;
    Jd(idx,idx) = J;
end

% Fourier Coefficients 
%[DFT,IDFT,DT] = FourierMatrix(d,N);
% [DFT,IDFT,DT,~,~] = DFTv3(d,N);
DFT  = sparse(pars.GMOS.DFT);
IDFT = sparse(pars.GMOS.IDFT);
DT   = sparse(DT);

%% Initial Guess
if txt
    fprintf('Load Initial Guess:\n')
end

% Define Initial Guess
U0     = Xqp0;
P0     = Wqp0(1);                        % Stroboscopic time 
p0     = Wqp0(2);                        % Rotation number
w00    = Wqp0(3);
w10    = Wqp0(4);

% Approximate family tangent
dz0    = Zqp0;
dU0    = reshape(dz0(1:end-6),D,Ns);
dP0    = dz0(end-5);
dp0    = dz0(end-4);
dw00   = dz0(end-3);
dw10   = dz0(end-2);
dl00   = dz0(end-1);
dl10   = dz0(end);

%% GMOS Algorithm
if txt
    fprintf('GMOS Algorithm:\n')
end

% Initialize Matrices
Xqp    = zeros(D,Ns,Nmax);
Wqp    = zeros(4,Nmax);
Bqp    = zeros(D,D,Nmax);
Zqp    = zeros(D*Ns+6,Nmax);
Sqp    = zeros(1,Nmax);

conv = 0;
% Compute up to Nmax family members
for ii = 1:Nmax
    
    % Tracking feedback
    if txt
        fprintf('Family member No. %d, ds = %e:\n', ii, ds)
    end
    
    % Partial Derivatives for Phase Constraints
    dUT0 = zeros(D,n*(m+1)+1);
    dUT1 = DT*U0;
    parfor kk = 1:Ns
        dUT0(:,kk) = 1/(2*pi) * (P0*Ff(P0*t(kk),U0(:,kk),f,pars) - p0*dUT1(:,kk));
    end
     
    for trial = 1:Ntrial
        % Predictor
        U  = U0 + ds*dU0;
        P  = P0 + ds*dP0;
        p  = p0 + ds*dp0;
        w0 = w00 + ds*dw00;
        w1 = w10 + ds*dw10;
        l0 = 0;
        l1 = 0;

        % Corrector
        for jj = 1:MaxIter
            
            % Rotation Matrix
            %R  = RotationMatrix(p,d,N);
            R = RotMatv2(p,d,N);
            R = sparse(real(IDFT*R*DFT));
            
            % Partial Derivatives for unfolding the parameters
            dUt0 = zeros(D,Ns);
            dUt1 = DT*U;
            parfor kk = 1:Ns
                dUt0(:,kk) = 1/(2*pi) * (P*Ff(P*t(kk),U(:,kk),f,pars) - p*dUt1(:,kk));
            end
            
            % Plot
            if(Plt)
                figure(99)
                u0 = reshape(U(:,1),d,N);
                ut = reshape(U(:,end),d,N);
                ur = reshape(R*U(:,end),d,N);
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
            
            % Initialization
            F_c  = cell(1,Ns+6);
            DF_c = cell(1,Ns+6);
            
            %%% Error Vector & Error Jacobian %%%
            % Quasi-periodicity
            END                      = D*Ns+6;
            F_c{1}                   = [1:D; ones(1,D); (R*U(:,end) - U(:,1))'];
            
            [i,j,r]                  = find(R*eye(D));
            DF_c{1}                  = [       1:D,                i',                1:D; ...
                1:D,  j' + END - D - 6,  (END-4)*ones(1,D); ...
                -ones(1,D),                r',   (-DT*R*U(:,end))'];
                        
            parfor kk = 2:Ns  
                % Indices
                idx = D*(kk-1)+1:D*kk;
                Q   = mod(kk-1,m+1);
                
                if(Q == 0)
                    %% Continuity Conditions
                    % Error vector
                    F_c{kk}       = [idx; ones(1,D); (U(:,kk) - sum(repmat(L,D,1).*U(:,kk-m-1:kk-1),2))'];
                    
                    % Error jacobian
                    idxL          = D*(kk-length(L)-1)+1:D*(kk-1);
                    DF_c{kk}      = [repmat(idx,1,m+2); idxL, idx; -L1_c, ones(1,D)];
                    
                else
                    %% Collocation Conditions
                    % Error vector
                    fF            = Ff(P*t(kk),U(:,kk),f,pars);
                    F_c{kk}       = [idx; ones(1,D); (P*fF - sum(repmat(DL(Q,:),D,1).*U(:,kk-Q:kk-Q+m),2))'];
                    
                    % Temporary matrix used to pass values to cell array
                    idxA          = D*Q+1:D*(Q+1);
                    tmp           = zeros(D,D*(m+1));
                    tmp(:,idxA)   = P*DFDXf(P*t(kk),U(:,kk),dfdx,pars);
                    tmp           = tmp - L2(D*(Q-1)+1:D*Q,:);
                    [i,j,s]       = find(tmp);
                    
                    DF_c{kk}      = [  (D*(kk-1) + i'),               idx,                idx,              idx; ...
                        (D*(kk-Q-1) + j'), (END-5)*ones(1,D),  (END-1)*ones(1,D),  (END)*ones(1,D); ...
                        s',               fF',   (Jd*dUt0(:,kk))', (Jd*dUt1(:,kk))'];
                end
            end
                       
            %% Phase Conditions
            % dU/dtht0
            F_c{Ns+1}         = [END-5; 1; dot(U(:) - U0(:), dUT0(:))/Npts];
            DF_c{Ns+1}        = [(END-5)*ones(1,D*Ns); ...
                1:D*Ns; ...
                dUT0(:)'/Npts];
            
            % dU/dtht1
            F_c{Ns+2}         = [END-4; 1; dot(U(:), dUT1(:))/Npts];
            DF_c{Ns+2}        = [(END-4)*ones(1,D*Ns); ...
                1:D*Ns; ...
                dUT1(:)'/Npts];
                       
            %% Parametrization Equations
            if param_const == 1
                %%% Fixing longitudinal frequency %%%
                F_c{Ns+3}       = [END-3; 1; P - P0]; % hold w0 constant
                DF_c{Ns+3}      = [END-3; END-5; 1];
            elseif param_const == 2
                %%% Fixing latitudinal frequency %%%
                F_c{Ns+3} = [END-3; 1; (p*w0/(2*pi))-w10];
                DF_c{Ns+3} = [END-3;END-4; w0/(2*pi)];
            elseif param_const == 3
                F_c{Ns+3} = [END-3; 1; (p-P*w10)/(2*pi-P*w00) + (dp0-w10*dP0)/(w00*dP0)]; % hold slope of movement direction constant
                DF{Ns+3} = [END-3, END-3;
                            END-5, END-4;
                    w00*(p-w10*P)/(2*pi-w00*P)^2 - w10/(2*pi-w00*P), 1/(2*pi-w00*P)];
            end
            
            % Pseudo-arclength continuation...
            psdarc          = dot(U(:) - U0(:), dU0(:))/Npts + (P - P0)*dP0 + (p - p0)*dp0 - ds;
            F_c{Ns+4}       = [END-2; 1; psdarc];
            DF_c{Ns+4}      = [(END-2)*ones(1,D*Ns), END-2, END-2;
                1:D*Ns, END-5, END-4;
                dU0(:)'/Npts,   dP0,   dp0];
                       
            %% Frequencies
            % w1
            F_c{Ns+5}       = [END-1; 1; P*w0 - 2 * pi];
            DF_c{Ns+5}      = [END-1, END-1;
                END-5, END-3;
                w0,     P];
            
            % w2
            F_c{Ns+6}       = [END; 1; P*w1 - p];
            DF_c{Ns+6}      = [  END,   END,   END; ...
                END-5, END-4, END-2; ...
                w1,    -1,     P];
            
            %% Create sparse error matrix
            IJV             = cell2mat(F_c);
            F               = full(sparse(IJV(1,:),IJV(2,:),IJV(3,:)));
                        
            %% Create sparse jacobian matrix
            IJV             = cell2mat(DF_c);
            DF              = sparse(IJV(1,:),IJV(2,:),IJV(3,:));
                        
            %% Newton Update
            dz              = -DF\F;
            test1           = sqrt(F(1:end-6)'*F(1:end-6)/Npts + F(end-5:end)'*F(end-5:end));
            test2           = sqrt(dz(1:end-6)'*dz(1:end-6)/Npts + dz(end-5:end-2)'*dz(end-5:end-2));
            if txt
                fprintf('|F|* = %.6e, |dz|* = %.6e\n',test1,test2)
            end    
            
            %% Check for convergence
            if test1 < Tol                
                % Success!
                if txt
                    fprintf('QP torus has been found!\n\n')
                end
                
                % Plot
                if(Plt)
                    figure(97)
                    for kk = 1:n*(m+1)
                        ut = reshape(U(:,kk),d,N);
                        if(d == 4)
                            plot(ut(1,:),ut(2,:),'.','Color',[0,clr(ii),1-clr(ii)])
                            hold on
                        else
                            plot3(ut(1,:),ut(2,:),ut(3,:),'.','Color',[0,clr(ii),1-clr(ii)])
                            hold on
                        end
                    end
                    axis equal
                    hold off
                    drawnow
                end
                               
                %% Stability
                if stab
                    A    = DF(D+1:end-6,1:D);
                    B    = DF(D+1:end-6,D+1:end-6);
                    PHI  = sparse([eye(D); -B\A]);
                    
                    % Floquet Matrix
                    G    = full(R*PHI(end-(D-1):end,1:D));
                else
                    G = 0;
                end
                               
                %% Compute Family Tangent
                dz0  = [U(:) - U0(:); P - P0; p - p0; w0 - w00; w1 - w10; 0; 0];
                % DG   = [DF(1:end-3,:); DF(end-1:end,:)]; [~,~,dz] = svd(full(DG));
                % dz0  = dz(:,end);
                % if(dot([U(:) - U0(:); P - P0; p - p0; w1 - w10; w2 - w20; 0; 0], dz0) < 0)
                % dz0 = - dz0;
                % end
                dz0  = dz0/sqrt(dz0(1:end-6)'*dz0(1:end-6)/Npts + dz0(end-5:end-4)'*dz0(end-5:end-4));
                dU0  = reshape(dz0(1:end-6),D,Ns);
                dP0  = dz0(end-5);
                dp0  = dz0(end-4);
                dw00 = dz0(end-3);
                dw10 = dz0(end-2);
                dl00 = 0;
                dl10 = 0;
                               
                %% Step-length Controller
                Eps  = Opt/jj;
                if(Eps > 2)
                    Eps = 2;
                elseif(Eps < 0.5)
                    Eps = 0.5;
                end
                ds   = min(dsMax, Eps*ds);
                               
                %% Store Results
                Xqp(:,:,ii) = U;
                Wqp(:,ii)   = [P; p; w0; w1];
                Bqp(:,:,ii) = G;
                Zqp(:,ii)   = dz0;
                Sqp(ii)     = ds;
                               
                %% Update Old Solution
                U0          = U;
                P0          = P;
                p0          = p;
                w00         = w0;
                w10         = w1;
                l10         = 0;
                l20         = 0;
                conv = 1;
                break               
            else
                % Newton's Update
                U  = reshape(U(:) + dz(1:end-6),D,n*(m+1)+1);
                P  = P + dz(end-5);
                p  = p + dz(end-4);
                w0 = w0 + dz(end-3);
                w1 = w1 + dz(end-2);
                l0 = l0 + dz(end-1);
                l1 = l1 + dz(end);
            end
        end
        if conv == 1
            if ds <= dsMin
                Xqp(:,:,ii:end) = [];
                Wqp(:,ii:end)   = [];
                Bqp(:,:,ii:end) = [];
                Zqp(:,ii:end)   = [];
                Sqp(ii:end)     = [];
                return
            else
                conv = 0;
                break
            end
        else
            if ds <= dsMin
                Xqp(:,:,ii:end) = [];
                Wqp(:,ii:end)   = [];
                Bqp(:,:,ii:end) = [];
                Zqp(:,ii:end)   = [];
                Sqp(ii:end)     = [];
                return
            else
                if trial < Ntrial
                    ds = ds/10;
                    if txt
                        fprintf('Retrying with ds = %e:\n\n',ds)
                    end
                else
                    if txt
                        fprintf('QP torus could not be found!\n\n')
                    end
                    Xqp(:,:,ii:end) = [];
                    Wqp(:,ii:end)   = [];
                    Bqp(:,:,ii:end) = [];
                    Zqp(:,ii:end)   = [];
                    Sqp(ii:end)     = [];
                    return
                end
            end
        end
    end % End of trials
end % End of continuation
end % End of GMOS



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function dX = Ff(t,X,f,pars)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% dX = Ff(t,X,f,pars)
% 
% Create Augmented Vector field
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % Dimensions
    d  = pars.d;
    N  = pars.GMOS.N;
    
    D  = d*N;

    
    % Initialization
    dX = zeros(D,1);

    
    % Create Augmented Vector field
    for ii = 1:N
        idx      = d*(ii-1)+1:d*ii;
        dX(idx)  = f(X(idx),pars);
    end
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function A = DFDXf(t,X,dfdx,pars)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% A = DFDXf(t,X,dfdx,pars)
% 
% Create Augmented Vector gradient
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Dimensions
    d  = pars.d;
    N  = pars.GMOS.N;
    
    D  = d*N;

    % Initialization
    A  = sparse(D,D);

    % Create Augmented Jacobiam matrix
    for ii = 1:N
        idx        = d*(ii-1)+1:d*ii;
        A(idx,idx) = dfdx(X(idx),pars);
    end
end











