function pars = InitGMOS(d)
ds       = 0; % initial step size
dsMax    = 1e-1; % max step size
dsMin    = 1e-7; % min step size
Iter     = 12; % max Newton iters
Ntrial   = 2; % # attempts to find torus
n        = 3; % # multiple shooting segments
M        = 1; % time nodes (how many strob maps to store for each segment)
N        = 41; % number of points in strob map
Nmax     = 1; % number of QPOs
Opt      = 7; % optimal no. iters
Plt      = 1; % plotting flag (0,1)
txt      = 1; % text flag (0,1)
stab     = 0; % compute stability info flag (0,1)
Tol      = 7e-11; % convergence tolerance
TanSpace = 1; % flag to compute tangent space

% Dynamics
pars.EOM = @CR3BP_EOM_PHI; % Equations of Motion
pars.VF = @CR3BP_EOM; % Vector Field

% Parametric Constraints
pars.PC = {@HoldW0};

% Parameters
pars.ds = ds;
pars.dsMax = dsMax;
pars.dsMin = dsMin;
pars.Iter = Iter;
pars.Ntrial = Ntrial;
pars.n = n;
pars.N = N;
pars.M = M;
pars.Nmax = Nmax;
pars.Opt = Opt;
pars.Plt = Plt;
pars.stab = stab;
pars.txt = txt;
pars.Tol = Tol;
pars.TanSpace = TanSpace;

% Fourier Matrices
[DFT,IDFT,DT] = NDDFT(d,N);
pars.DFT = DFT;
pars.IDFT = IDFT;
pars.DT = DT;

end