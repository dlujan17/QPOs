module GMOSPkg
using LinearAlgebra, Printf, Distributed, Interpolations, Statistics, Combinatorics

function InitGMOS(d::Int)
    pars = Dict{String,Any}()

    ds = 1e-4::Float64 # Starting step size
    dsMax = 1e-1::Float64 # Maximum step size
    dsMin = 1e-6::Float64 # Minimum step size
    MaxIter = 12::Int64 # Maximum number of iterations for Newton's method
    Opt = 5::Int64 # Optimal number of iterations for Newton's method
    Tol = 1e-11::Float64 # Tolerance level for Newton's method
    Ntrial = 2::Int64
    n = 3::Int64 # number of multiple-shooting segments
    M = 1::Int64 # number of time nodes for each multiple-shooting segment
    N = [41]::Array{Int64,1} # Starting number of Fourier modes
    Pmax = [81]::Array{Int64,1} # Maximum number of Fourier modes
    Pmin = [19]::Array{Int64,1} # Minimum number of Fourier modes
    Nmax = 20::Int64 # Maximum number of solutions
    txt = false # Text: can be {true,false,"verbose"}
    Plt = false # Plotting
    stab = false # Stability matrix
    TS = true # Tangent Space
    AFM = false # Adaptive Fourier Modes
    RAv = false # Resonance Avoidance
    
    # Parameters
    pars["ds"] = ds
    pars["dsMax"] = dsMax
    pars["dsMin"] = dsMin
    pars["Iter"] = MaxIter
    pars["Opt"] = Opt
    pars["Tol"] = Tol
    pars["Ntrial"] = Ntrial
    pars["n"] = n
    pars["M"] = M
    pars["N"] = N
    pars["Pmax"] = Pmax
    pars["Pmin"] = Pmin
    pars["Nmax"] = Nmax
    pars["txt"] = txt
    pars["Plt"] = Plt
    pars["stab"] = stab
    pars["TS"] = TS
    pars["AFM"] = AFM
    pars["RAv"] = RAv

    # Fourier Matrices
    (D, ID, DT) = ndDFT(d,N)
    pars["DFT"] = D
    pars["IDFT"] = ID
    pars["DT"] = DT

    return pars
end

function ndDFT(d::Int64,N)
    ndim = length(N)

    for k = 1:ndim
        if mod(N[k],2) == 0
            println("Must use odd number of points!")
            W = NaN
            InvW = NaN
            dW = NaN
            return W, InvW, dW
        end
    end

    W = Float64[]
    Wr = Float64[]
    InvWr = zeros(Float64,sum(N),sum(N))
    DInvWr = zeros(Float64,sum(N),sum(N))
    InvWNr = Float64[]
    DInvWNr = Float64[]

    for dim = 1:ndim
        n = N[dim]
        J = hcat(0:n-1)
        K = [hcat(0:(n-1)/2); hcat(-(n-1)/2:-1)]
        WN = exp.(-1im*(J*K')*2*pi/n)
        WNr = zeros(Float64,n,n)
        WNr[1,:] = ones(Float64,n,1)
        InvWNr = zeros(Float64,n,n)
        InvWNr[:,1] = 1/2*ones(Float64,n,1)
        DInvWNr = zeros(Float64,n,n)

        for kk in 1:Int((n-1)/2)
            WNr[2*kk,:] = cos.(J*2*pi*kk/n)
            WNr[2*kk+1,:]= -sin.(J*2*pi*kk/n)
            InvWNr[:,2*kk] = cos.(J*2*pi*kk/n)
            InvWNr[:,2*kk+1] = -sin.(J*2*pi*kk/n)
            DInvWNr[:,2*kk] = kk*sin.(J*2*pi*kk/n)
            DInvWNr[:,2*kk+1] = kk.*cos.(J*2*pi*kk/n)
        end
        InvWNr *= 2/n
        DInvWNr *= -2/n

        if dim == 1
            W = WN
            Wr = WNr
        else
            W = kron(W,WN)
            Wr = kron(Wr,WNr)
        end

        if dim == 1
            InvWr[1:N[1],1:N[1]] = InvWNr
            DInvWr[1:N[1],1:N[1]] = DInvWNr
        else
            InvWr[sum(N[1:dim-1])+1:sum(N[1:dim]),sum(N[1:dim-1])+1:sum(N[1:dim])] = InvWNr
            DInvWr[sum(N[1:dim-1])+1:sum(N[1:dim]),sum(N[1:dim-1])+1:sum(N[1:dim])] = DInvWNr
        end
    end

    InvW = 1/prod(N)*W'

    W = kron(W,I(d))
    InvW = kron(InvW,I(d))

    dW = zeros(Float64,d*prod(N),d*prod(N),length(N))

    if ndim == 1
        dW[:,:,1] = kron(DInvWr*Wr,I(d))
    else
        for dim = 1:ndim
            B = Float64[]
            for noper = 1:ndim
                if noper == 1
                    id0 = 1:N[1]
                else
                    id0 = sum(N[1:noper-1])+1:sum(N[1:noper])
                end

                if noper == 1
                    if dim == 1
                        B = DInvWr[id0,id0]
                    else
                        B = InvWr[id0,id0]
                    end
                else
                    if noper == dim
                        B = kron(B,DInvWr[id0,id0])
                    else
                        B = kron(B,InvWr[id0,id0])
                    end
                end
            end
            dW[:,:,dim] = kron(B*Wr,I(d))
        end
    end

    return W, InvW, dW
end

function ndRotMat(d::Int64,ρ,N) 
    Q = Float64[]
    for dim = 1:length(N)
        K = vcat(0:(N[dim]-1)/2,-(N[dim]-1)/2:-1)
        if dim == 1
            Q = Diagonal(exp.(-1im*K*ρ[dim]))
        else
            Q =  kron(Q,Diagonal(exp.(-1im*K*ρ[dim])))
        end
    end

    Q = kron(Q,I(d))
    return Q
end

function SparseIDs(d::Int64,N::Array{Int64,1},n::Int64)
    dTor = length(N)+1
    p = prod(N)
    D = p*d
    Dn = D*n
    if n > 1
        # multiple-shooting
        r = zeros(Int64,2*n*D^2+((dTor+2)*n+dTor-1)*D+5*dTor-1)
        c = zeros(Int64,2*n*D^2+((dTor+2)*n+dTor-1)*D+5*dTor-1)
    else
        # single-shooting
        r = zeros(Int64,D^2+(2*dTor+1)*D+5*dTor-1)
        c = zeros(Int64,D^2+(2*dTor+1)*D+5*dTor-1)
    end

    # Quasi-periodicity and Continuity constraints
    for kk = 1:n
        idx = D*(kk-1)+1:D*kk

        if n > 1
            # Continuity - MS
            r[D^2*(kk-1)+1:D^2*kk] = vec(repeat(hcat(idx),D,1))
            c[D^2*(kk-1)+1:D^2*kk] = vec(repeat(hcat(idx)',D,1))
        end

        if kk == 1
            # Quasi-periodicity
            if n > 1
                # MS
                ad_id = (2*n-1)*D^2 + (n-1)*D
                r[ad_id.+(1:D^2)] = vec(repeat(hcat(1:D),D,1))
                c[ad_id.+(1:D^2)] = vec(repeat(hcat((n-1)*D+1:Dn)',D,1))

                for jj in 1:dTor
                    ad_id = 2*n*D^2 + (n-1)*D + (jj-1)*D
                    r[ad_id.+(1:D)] = vcat(1:D)
                    c[ad_id.+(1:D)] = (Dn+jj)*ones(Int64,D)
                end
            else
                # SS
                r[1:D^2] = vec(repeat(hcat(1:D),D,1))
                c[1:D^2] = vec(repeat(hcat(1:D)',D,1))

                for jj in 1:dTor
                    ad_id = D^2 + (jj-1)*D
                    r[ad_id.+(1:D)] = vcat(1:D)
                    c[ad_id.+(1:D)] = (D+jj)*ones(Int64,D)
                end
            end
        else
            # Continuity - MS
            ad_id = n*D^2
            r[ad_id.+(D^2*(kk-2)+1:D^2*(kk-1))] = vec(repeat(hcat(idx),D,1))
            c[ad_id.+(D^2*(kk-2)+1:D^2*(kk-1))] = vec(repeat(hcat(D*(kk-2)+1:D*(kk-1))',D,1))

            ad_id = (2*n-1)*D^2
            r[ad_id.+(D*(kk-2)+1:D*(kk-1))] = vec(hcat(idx))
            c[ad_id.+(D*(kk-2)+1:D*(kk-1))] = (Dn+1)*ones(Int64,D)
        end
    end

    # Phase constraints
    if n > 1
        ad_id = 2*n*D^2 + (n+dTor-1)*D - D*n
    else
        ad_id = D^2 + (dTor-1)*D
    end
    for kk in 1:dTor
        ad_id += D*n
        r[ad_id.+(1:Dn)] = (Dn+kk)*ones(Int64,Dn)
        c[ad_id.+(1:Dn)] = vcat(1:Dn)
    end

    # Consistency constraints
    if n > 1
        ad_id = 2*n*D^2 + ((dTor+1)*n+dTor-1)*D
    else
        ad_id = D^2 + 2*dTor*D
    end
    for kk in 1:dTor
        if kk == 1
            r[ad_id.+(1:2)] = (Dn+dTor+kk)*ones(Int64,2)
            c[ad_id.+(1:2)] = [Dn+1;Dn+dTor+1]
            ad_id += 2
        else
            r[ad_id.+(1:3)] = (Dn+dTor+kk)*ones(Int64,3)
            c[ad_id.+(1:3)] = [Dn+1;Dn+kk;Dn+dTor+kk]
            ad_id += 3
        end
    end

    # Pseudo-arclength constraints
    if n > 1
        ad_id = 2*n*D^2 + ((dTor+1)*n+dTor-1)*D + 3*dTor - 1
    else
        ad_id = D^2 + 2*dTor*D + 3*dTor - 1
    end
    r[ad_id.+(1:Dn+2*dTor)] = (Dn+2*dTor+1)*ones(Int64,Dn+2*dTor)
    c[ad_id.+(1:Dn+2*dTor)] = vcat(1:Dn+2*dTor)

    return r, c
end

function Resample(x,d,n,P0,Pnew)
    # x: Array to rescale
    # d: size of phase space
    # n: number of multiple shooting segments
    # P0: initial number of Fourier modes
    # Pnew: new number of Fourier modes to resample to

    P0 = reverse(P0)
    D = d*prod(P0)
    Dn = D*n
    δ = size(x,1)-Dn

    Pnew = reverse(Pnew)
    pnew = prod(Pnew)
    Dnew = d*pnew
    Dnnew = Dnew*n
    
    c = size(x,2) # number columns in data set
    if c > 1
        y = zeros(Float64,Dnnew+δ,c)
        y[Dnnew+1:end,:] = x[Dn+1:end,:]
    else
        y = zeros(Float64,Dnnew+δ)
        y[Dnnew+1:end] = x[Dn+1:end]
    end
    
    P = copy(P0)

    for k in 1:c
        for j in 1:n
            for h in 1:d
                tmp = reshape(x[D*(j-1)+h:d:D*j,k],P0...)
                for m in 1:length(P0)
                    idxArr = [1:g for g in P]
                    idxArr[m] = 1:1
                    idxTup = Tuple(g for g in idxArr)
                    tmp = cat(tmp,tmp[idxTup...],dims=m)
                    P[m] += 1
                end
                P = copy(P0)
                rangeθ0 = Tuple(range(0,stop=2*pi,length=g) for g in (P0.+1))
                rangeθnew = Tuple(range(0,stop=2*pi,length=g)[1:end-1] for g in (Pnew.+1))
                sitp = scale(interpolate(tmp,BSpline(Cubic(Periodic(OnGrid())))),rangeθ0...)
                y[Dnew*(j-1)+h:d:Dnew*j,k] = reshape(sitp(rangeθnew...),pnew)
            end
        end
    end
    return y
end

function zeta(ds,N,dsMax::Float64,dsMin::Float64,Nmax::Int64,Nopt::Int64)
    w = collect(range(0,stop=1,length=length(ds)+1)[2:end])
    w ./= 1*sum(w)
    m = ones(Float64,length(ds)) # multiplier for step size dynamics
    n = ones(Float64,length(ds)) # multiplier for number of iterations dynamics
    dsflag = 0
    Nflag = 0
    # construct performance multipliers
    for k in eachindex(ds)
        if k == 1
            continue
        end
        if ds[k] < ds[k-1]
            if dsflag == -1
                m[k] = 0.95*m[k-1]
            else
                m[k] = 0.95
            end
            dsflag = -1
        elseif ds[k] > ds[k-1]
            if dsflag == 1
                m[k] = 1.05*m[k-1]
            else
                m[k] = 1.05
            end
            dsflag = 1
        else
            Nflag = 0
        end
        if N[k] > N[k-1]
            if Nflag == -1
                n[k] = 0.9*n[k-1]
            else
                n[k] = 0.9
            end
            Nflag = -1
        elseif N[k] < N[k-1]
            if Nflag == 1
                n[k] = 1.1*n[k-1]
            else
                n[k] = 1.1
            end
            Nflag = 1
        else
            Nflag = 0
        end
    end
    # compute performance for ds
    A = [log10(dsMin)^3 log10(dsMin)^2 log10(dsMin) 1;
        log10(dsMax)^3 log10(dsMax)^2 log10(dsMax) 1;
        (log10(dsMax*dsMin)/2)^3 (log10(dsMax*dsMin)/2)^2 log10(dsMax*dsMin)/2 1;
        3*log10(dsMax)^2 2*log10(dsMax) 1 0]
    b = [0.2;1;0.6;0]
    c = A\b
    σ = vec(hcat(log10.(ds).^3,log10.(ds).^2,log10.(ds),ones(length(ds)))*c)
    # σ = (1 .+(log10.(ds).-log10(dsMax))./log10(dsMax/dsMin)).^2
    # compute performance for number of iterations
    A = [1 1 1 1;Nopt^3 Nopt^2 Nopt 1;Nmax^3 Nmax^2 Nmax 1;3*Nopt^2 2*Nopt 1 0]
    b = [2;1;0;0]
    c = A\b
    ρ = vec(hcat(N.^3,N.^2,N,ones(Float64,length(N)))*c)
    # ρ = ((Nmax.+Nopt.-N)./Nmax).^3
    # compute performance index
    ζ = (m.*σ.*n.*ρ)'*w
    return (ζ,mean(σ),mean(ρ),mean(m),mean(n))
end

function SmallDenom(x::Float64,δ::Float64)
    n = 0 # numerator
    d = 1 # denominator
    pl = 0
    ql = 1
    pr = 1
    qr = 0
    count = 0
    while abs(x-n/d) >= δ
        count += 1
        n = pl+pr
        d = ql+qr
        if x < n/d
            pr = n
            qr = d
        else
            pl = n
            ql = d
        end
        if count == 10^4
            break
        end
    end
    return (n,d)
end

function BestStepSize(ω0,Δω,Δsmin,Δsmax,δ)
    nmax = 1000
    ncomb = length(combinations(ω0,2))
    Δss = range(Δsmin,stop=Δsmax,length=nmax)
    denom = zeros(Int64,ncomb,nmax)
    d = zeros(Int64,ncomb)
    for (id1,ds) in enumerate(Δss)
        ω = ω0 + ds*Δω
        for (id2,p) in enumerate(combinations(ω,2))
            (_,d[id2]) = SmallDenom(p[1]/p[2],δ)
        end
        denom[:,id1] = d
    end
    μdevω = vec(mean(abs.(log10.(denom).+log10(δ)/2),dims=1))
    (_,I) = findmin(μdevω)
    return Δss[I], denom
end

function InvSurfAmp(x,d,N)
    dSurf = length(N)
    amp = zeros(Float64,dSurf)
    x = reshape(x,d,prod(N))
    if dSurf >= 2
        for k in dSurf:-1:2
            cent = zeros(Float64,d,N[k-1])
            avgrad = zeros(Float64,N[k-1])
            p = N[k]
            for j in 1:N[k-1]
                cent[:,j] = mean(x[:,(p*(j-1)+1):(p*j)],dims=2)
                ar = mean(sqrt.(sum((x[:,p*(j-1)+1:p*j].-cent[:,j]).^2,dims=1)),dims=2)
                avgrad[j] = ar[1]
            end
            amp[k] = mean(avgrad)
            x = cent
        end
    end
    cent = mean(x,dims=2)
    ar = mean(sqrt.(sum((x.-cent).^2,dims=1)),dims=2)
    amp[1] = ar[1]
    return amp
end

function GMOS(Z0::Array{Float64,1},dZ0::Array{Float64,1},pars::Dict{String,Any})
    ### Parameters ###
    d     = pars["d"]::Int64 # dimension of each point in invaraint curve
    ds    = pars["GMOS"]["ds"]::Float64 # step-length
    dsMax = pars["GMOS"]["dsMax"]::Float64 # Max step-length
    dsMin = pars["GMOS"]["dsMin"]::Float64 # Min step-length
    MaxIter  = pars["GMOS"]["Iter"]::Int64 # Max # of iterations allowed
    n = pars["GMOS"]["n"]::Int64 # # multiple shooting segments
    M     = pars["GMOS"]["M"]::Int64 # # of time nodes
    P0     = pars["GMOS"]["N"]::Array{Int64,1} # # of GMOS points
    Pmax = pars["GMOS"]["Pmax"]::Array{Int64,1} # max # of GMOS points
    Pmin = pars["GMOS"]["Pmin"]::Array{Int64,1} # min # of GMOS points
    Nmax  = pars["GMOS"]["Nmax"]::Int64 # # of family members to be computed

    p = prod(P0) # number points in one invariant curve
    D = d*p # dimension of strob map
    pn = p*n # total number of points in invariant curves
    Dn = D*n # dimension of the problem
    dTor = length(P0)+1 # dimension of torus
    dFam = length(keys(pars["ParCon"]))+1 # dimension of family
    STMIC = Matrix(1.0I,d,d) # STM initial condition
    FTid = collect(1:(Dn+2*dTor+dFam)) # indices to keep for computing family tangent 
    deleteat!(FTid,Dn+2*dTor+1) # (remove pseudo-arclength)

    P = copy(P0)
    Pnew = copy(P)

    # Define initial guess
    U0 = Z0[1:Dn]
    T0 = Z0[Dn+1]
    ρ0 = Z0[Dn+2:Dn+dTor]
    ω0 = Z0[Dn+dTor+1:end]
    
    # Initialize Output
    if Nmax != Inf
        Xqp = Vector{Matrix{Float64}}(undef,Nmax)
        Wqp = zeros(Float64,2*dTor,Nmax)
        if pars["GMOS"]["stab"]
            Bqp = Vector{Matrix{Float64}}(undef,Nmax)
        else
            Bqp = []
        end
        Zqp = Vector{Vector{Float64}}(undef,Nmax)
        if pars["GMOS"]["TS"]
            Vqp = Vector{Matrix{Float64}}(undef,Nmax)
        else
            Vqp = []
        end
        Sqp = zeros(Float64,Nmax)
        Pqp = zeros(Int64,dTor-1,Nmax)
        ζqp = zeros(Float64,Nmax)
    else
        Xqp = Vector{Matrix{Float64}}(undef,1)
        Wqp = zeros(Float64,2*dTor,1)
        if pars["GMOS"]["stab"]
            Bqp = Vector{Matrix{Float64}}(undef,1)
        else
            Bqp = []
        end
        Zqp = Vector{Vector{Float64}}(undef,1)
        if pars["GMOS"]["TS"]
            Vqp = Vector{Matrix{Float64}}(undef,1)
        else
            Vqp = []
        end
        Sqp = zeros(Float64,1)
        Pqp = zeros(Int64,dTor-1,1)
        ζqp = zeros(Float64,1)
    end
    Iqp = []
    Σqp = []

    # Initialize Temporary Arrays
    t0 = zeros(Float64,pn)
    tf = zeros(Float64,pn)
    dUT = zeros(Float64,D,n,dTor)
    ft = zeros(Float64,Dn)
    Xt = zeros(Float64,Dn,M)
    PHItr = zeros(Float64,d^2*pn)
    PHItc = zeros(Float64,d^2*pn)
    PHItv = zeros(Float64,d^2*pn)
    Ut = zeros(Float64,Dn)
    Ur = zeros(Float64,Dn)
    F = zeros(Float64,Dn+2*dTor+dFam)
    (DFr,DFc) = SparseIDs(d,P0,n)
    DFid = length(DFr)
    DFv = zeros(Float64,DFid)
    FTVec = vcat(zeros(Float64,Dn+2*dTor+dFam-1),1.0)
    TanSpace = zeros(Float64,Dn+dTor,dTor)
    R = zeros(Float64,D,D)
    Amp = zeros(Float64,dTor-1)
    ζ = 0.0

    # Indices for STM
    for kk in 1:pn
        PHItr[d^2*(kk-1)+1:d^2*kk] = vec(repeat(hcat(d*(kk-1)+1:d*kk),1,d))
        PHItc[d^2*(kk-1)+1:d^2*kk] = vec(repeat(hcat(d*(kk-1)+1:d*kk)',d,1))
    end

    if pars["GMOS"]["txt"]=="verbose"
        if n > 1
            printstyled("GMOS ",dTor,"-D (Multiple-shooting)\n",color=:magenta)
        else
            printstyled("GMOS ",dTor,"-D (Single-shooting)\n",color=:magenta)
        end
    end

    sols = 0 # number of solutions
    Pcount = 0 # counts the number of solutions since last change in P
    attempts = 0 # counts the number of attempts with no convergence
    TSflag = false # tangent space flag
    ΔPflag = true # change in # Fourier coefficients flag
    conv = true # Convergence Indicator
    returnflag = false # flag to determine if the program should terminate
    NegativeFlag = false # flag indicates negative frequencies
    

    # Iterate over solutions
    while true
        # Track solution number
        sols += 1
        Pcount += 1

        # Iterate over trials
        while true
            attempts += 1
            # compute new step size to avoid resonances
            if pars["GMOS"]["RAv"] && !conv
                ds, _ = BestStepSize(ω0,dZ0[Dn+dTor+1:end],0.9*ds,1.1*ds,10^(-8))
            end

            if pars["GMOS"]["txt"]=="verbose" && conv
                printstyled("Family member No. ",sols,",\n",color=:cyan)
            end
            if pars["GMOS"]["txt"]=="verbose"
                printstyled("Attempt No. ",attempts,". P = ",P,", ds = ",ds,"\n")
            end

            if conv || ΔPflag
                # Time vector
                t = range(0.0,stop=T0,length=n+1)
                for jj in 1:n
                    t0[p*(jj-1)+1:p*jj] = t[jj].*ones(p)
                end

                # Vector Field
                pars["VF"](ft,U0,pars,t0)

                # Partial Derivatives for phase constraints
                dUT[:,:,1] = 1/ω0[1].*reshape(ft,D,n)
                for jj in 2:dTor
                    dUT[:,:,jj] = pars["GMOS"]["DT"][:,:,jj-1]*reshape(U0,D,n)
                    dUT[:,:,1] += -ω0[jj]/ω0[1].*dUT[:,:,jj]
                end

                # Update convergence flag
                conv = false
            end

            # Prediction
            U = U0 .+ ds*dZ0[1:Dn]
            T = T0 + ds*dZ0[Dn+1]
            ρ = ρ0 .+ ds*dZ0[Dn+2:Dn+dTor]
            ω = ω0 .+ ds*dZ0[Dn+dTor+1:end]

            # Newton iterations
            for iter in 1:(Pcount<=3 ? MaxIter+2 : MaxIter)
                
                # Rotation matrix
                R = real(pars["GMOS"]["IDFT"]*ndRotMat(d,ρ,P)*pars["GMOS"]["DFT"])

                # Time vector
                t = range(0.0,stop=T,length=n+1)
                for jj in 1:n
                    t0[p*(jj-1)+1:p*jj] = t[jj]*ones(Float64,p)
                    tf[p*(jj-1)+1:p*jj] = t[jj+1]*ones(Float64,p)
                end

                # Integrate GMOS points individually (work on parallelizing this)
                for kk in 1:pn
                    idx = d*(kk-1)+1:d*kk

                    IC = hcat(U[idx],STMIC)
                    prob = ODEProblem(pars["EOM"],IC,(t0[kk],tf[kk]),pars)
                    sol = solve(prob,VCABM(),reltol=2.3e-14,abstol=1e-16,saveat=range(t0[kk],stop=tf[kk],length=M+1))

                    Ut[idx] = sol[:,1,end]
                    Xt[idx,:] = sol[:,1,1:end-1]
                    PHItv[d^2*(kk-1)+1:d^2*kk] = reshape(sol[:,2:end,end],d^2)
                end

                # STM
                PHIn = dropzeros(sparse(PHItr,PHItc,PHItv))

                # Update vector field at Ut
                pars["VF"](ft,Ut,pars,t0)

                # Rotate points
                for kk in 1:n
                    Ur[D*(kk-1)+1:D*kk] = vec(R*Ut[D*(kk-1)+1:D*kk])
                end

                # Quasi-periodicity constraints
                F[1:D] = Ur[D*(n-1)+1:Dn] .- U[1:D]

                # Continuity constraints - multiple shooting
                if n > 1
                    F[D+1:Dn] = Ut[1:D*(n-1)] .- U[D+1:Dn]
                end

                # Jacobian - Quasi-periodicity and Continuity constraints
                for kk in 1:n

                    if n > 1
                        # Continuity - multiple shooting
                        DFv[D^2*(kk-1)+1:D^2*kk] = vec(-I(D))
                    end

                    if kk == 1
                        # Quasi-periodicity
                        if n > 1
                            # multiple shooting
                            ad_id = (2*n-1)*D^2 + (n-1)*D
                            DFv[ad_id.+(1:D^2)] = vec(R*PHIn[(n-1)*D+1:Dn,(n-1)*D+1:Dn])
                            
                            for jj in 1:dTor
                                ad_id = 2*n*D^2+(n-1)*D + (jj-1)*D
                                if jj == 1
                                    DFv[ad_id.+(1:D)] = vec(R*ft[D*(n-1)+1:Dn]/n)
                                else
                                    DFv[ad_id.+(1:D)] = vec(-pars["GMOS"]["DT"][:,:,jj-1]*Ur[D*(n-1)+1:Dn])
                                end
                            end
                        else
                            # single shooting
                            DFv[1:D^2] = vec(R*PHIn .- I(D))

                            for jj in 1:dTor
                                ad_id = D^2 + (jj-1)*D
                                if jj == 1
                                    DFv[ad_id.+(1:D)] = vec(R*ft)
                                else
                                    DFv[ad_id.+(1:D)] = vec(-pars["GMOS"]["DT"][:,:,jj-1]*reshape(Ur,D,n))
                                end
                            end
                        end
                    else
                        # Continuity - multiple shooting
                        ad_id = n*D^2
                        DFv[ad_id.+(D^2*(kk-2)+1:D^2*(kk-1))] = vec(PHIn[D*(kk-2)+1:D*(kk-1),D*(kk-2)+1:D*(kk-1)])

                        ad_id = (2*n-1)*D^2
                        DFv[ad_id.+(D*(kk-2)+1:D*(kk-1))] = vec(ft[D*(kk-2)+1:D*(kk-1)]/n)
                    end
                end

                # Phase constraints
                if n > 1
                    ad_id = 2*n*D^2 + (n+dTor-1)*D - Dn
                else
                    ad_id = D^2 + (dTor-1)*D
                end
                for kk in 1:dTor
                    if kk == 1
                        F[Dn+1] = dot(U.-U0,vec(dUT[:,:,1]))/pn
                    else
                        F[Dn+kk] = dot(U,vec(dUT[:,:,kk]))/pn
                    end
                    ad_id +=  Dn
                    DFv[ad_id.+(1:Dn)] = vec(dUT[:,:,kk]./pn)
                end

                # Consistency constraints
                F[Dn+dTor+1] = T*ω[1]-2*pi
                if n > 1
                    ad_id = 2*n*D^2 + ((dTor+1)*n+dTor-1)*D
                else
                    ad_id = D^2+2*dTor*D
                end
                DFv[ad_id.+(1:2)] = [ω[1];T]
                ad_id += 2
                for kk in 2:dTor
                    F[Dn+dTor+kk] = T*ω[kk]-ρ[kk-1]
                    DFv[ad_id.+(1:3)] = [ω[kk];-1;T]
                    ad_id += 3
                end
                
                # Pseudo-arclength constraints
                F[Dn+2*dTor+1] = dot(U.-U0,dZ0[1:Dn])/pn + (T-T0)*dZ0[Dn+1] + (ρ.-ρ0)'*dZ0[(Dn+2):(Dn+dTor)] + (ω.-ω0)'*dZ0[(Dn+dTor+1):end] - ds
                if n > 1
                    ad_id = 2*n*D^2 + ((dTor+1)*n+dTor-1)*D + 3*dTor - 1
                else
                    ad_id = D^2+2*dTor*D + 3*dTor - 1
                end
                DFv[ad_id.+(1:Dn+2*dTor)] = vcat(dZ0[1:Dn]/pn,dZ0[(Dn+1):end])
                
                # Parametric constraints
                ad_id = DFid
                for kk in 1:dFam-1
                    (F[Dn+2*dTor+1+kk],df,cid) = pars["ParCon"][kk](vcat(U,T,ρ,ω),vcat(U0,T0,ρ0,ω0),dZ0,pars)
                    if ΔPflag
                        append!(DFr,(Dn+2*dTor+1+kk)*ones(Int64,length(cid)))
                        append!(DFc,cid)
                        append!(DFv,df)
                        ΔPflag = false
                    else
                        if length(cid) == 1
                            DFv[ad_id+1] = df
                        else
                            DFv[ad_id.+(1:length(cid))] = df
                        end
                        ad_id += length(cid)
                    end
                end

                DF = dropzeros(sparse(DFr,DFc,DFv))

                # Newton update
                z = -DF\F

                normF = sqrt(dot(F[1:Dn],F[1:Dn])/pn + dot(F[Dn+1:end],F[Dn+1:end]))
                if pars["GMOS"]["txt"]=="verbose"
                    normz = sqrt(dot(z[1:Dn],z[1:Dn])/pn + dot(z[Dn+1:end],z[Dn+1:end]))
                    @printf("|F|* = %.4e, |z|* = %.4e, arc = %.4e\n",normF,normz,F[Dn+2*dTor+1])
                end

                # Check if error is under tolerance
                if normF < pars["GMOS"]["Tol"]
                    # we converged! :)
                    if sum(ω.<=0) > 0
                        printstyled("Program terminating because frequencies are negative.\n",color=:red)
                        NegativeFlag = true
                        break
                    end

                    if pars["GMOS"]["TS"]
                        # check for tangent space
                        if !TSflag
                            TanSpace = svd(Matrix(DF[1:Dn+2*dTor,:]))
                            if TanSpace.S[end-dFam+1] < 1e-8
                                TanSpace = Matrix(Transpose(TanSpace.Vt[end-dFam+1:end,:]))
                                TSflag = true
                            else
                                # No tangent space - apply Newton update and iterate again
                                U += reshape(z[1:Dn],D,n)
                                T += z[Dn+1]
                                ρ += z[Dn.+(2:dTor)]
                                ω += z[Dn+dTor+1:end]
                                if pars["GMOS"]["txt"]==true || pars["GMOS"]["txt"]=="verbose"
                                    printstyled("Tangent Space not found. Applying Newton update.\n",color=:yellow)
                                end
                                continue
                            end
                        else
                            # compute new tangent space
                            TanSpace = Matrix(qr([Matrix(DF[1:Dn+2*dTor,:]);TanSpace']\[zeros(Dn+2*dTor,dFam);I(dFam)]).Q)
                        end
                        # normalize vectors
                        TanSpace ./= sqrt.(sum(TanSpace[1:Dn,:].^2,dims=1)/pn + sum(TanSpace[Dn+1:end,:].^2,dims=1))
                    end

                    push!(Iqp,iter)
                    push!(Σqp,ds)

                    # stability
                    if pars["GMOS"]["stab"]
                        Φ = I(D)
                        for kk in 1:n
                            Φ = PHIn[D*(kk-1)+1:D*kk,D*(kk-1)+1:D*kk]*Φ
                        end
                        B = R*Φ
                    end
                    
                    # Family tangent
                    dz = vcat(DF[FTid,:],dZ0')\FTVec
                    if dot(dz,dZ0) < 0
                        dz *= -1
                    end
                    dZ0 = dz./sqrt(dot(dz[1:Dn],dz[1:Dn])/pn + dot(dz[Dn+1:end],dz[Dn+1:end]))

                    # Step size controller
                    ε = pars["GMOS"]["Opt"]/iter

                    # step size adjustment
                    if ε > 2
                        ε = 2
                    elseif ε < 0.5
                        ε = 0.5
                    end
                    ds = min(ε*ds,dsMax)
                    
                    if pars["GMOS"]["AFM"]
                        if Pcount >= 10
                            (ζ,_,_,_,_) = zeta(Σqp[sols-9:sols],Iqp[sols-9:sols],dsMax,dsMin,MaxIter,pars["GMOS"]["Opt"])

                            # check if solution needs rescaling
                            # step sizes are continually getting smaller(larger) and we're not at max(min) Fourier modes
                            if (ζ>=1.3 && P>Pmin) || (ζ<=0.5 && P<Pmax)
                                Pnew = ceil.(Int,P./((1+ζ)/2))
                                for kk in eachindex(Pnew)
                                    # ensure odd number of modes
                                    if mod(Pnew[kk],2) == 0
                                        Pnew[kk] -= 1
                                    end
                                    if Pnew[kk] > Pmax[kk]
                                        Pnew[kk] = Pmax[kk]
                                    elseif Pnew[kk] < Pmin[kk]
                                        Pnew[kk] = Pmin[kk]
                                    end
                                end
                                ds /= 2
                                ΔPflag = true
                                Pcount = 0
                            end
                        else
                            (ζ,_,_,_,_) = zeta(Σqp[sols-Pcount+1:sols],Iqp[sols-Pcount+1:sols],dsMax,dsMin,MaxIter,pars["GMOS"]["Opt"])
                        end
                    end

                    # compute better step size to avoid resonances
                    if pars["GMOS"]["RAv"]
                        ds, _ = BestStepSize(ω,dZ0[Dn+dTor+1:end],0.9*ds,1.1*ds,10^(-8))
                    end

                    # Store results
                    if Nmax != Inf
                        Xqp[sols] = Xt
                        Wqp[:,sols] = vcat(T,ρ,ω)
                        if pars["GMOS"]["stab"]
                            Bqp[sols] = B
                        end
                        Zqp[sols] = dZ0
                        if pars["GMOS"]["TS"]
                            Vqp[sols] = TanSpace
                        end
                        Sqp[sols] = ds
                        Pqp[:,sols] = P
                        ζqp[sols] = ζ
                    else
                        if sols == 1
                            Xqp[1] = Xt
                            Wqp[:,1] = vcat(T,ρ,ω)
                            if pars["GMOS"]["stab"]
                                Bqp[1] = B
                            end
                            Zqp[1] = dZ0
                            if pars["GMOS"]["TS"]
                                Vqp[1] = TanSpace
                            end
                            Sqp[1] = ds
                            Pqp[:,1] = P
                            ζqp[1] = ζ
                        else
                            push!(Xqp,Xt)
                            Wqp = hcat(Wqp,vcat(T,ρ,ω))
                            if pars["GMOS"]["stab"]
                                push!(Bqp,B)
                            end
                            Zqp = push!(Zqp,dZ0)
                            if pars["GMOS"]["TS"]
                                push!(Vqp,TanSpace)
                            end
                            push!(Sqp,ds)
                            Pqp = hcat(Pqp,P)
                            push!(ζqp,ζ)
                        end
                    end
                    
                    # Update solution (continuation)
                    U0 = copy(U)
                    T0 = copy(T)
                    ρ0 = copy(ρ)
                    ω0 = copy(ω)
                    conv = true
                    attempts = 0

                    Amp = InvSurfAmp(U[1:D],d,P)

                    # check to see if the number of Fourier modes has changed
                    if ΔPflag
                        # rescale current solution
                        U0 = Resample(U0,d,n,P,Pnew)
                        dZ0 = Resample(dZ0,d,n,P,Pnew)
                        if pars["GMOS"]["TS"]
                            TanSpace = Resample(TanSpace,d,n,P,Pnew)
                        end
                        
                        # re-initialize necessary variables
                        pars["GMOS"]["N"] = Pnew
                        p = prod(Pnew) # number points in one invariant curve
                        D = d*p # dimension of strob map
                        pn = p*n # total number of points in invariant curves
                        Dn = D*n # dimension of the problem
                        FTid = collect(1:(Dn+2*dTor+dFam)) # indices to keep for computing family tangent 
                        deleteat!(FTid,Dn+2*dTor+1) # (remove pseudo-arclength)

                        t0 = zeros(Float64,pn)
                        tf = zeros(Float64,pn)
                        dUT = zeros(Float64,D,n,dTor)
                        ft = zeros(Float64,Dn)
                        Xt = zeros(Float64,Dn,M)
                        PHItr = zeros(Float64,d^2*pn)
                        PHItc = zeros(Float64,d^2*pn)
                        PHItv = zeros(Float64,d^2*pn)
                        Ut = zeros(Float64,Dn)
                        Ur = zeros(Float64,Dn)
                        F = zeros(Float64,Dn+2*dTor+dFam)
                        (DFr,DFc) = SparseIDs(d,Pnew,n)
                        DFid = length(DFr)
                        DFv = zeros(Float64,DFid)
                        FTVec = vcat(zeros(Float64,Dn+2*dTor+dFam-1),1.0)
                        (pars["GMOS"]["DFT"],pars["GMOS"]["IDFT"],pars["GMOS"]["DT"]) = ndDFT(d,Pnew)

                        # Indices for STM
                        for kk in 1:pn
                            PHItr[d^2*(kk-1)+1:d^2*kk] = vec(repeat(hcat(d*(kk-1)+1:d*kk),1,d))
                            PHItc[d^2*(kk-1)+1:d^2*kk] = vec(repeat(hcat(d*(kk-1)+1:d*kk)',d,1))
                        end

                        P = copy(Pnew)
                    end

                    if pars["GMOS"]["txt"]=="verbose"
                        printstyled("Quasi-periodic torus has been found!\n",color=:green)
                        if pars["GMOS"]["AFM"]
                            println("ζ = ",ζ)
                        end
                        printstyled("ω = ",ω," and A = ",Amp,"\n\n")
                    end

                    break # break from Newton iterations
                else
                    # Update solution (Newton)
                    U += z[1:Dn]
                    T += z[Dn+1]
                    ρ += z[Dn+2:Dn+dTor]
                    ω += z[Dn+dTor+1:end]
                end
            end # End of Newton iterations
            
            # Check for program end conditions
            if haskey(pars["GMOS"],"TermFunc")
                returnflag = pars["GMOS"]["TermFunc"]((U,T,ρ,ω,B),(sols,ds,P))
            end
            if ((sols==Nmax && conv) || (ds<(Pcount<=3 ? dsMin/5 : dsMin) && P==(pars["GMOS"]["AFM"] ? Pmax : P0)) ||
                Amp<pars["GMOS"]["Tol"]*ones(dTor-1) || returnflag || NegativeFlag)

                # Print termination cause
                if pars["GMOS"]["txt"]==true || pars["GMOS"]["txt"] == "verbose"
                    if !conv
                        sols -= 1
                        println("Program terminating because QPO could not be found within the set tolerances.\n")
                    elseif sols==Nmax && conv
                        println("Program terminating because maximum number of family members found.\n")
                    elseif Amp < pars["GMOS"]["Tol"]*ones(dTor-1)
                        println("Program terminating because an amplitude is too small.\n")
                    elseif returnflag
                        println("Program terminating because user defined termination function activated.\n")
                    else
                        println("Program termination unknown.\n")
                    end
                end

                # Modify the output
                if sols == 0
                    # no solutions found
                    Xqp = fill(NaN,(Dn,M))
                    Wqp = fill(NaN,2*dTor)
                    Bqp = fill(NaN,(D,D))
                    Zqp = fill(NaN,Dn+2*dTor)
                    Vqp = fill(NaN,(Dn+2*dTor,dFam))
                    Sqp = [NaN]
                    Σqp = [ds]
                    Pqp = P
                    Iqp = [NaN]
                    ζqp = [NaN]
                else
                    if Nmax != Inf && sols < Nmax
                        Xqp = Xqp[1:sols]
                        Wqp = Wqp[:,1:sols]
                        if pars["GMOS"]["stab"]
                            Bqp = Bqp[1:sols]
                        end
                        Zqp = Zqp[1:sols]
                        if pars["GMOS"]["TS"]
                            Vqp = Vqp[1:sols]
                        end
                        Sqp = Sqp[1:sols]
                        Σqp = Σqp[1:sols]
                        Pqp = Pqp[:,1:sols]
                        Iqp = Iqp[1:sols]
                        ζqp = ζqp[1:sols]
                    end
                end
                return Xqp, Wqp, Bqp, Zqp, Vqp, Sqp, Σqp, Pqp, Iqp, ζqp
            end

            # Check conditions to retry computing solution
            if !conv
                # Check if we are near a resonance. How to move past this? Is there a reasonable
                # step size to move past it without changing direction?
                if pars["GMOS"]["AFM"] && ds<=(Pcount<=3 ? dsMin/5 : dsMin) && P<Pmax
                    # compute new step size
                    ds = dsMin/2

                    # increase the Fourier modes
                    Pnew = floor.(Int,P.*1.4)
                    for kk in eachindex(Pnew)
                        if mod(Pnew[kk],2) == 0
                            Pnew[kk] -= 1
                        end
                        if Pnew[kk] > Pmax[kk]
                            Pnew[kk] = Pmax[kk]
                        end
                    end

                    if pars["GMOS"]["txt"]==true || pars["GMOS"]["txt"]=="verbose"
                        printstyled("Increasing Fourier modes.\n",color=:yellow)
                    end

                    # rescale current solution
                    U0 = Resample(U0,d,n,P,Pnew)
                    dZ0 = Resample(dZ0,d,n,P,Pnew)
                    if pars["GMOS"]["TS"]
                        TanSpace = Resample(TanSpace,d,n,P,Pnew)
                    end
                    
                    # re-initialize necessary variables
                    pars["GMOS"]["N"] = Pnew
                    p = prod(Pnew) # number points in one invariant curve
                    D = d*p # dimension of strob map
                    pn = p*n # total number of points in invariant curves
                    Dn = D*n # dimension of the problem
                    FTid = collect(1:(Dn+2*dTor+dFam)) # indices to keep for computing family tangent 
                    deleteat!(FTid,Dn+2*dTor+1) # (remove pseudo-arclength)

                    t0 = zeros(Float64,pn)
                    tf = zeros(Float64,pn)
                    dUT = zeros(Float64,D,n,dTor)
                    ft = zeros(Float64,Dn)
                    Xt = zeros(Float64,Dn,M)
                    PHItr = zeros(Float64,d^2*pn)
                    PHItc = zeros(Float64,d^2*pn)
                    PHItv = zeros(Float64,d^2*pn)
                    Ut = zeros(Float64,Dn)
                    Ur = zeros(Float64,Dn)
                    F = zeros(Float64,Dn+2*dTor+dFam)
                    (DFr,DFc) = SparseIDs(d,Pnew,n)
                    DFid = length(DFr)
                    DFv = zeros(Float64,DFid)
                    FTVec = vcat(zeros(Float64,Dn+2*dTor+dFam-1),1.0)
                    (pars["GMOS"]["DFT"],pars["GMOS"]["IDFT"],pars["GMOS"]["DT"]) = ndDFT(d,Pnew)

                    # Indices for STM
                    for kk in 1:pn
                        PHItr[d^2*(kk-1)+1:d^2*kk] = vec(repeat(hcat(d*(kk-1)+1:d*kk),1,d))
                        PHItc[d^2*(kk-1)+1:d^2*kk] = vec(repeat(hcat(d*(kk-1)+1:d*kk)',d,1))
                    end

                    P = copy(Pnew)
                    ΔPflag = true
                    Pcount = 1
                elseif ds > (Pcount<=3 ? dsMin/5 : dsMin)
                    # retry with smaller step size
                    ds /= 4

                    if pars["GMOS"]["txt"]==true || pars["GMOS"]["txt"]=="verbose"
                        printstyled("Decreasing step size.\n",color=:yellow)
                    end
                end
            else
                # continue to next family member
                break # break from trials
            end
        end # End of trials
    end # End of continuation
    println("Program reached outside of the continuation loop. This shouldn't happen.\n")
    return Xqp, Wqp, Bqp, Zqp, Vqp, Sqp, Σqp, Pqp, Iqp, ζqp # should have terminated before this point, but just in case
end

end
