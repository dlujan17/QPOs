import math, numpy, scipy, scipy.special, scipy.sparse, scipy.integrate
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from system_mu import mu

def LagrangePolynomial(t,tm,jj):
    '''Calculate Lagrange Basis Polynomials'''
    
    # [val,dval] = LagrangePolynomial(t,tm,jj)
    # 
    # DEPENDENCIES: none
    # AUTHOR: David Lujan copied from Matlab originally by Nicola Baresi
    # CONTACT: david.lujan@colorado.edu

    M = tm.size
    
    if jj >= M:
        raise ValueError('jj index exceeds length of time mesh points!')

    # Initialization
    den  = 1
    val  = 1
    dval = 0


    # Compute Polynomial
    for mm in range(0,M):
        if(mm != jj):
            den *= (tm[jj]-tm[mm])
            val *= (t - tm[mm])
            
    # Compute Polynomial Derivative
    for mm in range(0,M):
        tmp = 1
        if(mm != jj):
            for nn in range(0,M):
                if(nn != mm and nn != jj):
                    tmp *= (t - tm[nn])

            dval = dval + tmp

    # Output val and dval
    val = val/den
    dval = dval/den

    return val, dval

def LagrangeMatrix(T,t):
    '''Calculate Lagrange Matrices'''
    # [L,DL] = LagrangeMatrix(T,t)
    #
    # DEPENDENCIES: LagrangePolynomial.py
    # AUTHOR: David Lujan copied from Matlab originally by Nicola Baresi
    # CONTACT: david.lujan@colorado.edu


    # Initialization
    M  = t.size
    L  = numpy.zeros(M)
    DL = numpy.zeros(M)

    for mm in range(0,M):
        L[mm], DL[mm] = LagrangePolynomial(T,t,mm)

    return L, DL
    
def FourierMatrix(n,N):
    '''Calculate Fourier Matrices given the dimension of the problem n and the
    number of GMOS solution points N'''

    # FourierMatrix(n,N)

    # INPUTS:
    #   n       dimension of problem (i.e. number of states)
    #   N       number of discrete points
    # OUTPUTS:
    #   R       Discrete Fourier Matrix transformation
    #   IR      Inverse Discrete Fourier Matrix transformation
    #   DR      Derivative Fourier Matrix transformation
    # 
    # DEPENDENCIES: none
    # AUTHOR: David Lujan copied from Matlab originally by Nicola Baresi
    # CONTACT: david.lujan@colorado.edu

    # Initialization 
    D  = numpy.zeros([n*N,n*N],dtype=complex)
    R  = numpy.zeros([n*N,n*N],dtype=complex)
    IR = numpy.zeros([n*N,n*N])
    DR = numpy.zeros([n*N,n*N])
    
    pi = numpy.pi
    
    # Order
    K = numpy.hstack([range(0,int((N-1)/2+1)),range(int(-(N-1)/2),0)])

    for ii in range(0,N):
     
        # k index
        kk = 0
    
        # Row Index
        id1 = range(n*ii,n*(ii+1))
    
        for jj in range(0,N):
        
            # Column Index
            id2 = range(n*jj,n*(jj+1))
        
            # DFT matrix
            D[numpy.ix_(id1,id2)] = numpy.eye(n)*numpy.exp(2*pi*K[ii]*K[jj]*1j/N)
        
            # IR matrix
            if (jj+1)%2 == 0:
                kk += 1
                IR[numpy.ix_(id1,id2)] = math.cos(2*pi*ii*K[kk]/N)*numpy.eye(n)
                DR[numpy.ix_(id1,id2)] = -K[kk]*math.sin(2*pi*ii*K[kk]/N)*numpy.eye(n)
            else:
                IR[numpy.ix_(id1,id2)] = math.sin(2*pi*ii*K[kk]/N)*numpy.eye(n)
                DR[numpy.ix_(id1,id2)] = K[kk]*math.cos(2*pi*ii*K[kk]/N)*numpy.eye(n)
        
        # A0
        IR[numpy.ix_(id1,range(0,n))] = numpy.eye(n)

    # From complex to real
    kk = 0
    for ii in range(2,N,2):
    
        # column index
        kk = kk + 1
    
        # Indexing
        row1 = range(n*(ii-1),n*ii)
        row2 = range(n*ii,n*(ii+1))
        col1 = range(n*kk,n*(kk+1))
        col2 = range(n*(N-kk),n*(N-kk+1))
    
        # Definitions
        R[numpy.ix_(row1,col1)] = numpy.eye(n)/N
        R[numpy.ix_(row1,col2)] = numpy.eye(n)/N
        R[numpy.ix_(row2,col1)] = -1j*numpy.eye(n)/N
        R[numpy.ix_(row2,col2)] = 1j*numpy.eye(n)/N

    R[numpy.ix_(range(0,n),range(0,n))] = numpy.eye(n)/N
    R = R@D
    if R.imag.max() > 1e-16:
        print('Error in R!\n')
        return

    R = R.real

    # Derivative Matrix
    DR = DR@R
    if DR.imag.max() > 1e-16:
        print('Error in DR!\n')
        return

    DR = DR.real

    outputs = {'R': R, 'invR': IR, 'derivR': DR}
    return outputs
    
def RotationMatrix(p,n,N):

    '''Generate Rotation matrix given the angle p, the dimension of the problem
    n, and the number of GMOS solution points, N'''
 
    # RotationMatrix(p,n,N)

    # Calcualte Indices
    K = numpy.hstack([range(0,int((N-1)/2+1)),range(int(-(N-1)/2),0)])

    # Initialization
    Q  = numpy.zeros([n*N,n*N])
    kk = 0

    for ii in range(2,N,2):
    
        # column index
        kk += 1
    
        # Indexing
        row1 = range(n*(ii-1),n*ii)
        row2 = range(n*ii,n*(ii+1))
    
        # Rotating Fourier Coefficients
        Q[numpy.ix_(row1,row1)] = numpy.eye(n)*math.cos(K[kk]*p)
        Q[numpy.ix_(row1,row2)] = -numpy.eye(n)*math.sin(K[kk]*p)
        Q[numpy.ix_(row2,row1)] = numpy.eye(n)*math.sin(K[kk]*p)
        Q[numpy.ix_(row2,row2)] = numpy.eye(n)*math.cos(K[kk]*p)

    # Add first block
    Q[numpy.ix_(range(0,n),range(0,n))] = numpy.eye(n)

    return Q
    
def GMOS_SingleShooting_InitialGuess(inputs):
    '''Generate Initial Guess for the GMOS single-shooting Method'''

    # GMOS_SingleShooting_InitialGuess(inputs)
 
    # INPUTs:
    # VARIABLE      TYPE             DESCRIPTION
    #   inputs      dict             input variable with fields:
    #   Xpo         1D np-array      Periodic Orbit Initial Conditions
    #   Tpo         float            Periodic Orbit Period
    #   Vpo         1D np-array      Center subspace Eigenvector
    #   Epo         float            Center subspace Eigenvalue
    #   pars        dict             List of Parameters
    #   'd'         int              No. of states
    #   'GMOS''ds'  float            Initial displacement along center subspace
    #   'GMOS''N'   int              No. of GMOS solution points
    # OUTPUTS:
    # VARIABLE     TYPE             DESCRIPTION
    #   outputs    dict             output variable with fields:
    #   Xqp0       2D np-array      QP Torus initial guess
    #   Wqp0       1D np-array      QP Torus frequencies initial guess
    #   Zqp0       1D np-array      Approximated Family Tangent
    # DEPENDENCIES: none
    # AUTHOR: David Lujan copied from Matlab originally by Nicola Baresi
    # CONTACT: david.lujan@colorado.edu

    # Extract Variables
    Xpo = inputs['Initialization']['Xpo']
    Tpo = inputs['Initialization']['Tpo']
    Vpo = inputs['Initialization']['Vpo']
    Epo = inputs['Initialization']['Epo']
    pars = inputs['parameters']
    
    # Problem parameters
    d     = pars['d']

    # GMOS parameters
    ds    = pars['GMOS']['ds']
    N     = pars['GMOS']['N']

    # Check Inputs
    if Epo.real == 0 or Epo.imag == 0:
        raise TypeError('Please check input eignevalues!')

    # Initial Guess 
    print('\nGenerating Initial Guess:\n')
    tht = numpy.linspace(0,2*numpy.pi,N+1)
    tht = numpy.delete(tht,-1,0)
    THT = numpy.tile(tht,[d,1])
    THT = THT.flatten('F')
    dU0 = numpy.tile(Vpo.real,N)*numpy.cos(THT) - numpy.tile(Vpo.imag,N)*numpy.sin(THT)
    U0 = numpy.tile(Xpo,N) + ds*dU0

    # Stroboscopic time
    P0     = Tpo                     
    dP0    = 0
    w10    = 2*numpy.pi/Tpo

    # Rotation number
    p0     = numpy.arctan2(Epo.imag,Epo.real)
    dp0    = 0
    w20    = p0/P0

    # QP Torus Initial Guess
    Xqp0   = U0
    Wqp0   = numpy.array([P0, p0, w10, w20])

    # Approximate family tangent
    Zqp0   = numpy.append(dU0,[dP0, dp0])/numpy.sqrt(numpy.dot(dU0,dU0)/N + dP0**2 + dp0**2)
    
    outputs = {'Xqp0': Xqp0, 'Wqp0': Wqp0, 'Zqp0': Zqp0}
    return outputs 
    
def GMOS_SingleShooting(inputs,EOM,VecField):
    '''Compute several members of a 2D quasi-periodic invariant tori family
        using the single-shooting version of the GMOS algorithm'''

    # GMOS_SingleShooting(inputs,EOM,VecField)

    # INPUT:
    # VARIABLE      TYPE               DESCRIPTION
    #  inputs        dict               input variable with fields:
    #   'Xqp0'        2D double array    Initial guess of QP Torus
    #   'Wqp0'        1D array           Initial guess of torus frequencies
    #   'Zqp0'        1D array           Initial guess of family tangent
    #   'pars'        dict               List of Parameters
    #     'd'          int                number of states
    #     'GMOS'       dict                contains values used in GMOS
    #       'ds'        double             initial step-length
    #       'dsMax'     double             maximum step-length allowed
    #       'Iter'      int                No. of Newton's method interations allowed
    #       'N'         int                No. of GMOS solution points
    #       'Nmax'      int                No. of quasi-periodic tori to be computed
    #       'M'         int                No. of nodes in time-domain
    #       'Opt'       int                optimal number of Newton's iteration
    #       'Plt'       int                Flag: 1 to enable plotting functions, 0 ow.
    #       'Tol'       double             Convergence Tolerance
    #  EOM           function           Equations of Motion
    #  VecField      function           Vector field
    #
    # OUTPUT:
    # VARIABLE      TYPE               DESCRIPTION
    #  outputs      dict                output variable with fields:
    #   'Xqp'        3D double array     QP tori computed with the algorithm      
    #   'Wqp'        2D double array     Frequencies of QP tori
    #   'Bqp'        3D double array     Floquet Matrices 
    #   'Zqp'        2D double array     Family tangent
    #   'Sqp'        1D double array     Step-lengths
    #
    # DEPENDENCIES:
    # - FourierMatrix.py
    # - RotationMatrix.py
    #
    # AUTHOR: David Lujan copied from Matlab originally by Nicola Baresi
    # CONTACT: david.lujan@colorado.edu
    #
    # REFERENCES:
    # [1] Olikara, Z. P. and Scheeres, D. J., JAS 2012 

    # Extract Variables
    Xqp0 = inputs['Initialization']['Xqp0']
    Wqp0 = inputs['Initialization']['Wqp0']
    Zqp0 = inputs['Initialization']['Zqp0']
    pars = inputs['parameters']
    
    # Parameters
    d     = pars['d']             # No. of states
    ds    = pars['GMOS']['ds']
    dsMax = pars['GMOS']['dsMax']
    Iter  = pars['GMOS']['Iter']
    N     = pars['GMOS']['N']
    Nmax  = pars['GMOS']['Nmax']
    M     = pars['GMOS']['M']
    Opt   = pars['GMOS']['Opt']
    Plt   = pars['GMOS']['Plt']
    Tol   = pars['GMOS']['Tol']
    
    D = d*N
    
    # Fourier Matrices
    FourierOutput = FourierMatrix(d,N)
    DFT = FourierOutput['R']
    IDFT = FourierOutput['invR']
    DT = FourierOutput['derivR']

    ## Initial Guess 
    print('\nLoad Initial Guess:\n')

    # Define Initial Guess
    U0 = Xqp0
    P0     = Wqp0[0]                        # Stroboscopic time 
    p0     = Wqp0[1]                        # Rotation number
    w10    = Wqp0[2]
    w20    = Wqp0[3]

    # Approximate family tangent
    dz0    = Zqp0
    dU0 = dz0[0:-2]
    dP0    = dz0[-2]
    dp0    = dz0[-1]


    ## GMOS Algorithm
    print('GMOS Algorithm (Single-shooting):\n')

    # Initialize Matrices
    Xqp    = numpy.zeros([d*N,M,Nmax])
    Wqp    = numpy.zeros([4,Nmax])
    Bqp    = numpy.zeros([d*N,d*N,Nmax])
    Zqp    = numpy.zeros([d*N+2,Nmax])
    Sqp    = numpy.zeros(Nmax)

    # Compute up to Nmax family members
    #for ii = 1:Nmax
    for ii in range(0,Nmax):
    
        # Tracking feedback
        print('Family member No.',ii+1,'ds =',ds,'\n')
        
        # Partial Derivatives for Phase Constraints
        dUT1 = (DT@U0).flatten('F')
        dUT0 = 1/w10*VecField(0,U0) - w20*dUT1
        
        # Predictor
        U  = U0 + ds*dU0
        P  = P0 + ds*dP0
        p  = p0 + ds*dp0
        
        time = numpy.linspace(0,P,M)
        STMIC = numpy.eye(D).flatten()
        
        # Corrector
        for jj in range(0,Iter):
        
            # Calculate Rotation Matrix
            R    = RotationMatrix(p,d,N)
            R    = IDFT@R@DFT
        
            # Integrate GMOS Points
            
            IC = numpy.concatenate([U,STMIC])
            sol = scipy.integrate.solve_ivp(EOM,[0,P],IC,method='DOP853',t_eval=time,
                rtol=3e-14,atol=1e-16)
            Ut = sol.y[0:D,-1]
            ft = VecField(0,Ut)
            Phi = sol.y[D:,-1].reshape((D,D),order='F')
            Xt = sol.y[0:D,:]

            # Additional Constraints
            phs0 = numpy.dot(U - U0, dUT0)/N
            phs1 = numpy.dot(U, dUT1)/N
            prd  = P - P0
            arcl = numpy.dot(U.flatten('F') - U0.flatten('F'), dU0.flatten('F'))/N \
                + (P - P0)*dP0 + (p - p0)*dp0 - ds
                
            # Rotate Points
            Ur = R@Ut
            
            # Plot
            #if Plt:
            #    plt.figure(99)
            #    plt.clf()
            #    ax = plt.axes(projection='3d')
            #    ax.plot3D(U[0,:],U[1,:],U[2,:],'ob')
            #    ax.plot3D(Ut[0,:],Ut[1,:],Ut[2,:],'or')
            #    ax.plot3D(Ur[0,:],Ur[1,:],Ur[2,:],'og')
            #    ax.plot3D(U[0,0],U[1,0],U[2,0],'ob',mfc='b')
            #    ax.plot3D(Ut[0,0],Ut[1,0],Ut[2,0],'or',mfc='r')
            #    ax.plot3D(Ur[0,0],Ur[1,0],Ur[2,0],'og',mfc='g')
            #    ax.axis('equal')
            #    plt.show()
        
                
            ## Error Vector 
            F = numpy.concatenate((Ur-U,numpy.array([phs0,phs1,prd,arcl])))

            ## Error Jacobian 
            DF = numpy.zeros([D+4,D+2])           
            DF[0:-4,0:-2] = R@Phi - numpy.eye(D)    # d arcl/dp
            DF[0:-4,-2] = R@ft
            DF[0:-4,-1] = -DT@Ur       # d R(U)/dp
          
            DF[-4,0:-2] = dUT0/N           # d phs0/du
            DF[-3,0:-2] = dUT1/N           # d phs1/du
        
            DF[-2,-2] = 1                               # d prd/dP
            DF[-1,0:-2] = dU0/N            # d arcl/du
            DF[-1,-2] = dP0                             # d arcl/dP
            DF[-1,-1] = dp0                            # d arcl/dp
        
                
            ## Newton's Update
            z = -numpy.linalg.pinv(DF)@F
            test1 = numpy.sqrt(numpy.dot(F[0:-4],F[0:-4])/N + numpy.dot(F[-4:],F[-4:]))
            test2 = numpy.sqrt(numpy.dot(z[0:-2],z[0:-2])/N + numpy.dot(z[-2:],z[-2:]))
            print('|F|* =',test1,'|z|* =',test2,'|F| =',numpy.linalg.norm(F),'|z| =',
                numpy.linalg.norm(z),'arc =',arcl,'\n')
        
            if ((test1 < Tol) or (test2 < Tol)):
                print('Quasi-periodic Torus has been found!\n\n')
            
                # Plot
                #if Plt:
                #    plt.figure(98)
                #    plt.clf()
                #    clr  = numpy.linspace(0,1,Nmax)
                #    xt = numpy.reshape(Xt[0:d*N],(N,d)).T
                #    if d == 4:
                #        plt.plot(xt[0,:],xt[1,:],'.','Color',[0,clr[ii],1-clr[ii]])
                #    else:
                #        plt.axes(prejection='3d').plot3D(xt[0,:],xt[1,:],xt[2,:],'.','Color',[0,clr[ii],1-clr[ii]])
                #    plt.axis('equal')
                #    plt.show()
            
                # Stability
                B   = R@Phi
                        
                # Compute Family Tangent
                dz0 = numpy.concatenate((U-U0,numpy.array([P-P0,p-p0])))/numpy.sqrt(numpy.dot(U-U0,U-U0)/N + (P-P0)**2 + (p-p0)**2)
                dP0 = dz0[-2]
                dp0 = dz0[-1]

                # Step-size Controller
                Eps = Opt/(jj+1)
                if Eps > 2:
                    Eps = 2
                elif Eps < 0.5:
                    Eps = 0.5

                ds = numpy.amin([Eps*ds,dsMax])
                        
                # Store Results
                Xqp[:,:,ii] = Xt
                Wqp[:,ii] = numpy.array([P,p,2*numpy.pi/P,p/P])
                Bqp[:,:,ii] = B
                Zqp[:,ii] = dz0
                Sqp[ii] = ds
                        
                # Update Old Solution
                U0 = U
                P0 = P
                p0 = p
                break
            else:
                # Update Points & Frequency Vector
                U += z[0:-2]
                P += z[-2]
                p += z[-1]
    
        if jj == (Iter-1):
            print('Quasi-periodic Torus could not be found!\n\n')

    outputs = {'Xqp': Xqp, 'Wqp': Wqp, 'Bqp': Bqp, 'Zqp': Zqp, 'Sqp': Sqp}
    return outputs
    
def GMOS_MultipleShooting_InitialGuess(inputs,EOM):
    '''Generate Initial Guess for the GMOS multiple-shooting Method'''
    
    # GMOS_MultipleShooting_InitialGuess(inputs,EOM)
 
    # INPUTS:
    # VARIABLE          TYPE            DESCRIPTION
    #   inputs          dict            input variable with fields:
    #   'Initialization'
    #    'Xpo'          1D np-array     Periodic Orbit Initial Conditions
    #    'Tpo'          float           Periodic Orbit Period
    #    'Vpo'          1D np-array     Center subspace Eigenvector
    #    'Epo'          float           Center subspace Eigenvalue
    #   'parameters'    dict            List of Parameters
    #    'd'            int             No. of states
    #    'GMOS'
    #     'ds'          float           Initial displacement along center subspace
    #     'n'           int             No. of Multiple-shooting segments
    #     'N'           int             No. of GMOS solution points
    #   EOM             function        Equations of Motion
    #
    # OUTPUTS:
    # VARIABLE          TYPE            DESCRIPTION
    # outputs           dict            output variable with fields:
    #  'Xqp0'           2D np-array     QP Torus initial guess
    #  'Wqp0'           1D np-array     QP Torus frequencies initial guess
    #  'Zqp0'           1D np-array     Approximated Family Tangent
    #
    # DEPENDENCIES: none
    # AUTHOR: David Lujan copied from Matlab originally by Nicola Baresi
    # CONTACT: david.lujan@colorado.edu

    # Extract Variables
    Xpo = inputs['Initialization']['Xpo']
    Tpo = inputs['Initialization']['Tpo']
    Vpo = inputs['Initialization']['Vpo']
    Epo = inputs['Initialization']['Epo']
    pars = inputs['parameters']
    
    # Problem parameters
    d     = pars['d']

    # GMOS parameters
    ds = pars['GMOS']['ds']
    n = pars['GMOS']['n']
    N = pars['GMOS']['N']
    D = d*N

    # Check Inputs
    if Epo.real == 0 or Epo.imag == 0:
        raise TypeError('Please check input eignevalues!')

    # Reintegrate Periodic Orbit
    time = numpy.linspace(0,Tpo,n+1)
    IC = numpy.concatenate([Xpo, numpy.eye(d).flatten()])
    sol = scipy.integrate.solve_ivp(EOM,[0,Tpo],IC,method='DOP853',t_eval=time,
        rtol=3e-14,atol=1e-16)

    # Construct Torus Skeleton
    tht = numpy.linspace(0,2*numpy.pi,N+1)
    tht = numpy.delete(tht,-1,0)
    THT = numpy.tile(tht,[d,1])
    THT = numpy.tile(THT.flatten('F'),n)
    
    Xt = numpy.tile(sol.y[0:d,0:-1].T,(1,N)).flatten()
    Phi = numpy.zeros((d*n,d))
    for ii in range(0,n):
        Phi[d*ii:d*(ii+1),:] = sol.y[d:,ii].reshape((d,d),order='F')

    Vt = (Phi@Vpo).reshape((d,n),order='F')
    Vt /= numpy.linalg.norm(Vt,axis=0)
    Vt = numpy.tile(Vt,(N,1)).flatten('F')   
    dU0 = Vt.real*numpy.cos(THT) - Vt.imag*numpy.sin(THT)
    U0 = Xt + ds*dU0

    # Stroboscopic time
    P0 = Tpo
    dP0 = 0
    w10 = 2*numpy.pi/Tpo

    # Rotation number
    p0 = numpy.arctan2(Epo.imag,Epo.real)     
    dp0 = 0
    w20 = p0/P0

    # Output
    # QP Torus Initial Guess
    Wqp0 = numpy.array([P0, p0, w10, w20])

    # Approximate family tangent
    Zqp0 = numpy.append(dU0,[dP0,dp0])/numpy.sqrt(numpy.dot(dU0,dU0)/D + dP0**2 + dp0**2)

    outputs = {'Xqp0': U0, 'Wqp0': Wqp0, 'Zqp0': Zqp0}
    return outputs
    
def GMOS_MultipleShooting(inputs,EOM,VecField):
    '''Compute several members of a 2D quasi-periodic invariant tori family
        using the multiple-shooting version of the GMOS algorithm'''
        
    # GMOS_MultipleShooting(inputs,EOM,VecField)

    # INPUT:
    # VARIABLE          TYPE           DESCRIPTION
    #  inputs           dict           input variable with fields:
    #   'Xqp0'          2D np-array    Initial guess of QP Torus
    #   'Wqp0'          1D np-array    Initial guess of torus frequencies
    #   'Zqp0'          1D np-array    Initial guess of family tangent
    #   'pars'          dict           List of Parameters
    #     'd'           int            number of states
    #     'GMOS'        dict           contains values used in GMOS
    #      'ds'         double         initial step-length
    #      'dsMax'      double         maximum step-length allowed
    #      'Iter'       int            No. of Newton's method interations allowed
    #      'n'          int            No. of multiple-shooting segments
    #      'N'          int            No. of GMOS solution points
    #      'Nmax'       int            No. of quasi-periodic tori to be computed
    #      'M'          int            No. of node points in time domain
    #      'Opt'        int            optimal number of Newton's iteration
    #      'Plt'        int            Flag: 1 to enable plotting functions, 0 ow.
    #      'Tol'        double         Convergence Tolerance
    #  EOM              function       Equations of Motion
    #  VecField         function       Vector field
    #
    # OUTPUT:
    # VARIABLE      TYPE               DESCRIPTION
    #  outputs      dict               output variable with fields:
    #   'Xqp'       3D np-array        QP tori computed with the algorithm      
    #   'Wqp'       2D np-array        Frequencies of QP tori
    #   'Bqp'       3D np-array        Floquet Matrices 
    #   'Zqp'       2D np-array        Family tangent
    #   'Sqp'       1D np-array        Step-lengths
    #
    # DEPENDENCIES: FourierMatrix.py, RotationMatrix.py
    # AUTHOR: David Lujan copied from Matlab originally by Nicola Baresi
    # CONTACT: david.lujan@colorado.edu
    #
    # REFERENCES:
    # [1] Olikara, Z. P. and Scheeres, D. J., JAS 2012 


    # Extract Variables
    Xqp0 = inputs['Initialization']['Xqp0']
    Wqp0 = inputs['Initialization']['Wqp0']
    Zqp0 = inputs['Initialization']['Zqp0']
    pars = inputs['parameters']

    # Parameters
    d     = pars['d']
    ds    = pars['GMOS']['ds']     # step-length
    dsMax = pars['GMOS']['dsMax']  # Max step-length
    Iter  = pars['GMOS']['Iter']   # Max no. of iterations allowed
    n     = pars['GMOS']['n']      # No. of Segments
    N     = pars['GMOS']['N']      # No. of GMOS Points in one stroboscopic map
    Nmax  = pars['GMOS']['Nmax']   # No. of family members to be computed
    M     = pars['GMOS']['M']      # No. of time nodes per segment
    Opt   = pars['GMOS']['Opt']    # No. of Optimal iterations
    Plt   = pars['GMOS']['Plt']    # Plot flag
    Tol   = pars['GMOS']['Tol']    # Tolerance

    D     = d*N # number of states in one stroboscopic map

    # Fourier Coefficients 
    FourierOutput = FourierMatrix(d,N)
    DFT = FourierOutput['R']
    IDFT = FourierOutput['invR']
    DT = FourierOutput['derivR']

    ## Initial Guess 
    print('\nLoad Initial Guess:\n')

    # Define Initial Guess
    U0     = Xqp0
    P0     = Wqp0[0]                        # Stroboscopic time 
    p0     = Wqp0[1]                        # Rotation number
    w10    = Wqp0[2]
    w20    = Wqp0[3]

    # Approximate family tangent
    dz0    = Zqp0
    dU0    = dz0[0:-2]
    dP0    = dz0[-2]
    dp0    = dz0[-1]

    ## Time Vector
    t     = numpy.linspace(0,P0,n+1) 
    time = numpy.linspace(t[0],t[1],M+1)
    
    ## GMOS Algorithm
    print('GMOS Algorithm (Multiple-Shooting):\n')

    # Initialize Matrices
    # Initialization
    Ut = numpy.zeros([D*n])
    Mt = numpy.zeros([D*n,D*n])
    ft = numpy.zeros([D*n])
    Xt = numpy.zeros([D*n,M])
    Xqp    = numpy.zeros([D*n,M,Nmax])
    Wqp    = numpy.zeros([4,Nmax])
    Bqp    = numpy.zeros([D,D,Nmax])
    Zqp    = numpy.zeros([D*n+2,Nmax])
    Sqp    = numpy.zeros(Nmax)
    # Error Jacobian
    DF = numpy.zeros((D*n+4,D*n+2))
    DF[range(0,D*n),range(0,D*n)] = -1
    DF[-2,-2] = 1
    STMIC = numpy.eye(int(D*n)).flatten()

    # Compute up to Nmax family members
    for ii in range(0,Nmax):
    
        # Tracking index
        print('Family member No.',ii+1,'ds =',ds,':\n')
        
        # Partial Derivatives for phase constraints
        dUT1 = (DT@U0.reshape((D,n),order='F')).flatten('F')
        dUT0 = 1/w10*VecField(0,U0) - w20*dUT1
        
        # Predictor
        U  = U0 + ds*dU0
        P  = P0 + ds*dP0
        p  = p0 + ds*dp0

        # Corrector
        for jj in range(0,Iter):
        
            # Rotation Matrix
            R = RotationMatrix(p,d,N)
            R = IDFT@R@DFT
        
            # Integrate Trajectories
            #parfor kk = 1:n
            for kk in range(0,n):
                IC    = numpy.concatenate([U[D*kk:D*(kk+1)], numpy.eye(D).flatten()])
                sol = scipy.integrate.solve_ivp(EOM,[t[0],t[1]],IC,method='DOP853',t_eval=time,
                    rtol=3e-14,atol=1e-16)
                x = sol.y
                # Store Results
                Ut[D*kk:D*(kk+1)] = x[0:D,-1]
                Mt[D*kk:D*(kk+1),D*kk:D*(kk+1)] = x[D:,-1].reshape((D,D),order='F')
                ft[D*kk:D*(kk+1)] = VecField(0,x[0:D,-1])
                Xt[D*kk:D*(kk+1),:] = x[0:D,0:-1]
            
            Ur = R@Ut[D*(n-1):D*n]
            
            #if jj != 0 and ii != 0:
            #    IC[0:D*n] = U
            #else:
            #    IC = numpy.concatenate([U,STMIC])
            #    
            #sol = scipy.integrate.solve_ivp(EOM,[t[0],t[1]],IC,method='DOP853',
            #    t_eval=time,rtol=3e-14,atol=1e-16)
            #Ut = sol.y[0:D*n,-1]
            #ft = VecField(0,Ut)
            #Mt = sol.y[D*n:,-1].reshape((D*n,D*n),order='F')
            #Xt = sol.y[0:D*n,0:-1]
            #Ur = R@sol.y[D*(n-1):D*n,-1]
        
            # Constraints
            phs0 = numpy.dot(U - U0,dUT0)/(n*N)
            phs1 = numpy.dot(U,dUT1)/(n*N)
            prd  = P - P0
            arc  = numpy.dot(U - U0,dU0)/(n*N) + (P - P0)*dP0 + (p - p0)*dp0 - ds

            # Plot -- fix this later
            #if(Plt):
            #    u0 = numpy.reshape(U[:,0],(N,d)).T
            #    ut = numpy.reshape(Ut[:,-1],(N,d)).T
            #    ur = numpy.reshape(Ur,(N,d)).T
            #    plt.figure(99)
            #    plt.clf()
            #    ax = plt.axes(projection='3d')
            #    ax.plot3D(u0[0,:],u0[1,:],u0[2,:],'ob')
            #    ax.plot3D(ut[0,:],ut[1,:],ut[2,:],'or')
            #    ax.plot3D(ur[0,:],ur[1,:],ur[2,:],'og')
            #    ax.plot3D(u0[0,0],u0[1,0],u0[2,0],'ob',mfc='b')
            #    ax.plot3D(ut[0,0],ut[1,0],ut[2,0],'or',mfc='r')
            #    ax.plot3D(ur[0,0],ur[1,0],ur[2,0],'og',mfc='g')
            #    ax.axis('equal')
            #    plt.show()
                
            # Error Vector
            F = numpy.concatenate((Ur-U[0:D], Ut[0:D*(n-1)]-U[D:], numpy.array([phs0, phs1, prd, arc])))
            
            # Error Jacobian
            DF[0:D,-D-2:-2] = R@Mt[(n-1)*D:,(n-1)*D:]
            DF[0:D,-2] = R@ft[D*(n-1):]/n
            DF[0:D,-1] = -DT@Ur
            for kk in range(1,n):
                DF[D*kk:D*(kk+1),D*(kk-1):D*kk] = Mt[D*(kk-1):D*kk,D*(kk-1):D*kk]
                
            DF[D:D*n,-2] = ft[0:D*(n-1)]/n
            DF[-4,0:D*n] = dUT0/(n*N)
            DF[-3,0:D*n] = dUT1/(n*N)
            DF[-1,0:D*n] = dU0/(n*N)
            DF[-1,-2] = dP0
            DF[-1,-1] = dp0

            # Newton's Update
            z = -1*scipy.linalg.pinv(DF)@F
            test1 = numpy.sqrt(numpy.dot(F[0:-4],F[0:-4])/N + numpy.dot(F[-4:],F[-4:]))
            test2 = numpy.sqrt(numpy.dot(z[0:-2],z[0:-2])/N + numpy.dot(z[-2:],z[-2:]))
            normF = numpy.linalg.norm(F)
            normz = numpy.linalg.norm(z) #look at DF because F should be correct
            print('|F| =',normF,'|z| =',normz,'arc =',arc,'\n')
        
            if(test1 < Tol or test2 < Tol):
                print('Quasi-periodic Torus has been found!\n\n')
            
                # Plot
                #if(Plt):
                #    plt.figure(98)
                #    plt.clf()
                #    clr  = linspace(0,1,pars['GMOS']['Nmax'])
                #    for kk in range(0,n*(M+1)):
                #        ut = numpy.reshape(Xt[:,kk],(d,N))
                #        if(d == 4):
                #            plt.plot(ut[0,:],ut[1,:],'.','Color',[0,clr[ii],1-clr[ii]])
                #        else:
                #            plt.axes(projection='3d').plot3D(ut[0,:],ut[1,:],ut[2,:],'.','Color',[0,clr[ii],1-clr[ii]])
                #
                #    plt.axis('equal')
                #    plt.show()
            
                # Stability
                Phi = numpy.eye(D)
                for kk in range(0,n):
                    Phi = Mt[D*kk:D*(kk+1),D*kk:D*(kk+1)]@Phi

                B = R@Phi
            
                # Compute Family Tangent
                dz0 = numpy.concatenate((U-U0, numpy.array([P-P0,p-p0])))/numpy.sqrt(numpy.dot(U-U0,U-U0)/D + (P-P0)**2 + (p-p0)**2)
                dU0 = dz0[0:-2]
                dP0 = dz0[-2]
                dp0 = dz0[-1]
            
                # Step-size Controller
                Eps = Opt/(jj+1)
                if(Eps > 2):
                    Eps = 2.0
                elif(Eps < 0.5):
                    Eps = 0.5
                
                ds = numpy.amin([Eps*ds,dsMax])
            
                # Store Results
                Xqp[:,:,ii] = Xt
                Wqp[:,ii]   = numpy.array([P, p, 2*numpy.pi/P, p/P])
                Bqp[:,:,ii] = B
                Zqp[:,ii]   = dz0
                Sqp[ii]     = ds
                        
                # Update Previous Solution
                U0   = U
                P0   = P
                p0   = p         
                break
            else:
                # Update solution
                U += z[0:-2]
                P += z[-2]
                p += z[-1]
    
        if(jj == Iter-1):
            print('Quasi-periodic torus could not be found!\n')

    outputs = {'Xqp': Xqp, 'Wqp': Wqp, 'Bqp': Bqp, 'Zqp': Zqp, 'Sqp': Sqp}
    return outputs
    
def GMOS_Collocation_InitialGuess(inputs,EOM):
    '''Generate Initial Guess for the GMOS Collocation Method'''
    
    # GMOS_Collocation_InitialGuess(inputs,Epo)
 
    # INPUTS:
    # VARIABLE          TYPE            DESCRIPTION
    #   inputs          dict            input variable with fields:
    #   'Initialization'
    #    'Xpo'          1D np-array     Periodic Orbit Initial Conditions
    #    'Tpo'          float           Periodic Orbit Period
    #    'Vpo'          1D np-array     Center subspace Eigenvector
    #    'Epo'          float           Center subspace Eigenvalue
    #   'parameters'    dict            List of Parameters
    #    'd'            int             No. of states
    #    'GMOS'
    #     'ds'          float           Initial displacement along center subspace
    #     'N'           int             No. of GMOS solution points
    #    'Collocation'
    #     'n'           int             No. of collocation segments
    #     'm'           int             Degree of Lagrange polynomials
    #   EOM             function        Equations of Motion
    #
    # OUTPUTS:
    # VARIABLE          TYPE            DESCRIPTION
    # outputs           dict            output variable with fields:
    #  'Xqp0'           2D np-array     QP Torus initial guess
    #  'Wqp0'           1D np-array     QP Torus frequencies initial guess
    #  'Zqp0'           1D np-array     Approximated Family Tangent
    #
    # DEPENDENCIES: none
    # AUTHOR: David Lujan copied from Matlab originally by Nicola Baresi
    # CONTACT: david.lujan@colorado.edu

    # Extract Variables
    Xpo = inputs['Initialization']['Xpo']
    Tpo = inputs['Initialization']['Tpo']
    Vpo = inputs['Initialization']['Vpo']
    Epo = inputs['Initialization']['Epo']
    pars = inputs['parameters']
    
    # Problem parameters
    d     = pars['d']

    # GMOS Collocation parameters
    n     = pars['GMOS']['Collocation']['n']
    m     = pars['GMOS']['Collocation']['m']
    Ns    = n*(m+1)+1

    # GMOS parameters
    ds    = pars['GMOS']['ds']
    N     = pars['GMOS']['N']
    Npts  = Ns*N

    # Check Inputs
    if Epo.real == 0 or Epo.imag == 0:
        raise TypeError('Please check input eignevalues!')

    ## Initialization ##
    # Create Time Points
    t = numpy.linspace(0,1,n+1)
    t = numpy.delete(t,-1,0)

    # Legendre Polynomial Roots
    PolyCoef = scipy.special.legendre(m)
    PolyRoots = numpy.roots(PolyCoef)
    tm = (numpy.sort(PolyRoots)+1)/(2*n)
    tm = numpy.insert(tm,0,0)

    # Create Time Vector
    t    = t[numpy.newaxis] + tm[numpy.newaxis].T
    t    = numpy.append(t.flatten('F'), 1)

    ## Initial Guess 
    # Integrate Initial Guess
    time   = Tpo*t
    IC     = numpy.append(Xpo, numpy.eye(d).flatten())
    sol = scipy.integrate.solve_ivp(EOM,[0,time[-1]],IC,method='DOP853',t_eval=time,
        rtol=3e-14,atol=1e-16)
    X0 = sol.y

    # Construct Torus skeleton
    tht    = numpy.linspace(0,2*numpy.pi,N+1)
    tht = numpy.delete(tht,-1)
    THT    = numpy.tile(tht,(d,1)) 
    THT    = THT.flatten('F')
    U0     = numpy.zeros([d*N,Ns])
    dU0    = numpy.zeros([d*N,Ns])

    for ii in range(0,Ns):
        Xt        = X0[0:d,ii]
        Mt        = X0[d:,ii].reshape((d,d),order='F') 
        Vt        = Mt@Vpo/numpy.linalg.norm(Mt@Vpo)
        Vt = numpy.tile(Vt,(N,1)).flatten()
        dU0[:,ii] = Vt.real*numpy.cos(THT) - Vt.imag*numpy.sin(THT)
        U0[:,ii]  = numpy.tile(Xt,N) + ds*dU0[:,ii] 

    # U0     = numpy.tile(X0[0:d,:],N)   # for cases where monodromy matrix is not available...

    P0     = Tpo                        # Stroboscopic time 
    dP0    = 0
    w10    = 2*numpy.pi/Tpo
    dw10   = 0

    p0     = numpy.arctan2(Epo.imag,Epo.real)     # Rotation number
    dp0    = 0
    w20    = p0/P0              
    dw20   = 0

    dl10   = 0
    dl20   = 0

    # QP Torus Initial Guess
    Xqp0   = U0
    Wqp0   = numpy.array([P0, p0, w10, w20])

    # Approximate family tangent
    Zqp0   = numpy.append(dU0.flatten('F'), [dP0, dp0, dw10, dw20, dl10, dl20])/numpy.sqrt(numpy.dot(dU0.flatten(),dU0.flatten())/Npts + dP0**2 + dp0**2) 

    outputs = {'Xqp0': Xqp0, 'Wqp0': Wqp0, 'Zqp0': Zqp0}
    return outputs
    
def GMOS_Collocation(inputs,VecField,JacMat):
    '''Compute several members of a 2D quasi-periodic invariant tori family
        using the collocation version of the GMOS algorithm'''
        
    # GMOS_Collocation(inputs,VecField,JacMat)

    # INPUT:
    # VARIABLE          TYPE           DESCRIPTION
    #  inputs           dict           input variable with fields:
    #   'Xqp0'          2D np-array    Initial guess of QP Torus
    #   'Wqp0'          1D np-array    Initial guess of torus frequencies
    #   'Zqp0'          1D np-array    Initial guess of family tangent
    #   'J'             2D np-array    Canonical Transformation [1]
    #   'pars'          dict           List of Parameters   
    #     'd'           int            number of states
    #     'GMOS'        dict           contains values used in GMOS
    #      'ds'         float          initial step-length
    #      'dsMax'      float          maximum step-length allowed
    #      'Iter'       int            No. of Newton's method interations allowed
    #      'N'          int            No. of GMOS solution points
    #      'Nmax'       int            No. of quasi-periodic tori to be computed
    #      'Opt'        int            optimal number of Newton's iteration
    #      'Plt'        int            Flag: 1 to enable plotting functions, 0 ow.
    #      'Tol'        float          Convergence Tolerance
    #      'Collocation'
    #       'n'         int            No. of Collocation segments
    #       'm'         int            degree of Legendre Polynomials
    #   VecField        function       Vector field
    #   JacMat          function       Partial derivatives of Vector field
    #
    # OUTPUT:
    # VARIABLE      TYPE               DESCRIPTION
    #  outputs      dict               output variable with fields:
    #   'Xqp'       3D np-array        QP tori computed with the algorithm      
    #   'Wqp'       2D np-array        Frequencies of QP tori
    #   'Bqp'       3D np-array        Floquet Matrices 
    #   'Zqp'       2D np-array        Family tangent
    #   'Sqp'       1D np-array        Step-lengths
    #
    # DEPENDENCIES: LagrangeMatrix.py, LagrangePolynomials.py, FourierMatrix.py, RotattionMatirx.py
    #
    # AUTHOR: David Lujan copied from Matlab originally by Nicola Baresi
    # CONTACT: david.lujan@colorado.edu
    #
    # REFERENCES:
    # [1] Olikara, Z. P., "Computation of Quasi-periodic Tori and Heteroclinic
    # Connections in Astrodynamics using Collocation Techniques", PhD Thesis,
    # University of Colorado Boulder, 2016, Chapter 2, pp. 24-31 

    # Extract Variables
    Xqp0 = inputs['Initialization']['Xqp0']
    Wqp0 = inputs['Initialization']['Wqp0']
    Zqp0 = inputs['Initialization']['Zqp0']
    J = inputs['Initialization']['J']
    pars = inputs['parameters']

    # Problem parameters
    d     = pars['d']

    # GMOS Collocation parameters
    n     = pars['GMOS']['Collocation']['n']
    m     = pars['GMOS']['Collocation']['m']
    Ns    = n*(m+1)+1

    # GMOS parameters
    ds    = pars['GMOS']['ds']     # step-length
    dsMax = pars['GMOS']['dsMax']  # Max step-length
    Iter  = pars['GMOS']['Iter']   # Max no. of iterations allowed
    N     = pars['GMOS']['N']      # No. of GMOS Points
    Nmax  = pars['GMOS']['Nmax']   # No. of family members to be computed
    Opt   = pars['GMOS']['Opt']    # No. of Optimal iterations
    Plt   = pars['GMOS']['Plt']    # Plot flag
    Tol   = pars['GMOS']['Tol']    # Tolerance

    D     = d*N
    Npts  = Ns*N
    NTot = D*Ns+6

    ## Initialization ###
    print('Initialization:\n')

    # Create Time Points
    t = numpy.linspace(0,1,n+1)
    t = numpy.delete(t,-1)

    # Legendre Polynomial Roots
    print('Compute Legendre Polynomial Roots...')
    PolyCoef = scipy.special.legendre(m)
    PolyRoots = numpy.roots(PolyCoef)
    tm = (numpy.sort(PolyRoots)+1)/(2*n)
    tm = numpy.insert(tm,0,0) 
    print('Done!\n')

    # Calculate Lagrange Polynomials
    print('Compute Lagrange Interpolating Polynomials...')
    L,DL = LagrangeMatrix(t[1],tm)
    DL    = numpy.zeros([m,m+1])
    for jj in range(1,m+1):
        temp,dL = LagrangeMatrix(tm[jj],tm)
        DL[jj-1,:] = dL
    
    L1_c  = numpy.tile(L,(D,1)).flatten('F')

    L2 = scipy.sparse.lil_matrix((m*D,(m+1)*D))
    for jj in range(0,m):
        rind = numpy.tile(numpy.arange(D*jj,D*(jj+1)),m+1)
        cind = numpy.arange(0,D*(m+1))
        L2[rind,cind] = numpy.tile(DL[jj,:],(D,1)).flatten('F')

    print('Done!\n')

    # Create Time Vector
    print('Create time vector...')
    t    = t[numpy.newaxis] + tm[numpy.newaxis].T
    t    = numpy.append(t.flatten('F'), 1)
    print('Done!\n')

    # Canonical Transformation
    print('Canonical Transformations...')

    Jd  = scipy.sparse.lil_matrix((d*N,d*N))
    for ii in range(0,N):
        Jd[d*ii:d*(ii+1),d*ii:d*(ii+1)] = J
        
    print('Done!\n')

    # Fourier Coefficients 
    print('Fourier Matrices...')
    FourierOutput = FourierMatrix(d,N)
    DFT = FourierOutput['R']
    IDFT = FourierOutput['invR']
    DT = FourierOutput['derivR']
    
    DFT  = scipy.sparse.lil_matrix(DFT)
    IDFT = scipy.sparse.lil_matrix(IDFT)
    DT   = scipy.sparse.lil_matrix(DT)
    
    print('Done!\n')

    ## Initial Guess 
    print('\nLoad Initial Guess:\n')

    # Define Initial Guess
    U0     = Xqp0
    P0     = Wqp0[0]                        # Stroboscopic time 
    p0     = Wqp0[1]                        # Rotation number
    w10    = Wqp0[2]
    w20    = Wqp0[3]

    # Approximate family tangent
    dz0    = Zqp0
    dU0    = dz0[0:-6].reshape((D,Ns),order='F')
    dP0    = dz0[-6]
    dp0    = dz0[-5]
    dw10   = dz0[-4]
    dw20   = dz0[-3]
    dl10   = dz0[-2]
    dl20   = dz0[-1]

    ## GMOS Algorithm
    print('GMOS Algorithm:\n')

    # Initialize Output Matrices
    Xqp    = numpy.zeros([D,Ns,Nmax])
    Wqp    = numpy.zeros([4,Nmax])
    Bqp    = numpy.zeros([D,D,Nmax])
    Zqp    = numpy.zeros([D*Ns+6,Nmax])
    Sqp    = numpy.zeros(Nmax)
    
    # Initialize internal Arrays
    dUT0 = numpy.zeros([D,n*(m+1)+1])
    dUt0 = numpy.zeros([D,Ns])
    F_c = numpy.zeros(NTot)
    DF_c = scipy.sparse.lil_matrix((NTot,NTot))
    DF_c[range(0,D),range(0,D)] = -numpy.ones(D)
    ## Parametrization Equations
    # Fixing longitudinal frequency
    DF_c[-4,-6] = 1
    # w2
    DF_c[-1,-5] = -1

    # Compute up to Nmax family members
    for ii in range(0,Nmax):
    
        # Tracking feedback
        print('Family member No.',ii+1,'ds =',ds,'\n')
     
        # Phase Constraints
        dUT1 = DT@U0
        #parfor kk = 1:Ns
        for kk in range(0,Ns):
            dUT0[:,kk] = 1/(2*numpy.pi)*(P0*VecField(P0*t[kk],U0[:,kk]) - p0*dUT1[:,kk])
        
        # Predictor
        U  = U0 + ds*dU0
        P  = P0 + ds*dP0
        p  = p0 + ds*dp0
        w1 = w10 + ds*dw10
        w2 = w20 + ds*dw20
        l1 = 0
        l2 = 0
    
        Ufam0 = U0
        dUfam0 = dU0
        Pfam0 = P0
        dPfam0 = dP0
        pfam0 = p0
        dpfam0 = dp0
        w1fam0 = w10
        dw1fam0 = dw10
        w2fam0 = w20
        dw2fam0 = dw20

        for trial in range(0,5):
            if trial > 0:
                U  = Ufam0 + ds*dUfam0
                P  = Pfam0 + ds*dPfam0
                p  = pfam0 + ds*dpfam0
                w1 = w1fam0 + ds*dw1fam0
                w2 = w2fam0 + ds*dw2fam0
                l1 = 0
                l2 = 0

            # Corrector
            for jj in range(1,Iter+1):
        
                # Rotation Matrix
                R = RotationMatrix(p,d,N)
                R = scipy.sparse.lil_matrix(IDFT@R@DFT)
                
                # Partial Derivatives for unfolding the parameters
                dUt1 = DT@U
                #parfor kk = 1:Ns
                for kk in range(0,Ns):
                    dUt0[:,kk] = 1/(2*numpy.pi)*(P*VecField(P*t[kk],U[:,kk]) - p*dUt1[:,kk])
                    
                # Plot
                #if Plt:
                #    plt.figure(99)
                #    plt.clf()
                #    ax = plt.axes(projection='3d')
                #    u0 = numpy.reshape(U[:,0],(N,d)).T
                #    ut = numpy.reshape(U[:,-1],(N,d)).T
                #    ur = numpy.reshape(R@U[:,-1],(N,d)).T
                #    ax.plot3(u0[0,:],u0[1,:],u0[2,:],'ob')
                #    ax.plot3(ut[0,:],ut[1,:],ut[2,:],'or')
                #    ax.plot3(ur[0,:],ur[1,:],ur[2,:],'og')
                #    ax.plot3(u0[0,0],u0[1,0],u0[2,0],'ob',mfc='b')
                #    ax.plot3(ut[0,0],ut[1,0],ut[2,0],'or',mfc='r')
                #    ax.plot3(ur[0,0],ur[1,0],ur[2,0],'og',mfc='g')
                #    ax.axis('equal')
                #    plt.show()

                
                ### Error Vector & Error Jacobian ###
                # Quasi-periodic Condition       
                F_c[0:D] = R@U[:,-1] - U[:,0]
                rind,cind = numpy.nonzero(R)
                DF_c[rind,cind+NTot-D-6] = R[rind,cind]
                DF_c[0:D,-5] = (-DT@R@U[:,-1])[numpy.newaxis].T
                 
                #parfor kk = 2:Ns
                for kk in range(1,Ns):
            
                    # Indices
                    idx = numpy.arange(D*kk,D*(kk+1))
                    Q = kk%(m+1)
            
                    if Q:
                        ## Collocation Conditions
                        # Error vector
                        F_c[idx] = P*VecField(P*t[kk],U[:,kk]) - numpy.sum(numpy.tile(DL[Q-1,:],(D,1))*U[:,kk-Q:kk-Q+m+1],1)
                        
                        # Temporary matrix used to pass values to cell array
                        tmp = numpy.zeros([D,D*(m+1)])
                        tmp[:,D*Q:D*(Q+1)] = P*JacMat(P*t[kk],U[:,kk])
                        tmp -= L2[D*(Q-1):D*Q,:]
                        rind, cind = numpy.nonzero(tmp)
                        DF_c[D*kk+rind,D*(kk-Q)+cind] = tmp[rind,cind]
                        DF_c[idx,-6] = VecField(P*t[kk],U[:,kk])
                        DF_c[idx,-2] = Jd@dUt0[:,kk]
                        DF_c[idx,-1] = Jd@dUt1[:,kk]
                               
                    else:
                        ## Continuity Conditions
                        # Error vector
                        F_c[idx] = U[:,kk] - numpy.sum(numpy.tile(L,(D,1))*U[:,kk-m-1:kk],1)
                        # Error jacobian
                        idxL = numpy.arange(D*(kk-L.size),D*kk)
                        rind = numpy.tile(idx,m+2)
                        cind = numpy.append(idxL,idx)
                        DF_c[rind,cind] = numpy.append(-L1_c,numpy.ones(D))
                        
                
                ## Phase Conditions
                # dU/dtht0
                F_c[-6] = numpy.dot(U.flatten('F')-U0.flatten('F'),dUT0.flatten('F'))/Npts
                DF_c[-6,0:D*Ns] = dUT0.flatten('F')/Npts
        
                # dU/dtht1
                F_c[-5] = numpy.dot(U.flatten('F'),dUT1.flatten('F'))/Npts
                DF_c[-5,0:D*Ns] = dUT1.flatten('F')/Npts
               
                ## Parametrization Equations
                # Fixing longitudinal frequency
                F_c[-4] = P-P0
        
                # Pseudo-arclength continuation...
                psdarc = numpy.dot(U.flatten('F') - U0.flatten('F'), dU0.flatten('F'))/Npts + (P - P0)*dP0 + (p - p0)*dp0 - ds
                F_c[-3] = psdarc
                DF_c[-3,0:D*Ns] = dU0.flatten('F')/Npts
                DF_c[-3,-6] = dP0
                DF_c[-3,-5] = dp0
        
                ## Frequencies
                # w1
                F_c[-2] = P*w1 - 2*numpy.pi
                DF_c[-2,-6] = w1
                DF_c[-2,-4] = P
        
                # w2
                F_c[-1] = P*w2 - p
                DF_c[-1,-6] = w2
                DF_c[-1,-3] = P
                
        
                ## Newton Update 
                dz = -scipy.sparse.linalg.spsolve(scipy.sparse.lil_matrix.tocsr(DF_c),F_c)
                test1 = numpy.sqrt(numpy.dot(F_c[0:-6],F_c[0:-6])/Npts + numpy.dot(F_c[-6:],F_c[-6:]))
                test2 = numpy.sqrt(numpy.dot(dz[0:-6],dz[0:-6])/Npts + numpy.dot(dz[-6:-2],dz[-6:-2]))
                print('|F|* =',test1,'|dz|* =',test2,'|F| =',numpy.linalg.norm(F_c),'|dz| =',numpy.linalg.norm(dz))
              
                ## Check for convergence
                if(test1 < Tol or test2 < Tol):
            
                    # Success!
                    print('QP torus has been found!\n\n')
                        
                    # Plot
                    #if(Plt):
                    #    plt.figure(97)
                    #    plt.clf()
                    #    clr  = numpy.linspace(0,1,Nmax)
                    #    for kk in range(0,n*(m+1)):
                    #        ut = numpy.reshape(U[:,kk],(N,d)).T
                    #        if(d == 4):
                    #            plot(ut[0,:],ut[1,:],'.','Color',[0,clr(ii),1-clr(ii)])
                    #        else:
                    #            plt.axes(prejection='3d').plot3D(ut[0,:],ut[1,:],ut[2,:],'.','Color',[0,clr(ii),1-clr(ii)])
                    #    plt.axis('equal')
                    #    plt.show()            
            
                    ## Stability
                    A = scipy.sparse.lil_matrix.tocsc(DF_c[D:-6,0:D])
                    B = scipy.sparse.lil_matrix.tocsc(DF_c[D:-6,D:-6])
                    PHI  = scipy.sparse.bmat([[scipy.sparse.eye(D)], 
                        [-scipy.sparse.linalg.spsolve(B,A)]])
                    PHI = scipy.sparse.lil_matrix(PHI)
                    
                    # Floquet Matrix
                    G = (R@PHI[-D:,0:D]).toarray()
                        
                    ## Compute Family Tangent
                    dz0  = numpy.append(U.flatten('F') - U0.flatten('F'), [P - P0, p - p0, w1 - w10, w2 - w20, 0, 0])
                    dz0  /= numpy.sqrt(numpy.dot(dz0[0:-6],dz0[0:-6])/Npts + numpy.dot(dz0[-6:-4],dz0[-6:-4]))
                    dU0  = dz0[0:-6].reshape((D,Ns),order='F')
                    dP0  = dz0[-6]
                    dp0  = dz0[-5]
                    dw10 = dz0[-4]
                    dw20 = dz0[-3]
                    dl10 = 0
                    dl20 = 0
                        
                    ## Step-length Controller
                    Eps  = Opt/jj
                    if(Eps > 2):
                        Eps = 2
                    elif(Eps < 0.5):
                        Eps = 0.5

                    ds = numpy.amin([dsMax, Eps*ds])
                                                    
                    ## Store Results
                    Xqp[:,:,ii] = U
                    Wqp[:,ii]   = [P, p, w1, w2]
                    Bqp[:,:,ii] = G
                    Zqp[:,ii]   = dz0
                    Sqp[ii]     = ds
                       
                    ## Update Old Solution
                    U0          = U
                    P0          = P
                    p0          = p
                    w10         = w1
                    w20         = w2
                    l10         = 0
                    l20         = 0
                    break
                       
                else:
                    # Newton's Update
                    U  += dz[0:-6].reshape((D,n*(m+1)+1),order='F')
                    P  += dz[-6]
                    p  += dz[-5]
                    w1 += dz[-4]
                    w2 += dz[-3]
                    l1 += dz[-2]
                    l2 += dz[-1]

            if jj < Iter:
                break
            else:
                # If convergence was not achieve, remove remaining tori and exit from
                # the function
                print('QP torus could not be found!\n\n')
                ds = ds/10
                print('Retrying with ds =',ds,'\n')

        # If convergence was not achieve, remove remaining tori and exit from
        # the function
        if(jj == Iter):
            print('QP torus could not be found!\n\n')

    outputs = {'Xqp': Xqp, 'Wqp': Wqp, 'Bqp': Bqp, 'Zqp': Zqp, 'Sqp': Sqp}
    return outputs