from __future__ import division
import scipy.integrate as sc
import numpy as np
import numpy.linalg as nplin

def find_eigen(T):
    """
    Returns the eigenvalues and eigenvectors for a second order tensor T.
    """
    dim = T.shape[0]
    eig_data = nplin.eig(T)
    eig_values = eig_data[0]
    v1 = eig_data[1][:,0]
    v2 = eig_data[1][:,1]
    if dim == 3:
        v3 = eig_data[1][:,2]
        return eig_values,v1,v2,v3
    else:
        return eig_values,v1,v2

from scipy.interpolate import griddata
def integrate(grid_points,U,p0,s,direction='major',solver=None):
    dim = 2
    if len(U) == 12:
        dim = 3
        U_x, U_y, U_z, U_x_, U_y_, U_z_, U_x__, U_y__, U_z__,l_minor, l_major, l_medium = U

        if direction == 'major':
            U__x = U_x.flatten(); U__y = U_y.flatten(); U__z = U_z.flatten();
        else:
            U__x = U_x_.flatten(); U__y = U_y_.flatten(); U__z = U_z_.flatten();
        points = zip(grid_points[0].flatten(),grid_points[1].flatten(),grid_points[2].flatten())
        def f_3D(x,t):
            return [griddata(points,U__x,x)[0], griddata(points,U__y,x)[0], griddata(points,U__z,x)[0]]
        f = f_3D 
    else:
        U_x, U_y, U_x_, U_y_, l_minor, l_major = U
        if direction == 'major':
            U__x = U_x.flatten(); U__y = U_y.flatten();
        else:
            U__x = U_x_.flatten(); U__y = U_y_.flatten();
        points = zip(grid_points[0].flatten(),grid_points[1].flatten())
        def f_2D(x,t):
            return [griddata(points,U__x,x)[0], griddata(points,U__y,x)[0]]
        f = f_2D 

    if solver == None: # use lsoda from scipy.integrate.odeint        
        print 'Running solver ...'
        p = sc.odeint(f,p0,s)
    else: # use any other solver from scipy.integrate.ode
        # vode,zvode,lsoda,dopri5,dop853
        r = sc.ode(lambda t,x: f(x,t)).set_integrator(solver)
        r.set_initial_value(p0)
        y = []
        t_end = s[-1] 
        dt = s[1]-s[0]
        print 'Running solver ...'
        while r.successful() and r.t <= t_end:
            r.integrate(r.t + dt)
            y.append(r.y)
            p = np.array(y) 
    if direction == 'major':
        if dim == 2:
            p_ = _expand_hyperstreamline(p,U_x_,U_y_,l_minor,points)
        else:
            p_ = _expand_hyperstreamline3D(p,U_x_,U_y_,U_z_,U_x__,U_y__,U_z__,l_minor,l_medium,points)
    else:
        if dim == 2:
            p_ = _expand_hyperstreamline(p,U_x,U_y,l_major,points)
        else:
            p_ = _expand_hyperstreamline3D(p,U_x,U_y,U_z,U_x__,U_y__,U_z__,l_major,l_medium,points)
    return p,p_

def _expand_hyperstreamline(p,Ux,Uy,l,points):
    p_ = np.zeros_like(p)
    def f(x):
        return [griddata(points,Ux,x)[0], griddata(points,Uy,x)[0]]
    i = 0    
    for p_val in p:
        print p_val
        p_[i] = f(p_val)*l[p_val]
        i = i + 1
    return p_
    

def _expand_hyperstreamline3D(p,Ux,Uy,Uz,Ux_,Uy_,Uz_,l,l_,points):
    p_ = np.zeros_like(p)
    def f1(x):
        return [griddata(points,Ux,x)[0], griddata(points,Uy,x)[0],griddata(points,Uz,x)[0]]
    def f2(x):
        return [griddata(points,Ux_,x)[0], griddata(points,Uy_,x)[0],griddata(points,Uz_,x)[0]]
    return p_

def extract_eigen(eigen_field):
    """
    Performs a sorting of minor, major, and (if 3D) medium eigenvectors and eigenvalues.
    The 'eigen_field' is assumed to be of the form
    
                 [[ l1,  l2],
                  [v1x, v1y],
                  [v2x, v2y]]
    and for 3D
                 [[ l1,  l2,  l3],
                  [v1x, v1y, v1z],
                  [v2x, v2y, v2z],
                  [v3x, v3y, v3z]]
    Finally, the corresponding eigenvalues are returned as well.
    """
    return _sort_eig(eigen_field)

def _sort_eig(U):
    dim = U.shape[1]
    Nx = U.shape[2]  
    Ny = U.shape[3]
    if dim == 3:
        Nz = U.shape[4]
        return _sort_eig_3D(U,Nx,Ny,Nz)

    U_x = np.zeros([Nx,Ny]);   U_y = np.zeros_like(U_x); # major eigenvectors
    U_x_ = np.zeros_like(U_x); U_y_ = np.zeros_like(U_x); # minor eigenvectors
    l_major = np.zeros_like(U_x); l_minor = np.zeros_like(U_x)
    print 'Sorting eigenvalues and eigenvectors ...'   
    for i in range(Nx):
        for j in range(Ny):
            if U[0,0,i,j] <= U[0,1,i,j]: # if lambda_1 < lambda_2
                l_minor[i,j] = U[0,0,i,j]
                l_major[i,j] = U[0,1,i,j]
                U_x[i,j] = U[2,0,i,j]
                U_y[i,j] = U[2,1,i,j]
                U_x_[i,j] = U[1,0,i,j]
                U_y_[i,j] = U[1,1,i,j]
            else:
                l_major[i,j] = U[0,1,i,j]
                l_minor[i,j] = U[0,0,i,j]
                U_x[i,j] = U[1,0,i,j]
                U_y[i,j] = U[1,1,i,j]
                U_x_[i,j] = U[2,0,i,j]
                U_y_[i,j] = U[2,1,i,j]
    return U_x_, U_y_, U_x, U_y, l_minor, l_major                   

def _sort_eig_3D(U,Nx,Ny,Nz):
    U_x = np.zeros([Nx,Ny,Nz]); U_y = np.zeros_like(U_x); U_z = np.zeros_like(U_x); # major
    U_x_ = np.zeros_like(U_x); U_y_ = np.zeros_like(U_x); U_z_ = np.zeros_like(U_x); # minor
    U_x__ = np.zeros_like(U_x); U_y__ = np.zeros_like(U_x); U_z__ = np.zeros_like(U_x); # medium
    l_major = np.zeros_like(U_x); l_minor = np.zeros_like(U_x); l_medium = np.zeros_like(U_x);
    print 'Sorting eigenvalues and eigenvectors ...'  
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                if (U[0,0,i,j,k] >= U[0,1,i,j,k]) and (U[0,0,i,j,k] >= U[0,2,i,j,k]):
                    l_major[i,j,k] = U[0,0,i,j,k]
                    if U[0,1,i,j,k] >= U[0,2,i,j,k]:
                        l_minor[i,j,k] = U[0,2,i,j,k]
                        l_medium[i,j,k] = U[0,1,i,j,k]
                        U_x_[i,j,k] = U[2,0,i,j,k]
                        U_y_[i,j,k] = U[2,1,i,j,k]
                        U_z_[i,j,k] = U[2,2,i,j,k]
                        U_x__[i,j,k] = U[3,0,i,j,k]
                        U_y__[i,j,k] = U[3,1,i,j,k]
                        U_z__[i,j,k] = U[3,2,i,j,k]
                    else:
                        l_minor[i,j,k] = U[0,1,i,j,k]
                        l_medium[i,j,k] = U[0,2,i,j,k]
                        U_x_[i,j,k] = U[3,0,i,j,k]
                        U_y_[i,j,k] = U[3,1,i,j,k]
                        U_z_[i,j,k] = U[3,2,i,j,k]
                        U_x__[i,j,k] = U[2,0,i,j,k]
                        U_y__[i,j,k] = U[2,1,i,j,k]
                        U_z__[i,j,k] = U[2,2,i,j,k]
                    U_x[i,j,k] = U[1,0,i,j,k]
                    U_y[i,j,k] = U[1,1,i,j,k]
                    U_z[i,j,k] = U[1,2,i,j,k]
                elif (U[0,1,i,j,k] >= U[0,0,i,j,k]) and (U[0,1,i,j,k] >= U[0,2,i,j,k]):
                    l_major[i,j,k] = U[0,1,i,j,k]
                    if U[0,0,i,j,k] >= U[0,2,i,j,k]:
                        l_minor[i,j,k] = U[0,2,i,j,k]
                        l_medium[i,j,k] = U[0,0,i,j,k]
                        U_x_[i,j,k] = U[3,0,i,j,k]
                        U_y_[i,j,k] = U[3,1,i,j,k]
                        U_z_[i,j,k] = U[3,2,i,j,k]
                        U_x__[i,j,k] = U[1,0,i,j,k]
                        U_y__[i,j,k] = U[1,1,i,j,k]
                        U_z__[i,j,k] = U[1,2,i,j,k]
                    else:
                        l_minor[i,j,k] = U[0,0,i,j,k]
                        l_medium[i,j,k] = U[0,2,i,j,k]
                        U_x_[i,j,k] = U[1,0,i,j,k]
                        U_y_[i,j,k] = U[1,1,i,j,k]
                        U_z_[i,j,k] = U[1,2,i,j,k]
                        U_x__[i,j,k] = U[3,0,i,j,k]
                        U_y__[i,j,k] = U[3,1,i,j,k]
                        U_z__[i,j,k] = U[3,2,i,j,k]
                    U_x[i,j,k] = U[2,0,i,j,k]
                    U_y[i,j,k] = U[2,1,i,j,k]
                    U_z[i,j,k] = U[2,2,i,j,k]
                else:
                    l_major[i,j,k] = U[0,2,i,j,k]
                    if U[0,0,i,j,k] >= U[0,1,i,j,k]:
                        l_minor[i,j,k] = U[0,1,i,j,k]
                        l_medium[i,j,k] = U[0,0,i,j,k]
                        U_x_[i,j,k] = U[2,0,i,j,k]
                        U_y_[i,j,k] = U[2,1,i,j,k]
                        U_z_[i,j,k] = U[2,2,i,j,k]
                        U_x__[i,j,k] = U[1,0,i,j,k]
                        U_y__[i,j,k] = U[1,1,i,j,k]
                        U_z__[i,j,k] = U[1,2,i,j,k]
                    else:
                        l_minor[i,j,k] = U[0,0,i,j,k]
                        l_medium[i,j,k] = U[0,1,i,j,k]
                        U_x_[i,j,k] = U[1,0,i,j,k]
                        U_y_[i,j,k] = U[1,1,i,j,k]
                        U_z_[i,j,k] = U[1,2,i,j,k]
                        U_x__[i,j,k] = U[2,0,i,j,k]
                        U_y__[i,j,k] = U[2,1,i,j,k]
                        U_z__[i,j,k] = U[2,2,i,j,k]
                    U_x[i,j,k] = U[3,0,i,j,k]
                    U_y[i,j,k] = U[3,1,i,j,k]
                    U_z[i,j,k] = U[3,2,i,j,k]
    return U_x, U_y, U_z, U_x_, U_y_, U_z_, U_x__, U_y__, U_z__,l_minor, l_major, l_medium

def _run_example_flat_sphere(xstart,xend,N,direction='major',solver=None):
    """
    A test example, using the metric of a flat sphere, to calculate hyperstreamlines
    for a 2D grid.
    """ 
    x0,y0 = xstart
    xN,yN = xend
    Nx,Ny = N
    x,y = np.mgrid[x0:xN:Nx*1j,y0:yN:Ny*1j]
    # Initialize the metric for the flat sphere
    g = np.array([[1,0],[0,1]],dtype=np.float32)
    T = np.zeros([2,2,Nx,Ny],dtype=np.float32) # The tensor field
    eig_field = np.zeros([3,2,Nx,Ny],dtype=np.float32) # The "eigen" field

    print "Determining eigenvectors for the flat metric of a sphere over the mesh..."
    for i in range(Nx):
        for j in range(Ny):
            g[1,1]= np.sin(y[i,j])**2
            T[:,:,i,j] = g[:,:]
            eig_field[:,:,i,j] = find_eigen(T[:,:,i,j])

    INITIAL_POINT = (1.,1.)
    t0 = 0
    t1 = 2*np.pi
    dt = 0.01  
    t = np.arange(t0,t1+dt,dt)   
    U = extract_eigen(eig_field)
    p,p_ = integrate([x,y],U,INITIAL_POINT,t,direction=direction,solver=solver)
    return p,p_

def _run_example_3D(xstart,xend,N,direction='major',solver=None):
    """
    A 3D test example
    """ 
    x0,y0,z0 = xstart
    xN,yN,yN = xend
    Nx,Ny,Nz = N
    x,y,z = np.mgrid[x0:xN:Nx*1j,y0:yN:Ny*1j,z0:zN:Nz*1j]
    # Initialize the metric for the flat sphere
    g = np.array([[1,0,0],[0,.5,0],[0,0,1]],dtype=np.float32)
    T = np.zeros([3,3,Nx,Ny,Nz],dtype=np.float32) # The tensor field
    eig_field = np.zeros([4,3,Nx,Ny,Nz],dtype=np.float32) # The "eigen" field

    print "Determining eigenvectors for the flat metric of a sphere over the mesh..."
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                g[2,2]= np.sin(y[i,j,k])**2 + np.cos(x[i,j,k])**2
                T[:,:,i,j,k] = g[:,:]
                eig_field[:,:,i,j,k] = find_eigen(T[:,:,i,j,k])

    INITIAL_POINT = (1.,1.,1.)
    t0 = 0
    t1 = 2*np.pi
    dt = 0.01  
    t = np.arange(t0,t1+dt,dt)   
    U = extract_eigen(eig_field)
    p,p_ = integrate([x,y,z],U,INITIAL_POINT,t,direction=direction,solver=solver)
    return p,p_

if __name__ == "__main__":
    import sys
    x0 = 0; y0 = 0; z0 = 0
    xN = np.pi/2; yN = np.pi; zN = 1
    Nx = 22; Ny = 22; Nz = 22
    N = (Nx,Ny,Nz)#N = (Nx,Ny)
    xstart = (x0,y0,z0); xend = (xN,yN,zN) 
    #xstart = (x0,y0); xend = (xN,yN) 
    solver = None # solvers: lsoda (default), vode,zvode,lsoda,dopri5,dop853
   
    if len(sys.argv) > 1:
        if sys.argv[1] == "major":
            p,p_= _run_example_flat_sphere(xstart,xend,N,'major',solver=solver)
        else:
            p,p_= _run_example_flat_sphere(xstart,xend,N,'minor',solver=solver)
    else:
        #p,p_= _run_example_flat_sphere(xstart,xend,N,'major',solver=solver)
        p,p_= _run_example_3D(xstart,xend,N,'major',solver=solver)
    
    import matplotlib.pylab as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')    
    plt.plot(p[:,0],p[:,1],p[:,2])
    plt.show()
