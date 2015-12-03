import numpy as np
import scipy.integrate as sc
import sympy as sym

def f3D(y,s,*args):
    C,u,v,w = args
    y0 = y[0] # u
    y1 = y[1] # u'
    y2 = y[2] # v
    y3 = y[3] # v'
    y4 = y[4] # w
    y5 = y[5] # w'
    C = C.subs({u:y0,v:y2,w:y4})

    dy = np.zeros_like(y)
    dy[0] = y1
    dy[2] = y3
    dy[4] = y5
    dy[1] = -C[0,0][0]*dy[0]**2\
          -2*C[0,0][1]*dy[0]*dy[2]\
          -2*C[0,0][2]*dy[0]*dy[4]\
          -2*C[0,1][2]*dy[2]*dy[4]\
            -C[0,1][1]*dy[2]**2\
            -C[0,2][2]*dy[4]**2
    dy[3] = -C[1,0][0]*dy[0]**2\
          -2*C[1,0][1]*dy[0]*dy[2]\
          -2*C[1,0][2]*dy[0]*dy[4]\
          -2*C[1,1][2]*dy[2]*dy[4]\
            -C[1,1][1]*dy[2]**2\
            -C[1,2][2]*dy[4]**2
    dy[5] = -C[2,0][0]*dy[0]**2\
          -2*C[2,0][1]*dy[0]*dy[2]\
          -2*C[2,0][2]*dy[0]*dy[4]\
          -2*C[2,1][2]*dy[2]*dy[4]\
            -C[2,1][1]*dy[2]**2\
            -C[2,2][2]*dy[4]**2
    return dy

def f(y,s,*args):
    """
    The geodesic differential equations are solved. 
    Described as a system of first order differential-
    equations :

    y0 = u
    y1 = u'
    y2 = v
    y3 = v'

    dy0 = y1
    dy1 = u''
    dy2 = y2
    dy3 = v''     
    
    Input : 
    C is the Christoffel symbol of second kind
    u and v are symbolic expressions.

    Output :
    dy = [dy0,dy1,dy2,dy3]
    """
    C,u,v = args
    y0 = y[0] # u
    y1 = y[1] # u'
    y2 = y[2] # v
    y3 = y[3] # v'
    dy = np.zeros_like(y)
    dy[0] = y1
    dy[2] = y3

    C = C.subs({u:y0,v:y2})
    dy[1] = -C[0,0][0]*dy[0]**2\
          -2*C[0,0][1]*dy[0]*dy[2]\
            -C[0,1][1]*dy[2]**2
    dy[3] = -C[1,0][0]*dy[0]**2\
          -2*C[1,0][1]*dy[0]*dy[2]\
            -C[1,1][1]*dy[2]**2
    return dy

def solve(C,u0,s0,s1,ds,solver=None):
    from sympy.abc import u,v
    global f
    if len(u0) == 6: # 3D problem
        from sympy.abc import w
        args = (C,u,v,w)
        f = f3D
    else:
        args = (C,u,v)

    if solver == None: # use lsoda from scipy.integrate.odeint
        s = np.arange(s0,s1+ds,ds)
        print 'Running solver ...'
        return sc.odeint(f,u0,s,args=args)
    else: # use any other solver from scipy.integrate.ode
        # vode,zvode,lsoda,dopri5,dop853
        r = sc.ode(lambda t,x,args: f(x,t,*args)).set_integrator(solver)
        r.set_f_params(args)
        r.set_initial_value(u0)
        y = []
        print 'Running solver ...'
        while r.successful() and r.t <= s1:
            r.integrate(r.t + ds)
            y.append(r.y)
        return np.array(y)

def two_points(p1,p2,s0,s1,ds,C,tol=1e-6,surface=None):
    """
    The function attempts to find the geodesic between two points p1 and p2.
    """
    p1 = np.array(p1)
    p2 = np.array(p2)
    if (np.fabs(p1-p2) <= tol).all() == 1:
        raise ValueError('Point 1 and point 2 are the same point : (%.1f,%.1f)'%(p2[0],p2[1]))
    found = False
    X_ = []
    u_ = 4*[0]; u_[0] = p1[0]; u_[2] = p1[1]
    du = np.arange(-.2,.2,ds)
    N = du.shape[0]
    i = 0
    while (i < N) and (not found):
        u_[1] = du[i]
        j = 0
        while (j < N) and (not found):
            u_[3] = du[j]
            print 'Testing initial conditions :'
            print u_
            X = solve(C,u_,s0,s1,ds)
            u__ = np.where(np.fabs(X[:,0]-p2[0]) <= tol)[0]
            v__ = np.where(np.fabs(X[:,2]-p2[1]) <= tol)[0]
            if (u__ == v__).any() == True:
                found = True
                X_ = X
                print 'Following initial conditions connect the two provided points '
                print  '(%.6f,%.6f)'%(u_[0],u_[2]), ', (%.6f,%.6f)'%(p2[0],p2[1])
                print "u' = %f , v' = %f"%(u_[1],u_[3])
            j = j + 1
        i = i + 1
    if (len(X_) > 0) and (surface is not None):
        print 'Plotting the geodesics for provided surface...',surface
        if surface == 'catenoid':
            display_catenoid(u_,s0,s1,ds,show=True)
        elif surface == 'torus':
            display_torus(u_,s0,s1,ds,show=True)
        elif surface == 'sphere':
            display_sphere(u_,s0,s1,ds,show=True)
        elif surface == 'egg_carton':
            display_egg_carton(u_,s0,s1,ds,show=True)
    return X

def Christoffel_2nd(g=None,metric=None): # either g is supplied as arugment or the two-form
    from sympy.abc import u,v
    from sympy.diffgeom import metric_to_Christoffel_2nd
    from sympy import asin, acos, atan, cos, log, ln, exp, cosh, sin, sinh, sqrt, tan, tanh
    if metric is None: # if metric is not specified as two_form
        import tensor as t
        R = t.Riemann(g,g.shape[0])
        metric = R.metric
    C = sym.Matrix(eval(str(metric_to_Christoffel_2nd(metric))))
    return C

def catenoid():
    import find_metric
    g = find_metric.cylindrical_catenoid_coordinates()
    C = Christoffel_2nd(g)
    return C

def torus(a=1,c=2):
    import find_metric
    g = find_metric.torus_metric(a,c)
    C = Christoffel_2nd(g)
    return C

def toroid(u=1,v=None,a=1):
    import find_metric as fm
    g, diff = fm.toroidal_coordinates()
    if v is None:
        g = g.subs('u',u)[:2,:2]
    else:
        g = g.subs('v',v)[1:,1:]
    g = g.subs('a',a)
    
    import tensor as t
    R = t.Riemann(g,dim=2,sys_title='toroid')
    C = Christoffel_2nd(metric=R.metric)
    return C
        

def egg_carton():
    import tensor as t
    import find_metric as fm
    g,diff = find_metric.egg_carton_metric()
    R = t.Riemann(g,dim=2,sys_title='egg_carton',flat_diff=diff)
    """
    # this works :
    from sympy.abc import u,v
    u_,v_ = R.system.coord_functions()
    du,dv = R.system.base_oneforms()
    metric = R.metric.subs({u:u_,v:v_,'dv':dv,'du':du})
    """
    C = Christoffel_2nd(metric=R.metric)
    return C

def flat_kerr(a=0,G=1,M=0.5):
    import find_metric as fm
    from sympy.diffgeom import CoordSystem, Manifold, Patch, TensorProduct
    
    manifold = Manifold("M",3)
    patch = Patch("P",manifold)
    kerr = CoordSystem("kerr", patch, ["u","v","w"])
    u,v,w = kerr.coord_functions()
    du,dv,dw = kerr.base_oneforms()

    g11 = (a**2*sym.cos(v) + u**2)/(-2*G*M*u + a**2 + u**2)
    g22 = a**2*sym.cos(v) + u**2
    g33 = -(1 - 2*G*M*u/(u**2 + a**2*sym.cos(v)))
    # time independent : unphysical ? 
    #g33 = 2*G*M*a**2*sym.sin(v)**4*u/(a**2*sym.cos(v) + u**2)**2 + a**2*sym.sin(v)**2 + sym.sin(v)**2*u**2
    metric = g11*TensorProduct(du, du) + g22*TensorProduct(dv, dv) + g33*TensorProduct(dw, dw)
    C = Christoffel_2nd(metric=metric)
    return C

def flat_sphere():
    import find_metric as fm
    import tensor as t
    g,diff = find_metric.flat_sphere()
    R = t.Riemann(g,dim=2,sys_title='flat_sphere',flat_diff=diff)
    C = Christoffel_2nd(metric=R.metric)
    return C

def sphere():
    from sympy.abc import u,v
    from sympy import tan, cos ,sin
    """    
    return flat_sphere() # in correct entries in Christoffel symbol of 2nd kind
    """
    return sym.Matrix([[(0,-tan(v)),  (0, 0)],[(sin(v)*cos(v), 0), (0, 0)]])

def mobius_strip():
    import find_metric as fm
    import tensor as t
    g,diff = fm.mobius_strip()
    R = t.Riemann(g,dim=2,sys_title='mobius_strip',flat_diff = diff)
    #metric=R.metric
    from sympy.diffgeom import TensorProduct, Manifold, Patch, CoordSystem
    manifold = Manifold("M",2)
    patch = Patch("P",manifold)
    system = CoordSystem('mobius_strip', patch, ["u", "v"])
    u, v = system.coord_functions()
    du,dv = system.base_oneforms()
    from sympy import cos
    metric = (cos(u/2)**2*v**2/4 + cos(u/2)*v + v**2/16 + 1)*TensorProduct(du, du) + 0.25*TensorProduct(dv, dv)
    C = Christoffel_2nd(metric=metric)
    return C

def display_mobius_strip(u0,s0,s1,ds,solver=None,show=False):
    C = mobius_strip() # Find the Christoffel tensor for mobius strip
    X = solve(C,u0,s0,s1,ds,solver)

    import matplotlib.pylab as plt
    from mpl_toolkits.mplot3d import Axes3D
    u,v = plt.meshgrid(np.linspace(-2*np.pi,np.pi,250),np.linspace(-np.pi,np.pi,250))
    x = (1 + np.cos(u/2.)*v/2.)*np.cos(u)
    y = (1 + np.cos(u/2.)*v/2.)*np.sin(u)
    z = np.sin(u/2.)*v/2.

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=10, azim=81)
    # use transparent colormap
    import matplotlib.cm as cm
    theCM = cm.get_cmap()
    theCM._init()
    alphas = -.5*np.ones(theCM.N)
    theCM._lut[:-3,-1] = alphas
    ax.plot_surface(x,y,z,linewidth=0,cmap=theCM)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.hold('on')

    # plot the parametrized data on to the catenoid
    u,v = X[:,0], X[:,2]
    x = (1 + np.cos(u/2.)*v/2.)*np.cos(u)
    y = (1 + np.cos(u/2.)*v/2.)*np.sin(u)
    z = np.sin(u/2.)*v/2.

    ax.plot(x,y,z,'--r')
    s0_ = s0/np.pi
    s1_ = s1/np.pi
    fig.suptitle("$s\in[%1.f,%1.f\pi]$ , $u = %.1f$ , $u' = %.1f$, $v = %.1f$ , $v' = %.1f$"%(s0,s1_,u0[0],u0[1],u0[2],u0[3]))
    if show == True:
        plt.show()
    return X,plt
    
def display_catenoid(u0,s0,s1,ds,solver=None,show=False):
    C = catenoid() # Find the Christoffel tensor for cylindrical catenoid
    X = solve(C,u0,s0,s1,ds,solver)

    import matplotlib.pylab as plt
    from mpl_toolkits.mplot3d import Axes3D
    N = X[:,0].shape[0]
    u,v = plt.meshgrid(np.linspace(-np.pi,np.pi,150),np.linspace(-np.pi,np.pi,150))
    x = np.cos(u) - v*np.sin(u)
    y = np.sin(u) + v*np.cos(u)
    z = v

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=20, azim=-163)
    # use transparent colormap
    import matplotlib.cm as cm
    theCM = cm.get_cmap()
    theCM._init()
    alphas = -.5*np.ones(theCM.N)
    theCM._lut[:-3,-1] = alphas
    ax.plot_surface(x,y,z,linewidth=0,cmap=theCM)
    plt.hold('on')

    # plot the parametrized data on to the catenoid
    u,v = X[:,0], X[:,2]
    x = np.cos(u) - v*np.sin(u)
    y = np.sin(u) + v*np.cos(u)
    z = v

    ax.plot(x,y,z,'--r')
    s0_ = s0/np.pi
    s1_ = s1/np.pi
    fig.suptitle("$s\in[%.1f\pi,%.1f\pi]$ , $u' = %.1f$ , $v' = %.2f$"%(s0_,s1_,u0[1],u0[3]))
    if show == True:
        plt.show()
    return X,plt

def display_sphere(u0,s0,s1,ds,solver=None,metric=None,show=False):
    if metric == 'flat':    
        C = flat_sphere()
        if u0[0] == 0 or u0[2] == 0:
            print 'Division by zero may occur for provided values of u(s0) and v(s0)'
    else:
        C = sphere()
    X = solve(C,u0,s0,s1,ds,solver)
    import matplotlib.pylab as plt
    from mpl_toolkits.mplot3d import Axes3D
    u,v = plt.meshgrid(np.linspace(0,2*np.pi,250),np.linspace(0,2*np.pi,250))
    x = np.cos(u)*np.cos(v)
    y = np.sin(u)*np.cos(v)
    z = np.sin(v)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    if metric == 'flat':
        ax.view_init(elev=90., azim=0)
    else:
        ax.view_init(elev=0., azim=13)
    ax.plot_surface(x,y,z,linewidth=0,cmap='Pastel1')
    plt.hold('on')
    # plot the parametrized data on to the sphere
    u,v = X[:,0], X[:,2]
    x = np.cos(u)*np.cos(v)
    y = np.sin(u)*np.cos(v)
    z = np.sin(v)

    ax.plot(x,y,z,'--r')
    from math import pi
    s1_ = s1/pi
    fig.suptitle("$s\in[%1.f,%1.f\pi]$ , $u' = %.1f$ , $v' = %.1f$"%(s0,s1_,u0[1],u0[3]))
    if show == True:
        plt.show()
    return X,plt

def display_torus(u0,s0,s1,ds,a=1,c=2,solver=None,show=False):
    C = torus(a,c) # Find the Christoffel tensor for the torus
    X = solve(C,u0,s0,s1,ds,solver)

    import matplotlib.pylab as plt
    from mpl_toolkits.mplot3d import Axes3D
    N = X[:,0].shape[0]
    u,v = plt.meshgrid(np.linspace(0,2*np.pi,250),np.linspace(0,2*np.pi,250))
    x = (c + a*np.cos(v))*np.cos(u)
    y = (c + a*np.cos(v))*np.sin(u)
    z = np.sin(v)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=-60, azim=100)
    # use transparent colormap -> negative
    import matplotlib.cm as cm
    theCM = cm.get_cmap()
    theCM._init()
    alphas = 2*np.ones(theCM.N)
    theCM._lut[:-3,-1] = alphas
    ax.plot_surface(x,y,z,linewidth=0,cmap=theCM)
    plt.hold('on')

    # plot the parametrized data on to the torus
    u,v = X[:,0], X[:,2]
    x = (c + a*np.cos(v))*np.cos(u)
    y = (c + a*np.cos(v))*np.sin(u)
    z = np.sin(v)

    ax.plot(x,y,z,'--r')
    s1_ = s1/pi
    fig.suptitle("$s\in[%1.f,%1.f\pi]$ , $u = %.1f$ , $u' = %.1f$, $v = %.1f$ , $v' = %.1f$"%(s0,s1_,u0[0],u0[1],u0[2],u0[3]))
    if show == True:
        plt.show() 
    return X,plt

def display_toroid(u0,s0,s1,ds,u_val=1,v_val=None,a=1,solver=None,show=False):
    C = toroid(u_val,v_val,a) # Find the Christoffel tensor for toroid
    X = solve(C,u0,s0,s1,ds,solver)
    
    import matplotlib.pylab as plt
    from mpl_toolkits.mplot3d import Axes3D
    from math import pi
    if v_val is None:
        u = u_val # toroids
        v,w = plt.meshgrid(np.linspace(-pi,pi,250),np.linspace(0,2*pi,250))
    else:
        v = v_val # spherical bowls
        u,w = plt.meshgrid(np.linspace(0,2,250),np.linspace(0,2*pi,250))

    x = (a*np.sinh(u)*np.cos(w))/(np.cosh(u) - np.cos(v))
    y = (a*np.sinh(u)*np.sin(w))/(np.cosh(u) - np.cos(v))  
    z = (a*np.sin(v))/(np.cosh(u) - np.cos(v))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=90., azim=0)
    # use transparent colormap
    import matplotlib.cm as cm
    theCM = cm.get_cmap()
    theCM._init()
    alphas = -.5*np.ones(theCM.N)
    theCM._lut[:-3,-1] = alphas
    ax.plot_surface(x,y,z,linewidth=0,cmap=theCM)
    plt.hold('on')

    # plot the parametrized data on to the toroid
    if v_val is None:
        w,v = X[:,0], X[:,2]
    else:
        w,u = X[:,0], X[:,2]
    x = (a*np.sinh(u)*np.cos(w))/(np.cosh(u) - np.cos(v))
    y = (a*np.sinh(u)*np.sin(w))/(np.cosh(u) - np.cos(v))  
    z = (a*np.sin(v))/(np.cosh(u) - np.cos(v))

    s1_ = s1/pi
    ax.plot(x,y,z,'--r')
    fig.suptitle('$s\in[%.1f\pi \, , \,%2.1f\pi]$'%(s0,s1_))
    if show == True:
        plt.show()
    return X,plt

def display_egg_carton(u0,s0,s1,ds,solver=None,show=False):
    C = egg_carton() # Find the Christoffel tensor for egg carton surface
    X = solve(C,u0,s0,s1,ds,solver)

    import matplotlib.pylab as plt
    from mpl_toolkits.mplot3d import Axes3D
    from math import pi
    N = X[:,0].shape[0]
    u,v = plt.meshgrid(np.linspace(s0,s1,N),np.linspace(s0,s1,N))
    x = u
    y = v
    z = np.sin(u)*np.cos(v)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=90., azim=0)
    # use transparent colormap
    import matplotlib.cm as cm
    theCM = cm.get_cmap()
    theCM._init()
    alphas = -.5*np.ones(theCM.N)
    theCM._lut[:-3,-1] = alphas
    ax.plot_surface(x,y,z,linewidth=0,cmap=theCM)
    plt.hold('on')

    # plot the parametrized data on to the egg carton
    u,v = X[:,0], X[:,2]
    x = u
    y = v
    z = np.sin(u)*np.cos(v)

    s0_ = s0/pi
    s1_ = s1/pi
    ax.plot(x,y,z,'--r')
    fig.suptitle('$s\in[%.1f\pi \, , \,%2.1f\pi]$'%(s0_,s1_))
    if show == True:
        plt.show()
    return X,plt

def display_3D_Kerr(u0,s0,s1,ds,solver=None,show=True,args=None,multiple=True):
    if args == None:
        C = flat_kerr() # use default values
    else:
        a = args[0]
        G = args[1]
        M = args[2]
        C = flat_kerr(a,G,M) # Find the Christoffel tensor for 3D Kerr metric on 4D manifold

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    if multiple is not True:
        X = solve(C,u0,s0,s1,ds,solver)
        r = X[:,0]
        theta = X[:,2]
        # for time independent kerr metric use :
        #phi = X[:,4]
        #x = r*np.sin(theta)*np.cos(phi)
        #y = r*np.sin(theta)*np.sin(phi)
        #z = r*np.cos(theta)
        #ax.plot(x,y,z,'b')
        t = X[:,4]
        x = r*np.sin(theta)
        y = r*np.cos(theta)
        z = t 
        ax.plot(x,y,z,'b')
    
    if multiple is True:
        plt.hold('on')
        N = 50
        t = np.linspace(0.01,np.pi-.01,N)
        for i in range(N):
            u0[0] = np.sin(t[i])
            u0[2] = np.cos(t[i])
            if u0[0] < 0:
                u0[0] = -.71+u0[0]
            else:
                u0[0] = .71+u0[0]
            print 'i=%d'%i, u0
            X = solve(C,u0,s0,s1,ds,solver)
            r = X[:,0]
            theta = X[:,2]
            #phi = X[:,4]
            #x = r*np.sin(theta)*np.cos(phi)
            #y = r*np.sin(theta)*np.sin(phi)
            #z = r*np.cos(theta)
            #ax.plot(x,y,z,'b')
            t = X[:,4]
            x = r*np.sin(theta)
            y = r*np.cos(theta)
            z = t 
            ax.plot(x,y,z,'b')
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()
  
    return X,plt

def display_multiple_geodesics(u0,s0,s1,ds,surface,with_object=True):
    import matplotlib.pylab as plt
    from mpl_toolkits.mplot3d import Axes3D
    def solve_multiple(C,u0,s0,s1,ds,s):
        from sympy.abc import u,v
        print 'Running solver...'
        return sc.odeint(f,u0,s,args=(C,u,v))
    
    def display_multiple_catenoid(u0,s0,s1,ds,C,s):
        X = solve_multiple(C,u0,s0,s1,ds,s)
        plt.hold('on')
        # plot the parametrized data on to the catenoid
        u,v = X[:,0], X[:,2]
        x = np.cos(u) - v*np.sin(u)
        y = np.sin(u) + v*np.cos(u)
        z = v
        ax.plot(x,y,z,'--r')
        return plt

    def display_multiple_egg_carton(u0,s0,s1,ds,C,s):
        X = solve_multiple(C,u0,s0,s1,ds,s)
        plt.hold('on')
        # plot the parametrized data on to the egg carton
        u,v = X[:,0], X[:,2]
        x = u
        y = v
        z = np.sin(u)*np.cos(v)
        ax.plot(x,y,z,'--r')
        return plt

    def display_multiple_sphere(u0,s0,s1,ds,C,s):
        X = solve_multiple(C,u0,s0,s1,ds,s)
        plt.hold('on')
        # plot the parametrized data on to the sphere
        u,v = X[:,0], X[:,2]
        x = np.cos(u)*np.cos(v)
        y = np.sin(u)*np.cos(v)
        z = np.sin(v)
        ax.plot(x,y,z,'--r')
        return plt

    def display_multiple_torus(u0,s0,s1,ds,C,s):
        X = solve_multiple(C,u0,s0,s1,ds,s)
        plt.hold('on')
        # plot the parametrized data on to the sphere
        u,v = X[:,0], X[:,2]
        x = (2 + 1*np.cos(v))*np.cos(u)
        y = (2 + 1*np.cos(v))*np.sin(u)
        z = np.sin(v)
        ax.plot(x,y,z,'--r')
        return plt

    u0_range = np.arange(s0,s1+ds,ds)
    N = u0_range.shape[0]
        
    fig = plt.figure()
    if surface == 'catenoid':
        if with_object:
            u,v = plt.meshgrid(np.linspace(-np.pi,np.pi,150),np.linspace(-np.pi,np.pi,150))
            x = np.cos(u) - v*np.sin(u)
            y = np.sin(u) + v*np.cos(u)
            z = v
        C = catenoid()
    elif surface == 'egg_carton':
        if with_object:
            u,v = plt.meshgrid(np.linspace(-4,4,250),np.linspace(-4,4,250))
            x = u
            y = v
            z = np.sin(u)*np.cos(v)
        C = egg_carton()
    elif surface == 'sphere':
        if with_object:
            u,v = plt.meshgrid(np.linspace(0,2*np.pi,250),np.linspace(0,2*np.pi,250))
            x = np.cos(u)*np.cos(v)
            y = np.sin(u)*np.cos(v)
            z = np.sin(v)
        C = sphere()
    elif surface == 'torus':
        if with_object:
            u,v = plt.meshgrid(np.linspace(0,2*np.pi,150),np.linspace(0,2*np.pi,150))
            x = (2 + 1*np.cos(v))*np.cos(u)
            y = (2 + 1*np.cos(v))*np.sin(u)
            z = np.sin(v)
        C = torus()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(azim=65, elev=67)
    if with_object:
        theCM = 'Pastel1'
        ax.plot_surface(x,y,z,linewidth=0,cmap=theCM)
    plt.hold('on')

    if surface == 'catenoid':
        for u_val in u0_range:
            u0[3] = u_val
            plt = display_multiple_catenoid(u0,s0,s1,ds,C,u0_range)
    elif surface == 'egg_carton':
        for u_val in u0_range:
            u0[0] = u_val
            plt = display_multiple_egg_carton(u0,s0,s1,ds,C,u0_range)
    elif surface == 'sphere':
        for u_val in u0_range:
            u0[0] = u_val
            plt = display_multiple_sphere(u0,s0,s1,ds,C,u0_range)
    elif surface == 'torus':
        for u_val in u0_range:
            u0[0] = u_val # alternate v0 values
            plt = display_multiple_torus(u0,s0,s1,ds,C,u0_range)
            

    from math import pi
    s0_ = s0#/pi
    s1_ = s1/pi
    fig.suptitle("$s\in[%1.f,%1.f\pi]$ , $u' = %.1f$ , $v = %.1f$ , $v' = %.1f$"%(s0_,s1_,u0[1],u0[2],u0[3]))
    plt.show()
  
if __name__ == '__main__':
    import sys
    from math import pi, sqrt
    if len(sys.argv) > 1:
        u0 = eval(sys.argv[1])  # evaluate the input for a list [u(0), u'(0), v(0), v'(0)]
        if type(u0) is not list:
            raise TypeError("The first argument must be a list : [u(0),v(0),u'(0),v'(0)]")
        s0 = float(eval(sys.argv[2])) # evaluate math expressions such as pi, '*', sqrt
        s1 = float(eval(sys.argv[3]))
        ds = float(eval(sys.argv[4]))
        display = sys.argv[5]
        if display == 'catenoid':
            display_catenoid(u0,s0,s1,ds)
        if display == 'sphere':
            display_sphere(u0,s0,s1,ds)
        if display == 'torus':
            display_torus(u0,s0,s1,ds)
        if display == 'egg_carton':
            display_egg_carton(u0,s0,s1,ds)
    else:
        """
        u0 = [0,-.5,.5,0] # u(0), u'(0), v(0), v'(0)
        s0 = -pi/2
        s1 = 3*pi
        ds = 0.1
        display_catenoid(u0,s0,s1,ds,show=True)
        #display_multiple_geodesics(u0,s0,s1,ds,'catenoid',with_object=False)

        u0 = [0.75,0.1,.75,0.1]
        s0 = 0
        s1 = 18*pi
        ds = 2
        #display_sphere(u0,s0,s1,ds,show=True)
        display_multiple_geodesics(u0,s0,s1,ds,'sphere',with_object=False)

        u0 = [0,.2,0,.2] # u(s0), u'(s0), v(s0), v'(s0)
        s0 = 0
        s1 = 25*pi
        ds = .1
        a = 1
        c = -2
        display_torus(u0,s0,s1,ds,a=a,c=c,show=True)
        #display_multiple_geodesics(u0,s0,s1,ds,'torus',with_object=False)

        u0 = [0,.5,0.5,sqrt(3)/2] # u(0), u'(0), v(0), v'(0)
        s0 = -pi
        s1 = pi
        ds = 0.05
        display_egg_carton(u0,s0,s1,ds,show=True)
        #display_multiple_geodesics(u0,s0,s1,ds,'egg_carton')
        
        u0 = [1.5,.1,-1,0] # u(0), u'(0), v(0), v'(0)
        s0 = 0
        s1 = 80
        ds = 0.1
        display_mobius_strip(u0,s0,s1,ds,show=True,solver=None)


        u0 = [0,0,0,.1] # u(s0), u'(s0), v(s0), v'(s0)
        s0 = 0
        s1 = 10*pi
        ds = 0.5
        display_toroid(u0,s0,s1,ds,u_val=1,show=True)

        # check if two points are connected on great circle
        p1 = (1,1) # taken from a sphere simulation for u' = .2, v' = 0
        p2 = (0.16242917238268037, 0.80611950949349132) # entry 200 in X
        s0 = 0
        s1 = 18*pi
        ds = 0.05
        C = sphere()
        two_points(p1,p2,s0,s1,ds,C,tol=1e-6,surface='sphere')
        """
        u0 = [.7,.1,.1,.1,0,.1] # u(s0), u'(s0), v(s0), v'(s0), w(s0), w'(s0)
        s0 = 0
        s1 = 18
        ds = 0.01
        # A singularity near origo : a = 0, G = 1, M = 0.5
        a = 0
        G = 1
        M = 0.35
        display_3D_Kerr(u0,s0,s1,ds,show=True,solver=None,args = (a,G,M))
