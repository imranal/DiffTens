import sympy as sym 

def curve_to_metric(ds2,dim,diff_=None):
    if dim < 2:
        raise ValueError("The metric is implemented for at least dim = 2.")
    ds2 = str(ds2)
    ds2 = ds2.replace(' ','')
    differentials = []
    if diff_ is None:
        if dim >= 2:
            differentials = ['du','dv']
        if dim >= 3:
            differentials.append('dw')
        if dim == 4:
            differentials.append('dt')
    else:
        M = sym.Matrix(diff_)
        for i in range(0,M.shape[0]):
            n = str(M[i,i]).find('*')
            diff = str(M[i,i])[0:n]
            differentials.append(diff)
    
    for diff in differentials:
        ds2 = ds2.replace(diff+'**2',diff+'*'+diff)
    
    def split_elements(expr,tmp_list,side='right'):
        new_list = []
        if type(tmp_list) is list:
            for term in tmp_list:
                split_terms = term.split(expr)
                if type(split_terms) is list:
                    for sterms in split_terms:
                        new_list.append(sterms)
                else:
                    new_list.append(split_terms)
        else:
            split_terms = tmp_list.split(expr) # split at found expression
            if type(split_terms) is list:
                for sterms in split_terms:
                    new_list.append(sterms) 
            else:
                new_list.append(split_terms)
        
        lost = expr[:-1]
        sign = '-' # special case to be handeled
        if side=='left':
            lost = expr[1:] 
        for i in range(len(new_list)):
            term = new_list[i]
            if side=='left':
                first = 0
                if term[first] == '*':
                    new_list[i] = lost+new_list[i]
            else:
                last = len(term) - 1
                if term[last] == '*':
                    new_list[i] = new_list[i]+lost
                    if expr[-1] == sign: # if expression contains '-' at end
                        # for next in list add '-'
                        new_list[i+1] = '-'+new_list[i+1]
        return new_list

    n = ds2
    L='left'
    R='right'
    p = '+'
    m = '-'
    for diff in differentials: # for each differential : dx_i = du,dv,dw,dt
        expr = diff+p # 'dx_i+'
        n = split_elements(expr,n,R)
        expr = diff+m # 'dx_i-'
        n = split_elements(expr,n,R)
        expr = p+diff # '+dx_i'
        n = split_elements(expr,n,L)
        expr = m+diff # '-dx_i'
        n = split_elements(expr,n,L)
    
    # define the matrix structure of the metric components for mapping the 
    if dim == 2: # curve elements to their corresponding location
        if diff_ is None:
            diff  = [['du*du','du*dv'],
                    ['dv*du','dv*dv']]
        else:
            diff = diff_
    if dim == 3:
        if diff_ is None:
            diff  = [['du*du','du*dv','du*dw'],
                     ['dv*du','dv*dv','dv*dw'],
                     ['dw*du','dw*dv','dw*dw']]
        else:
            diff = diff_
    if dim == 4:
        if diff_ is None:
            diff  = [['du*du','du*dv','du*dw','dt*du'],
                     ['dv*du','dv*dv','dv*dw','dv*dt'],
                     ['dw*du','dv*dw','dw*dw','dw*dt'],
                     ['dt*du','dt*dv','dt*dw','dt*dt']]
        else:
            diff = diff_
    # add the elements in g without the above differentials
    elements = n
    g =  [['0' for _ in range(dim)] for _ in range(dim)]
    for element in elements:
        for i in range(dim):
            for j in range(dim):
                if (element.find(diff[i][j]) != -1):
                    if (element.find('*' + diff[i][j]) != -1):
                        g[i][j] = element.replace('*'+ diff[i][j],'')
                    else:
                        g[i][j] = element.replace(diff[i][j],'1')
                    g[j][i] = g[i][j] 
    from sympy import Matrix, sin, cos, exp, log, cosh, sinh, sqrt, tan, tanh
    from sympy.abc import u,v,w,t
    return Matrix(g)
  

def metric(coord1,coord2,form="simplified",write_to_file=False):
    """
    Calculates the metric for the coordinate transformation
    between cartesian coordinates to another orthogonal
    coordinate system.
    """
    from sympy import diff
    x,y = coord1[0], coord1[1]
    u,v = coord2[0], coord2[1]
    dim = len(coord1)
    if len(coord2) != dim:
        import sys
        sys.exit("Coordinate systems must have same dimensions.") 
    if dim >= 3:
        z = coord1[2]
        w = coord2[2]
    if dim == 4:
        t1 = coord1[3]
        t2 = coord2[3]
    dxdu = diff(x,u)
    dxdv = diff(x,v)
    dydu = diff(y,u)
    dydv = diff(y,v)
    if dim >= 3:
        dxdw = diff(x,w)
        dydw = diff(y,w)
        dzdu = diff(z,u)
        dzdv = diff(z,v)
        dzdw = diff(z,w)
    if dim == 4:
        dxdt = diff(x,t2)
        dydt = diff(y,t2)
        dzdt = diff(z,t2)
        dtdu = diff(t1,u)
        dtdv = diff(t1,v)
        dtdw = diff(t1,w)
        dtdt = diff(t1,t2)
        

    import numpy as np
    from sympy import Matrix
    g = Matrix(np.zeros([dim,dim]))
    g[0,0] = dxdu*dxdu + dydu*dydu 
    g[0,1] = dxdu*dxdv + dydu*dydv
    g[1,1] = dxdv*dxdv + dydv*dydv
    g[1,0] = g[0,1]
    if dim >= 3:
        g[0,0] += dzdu*dzdu
        g[0,1] += dzdu*dzdv; g[1,0] = g[0,1]
        g[0,2] = dxdu*dxdw + dydu*dydw + dzdu*dzdw; g[2,0] = g[0,2]
        g[1,1] += dzdv*dzdv
        g[1,2] = dxdv*dxdw + dydv*dydw + dzdv*dzdw; g[2,1] = g[1,2]
        g[2,2] = dxdw*dxdw + dydw*dydw + dzdw*dzdw
    if dim == 4:
        g[0,0] += dtdu*dtdu
        g[0,1] += dtdu*dtdv; g[1,0] = g[0,1]
        g[0,2] += dtdu*dtdw; g[2,0] = g[0,2]
        g[0,3] = dxdu*dxdt + dydu*dydt + dzdu*dzdt + dtdu*dtdt; g[3,0] = g[0,3]
        g[1,1] += dtdv*dtdv
        g[1,2] += dtdv*dtdw; g[2,1] = g[1,2]
        g[1,3] = dxdv*dxdt + dydv*dydt + dzdv*dzdt + dtdv*dtdt; g[3,1] = g[1,3] 
        g[2,2] += dtdw*dtdw
        g[2,3] = dxdw*dxdt + dydw*dydt + dzdw*dzdt + dtdw*dtdt; g[3,2] = g[2,3]
        g[3,3] = dxdt*dtdt + dydt*dtdt + dzdt*dtdt + dtdt*dtdt
    
    if form=="simplified": 
        def symplify_expr(expr):
            new_expr = sym.trigsimp(expr) 
            new_expr = sym.simplify(new_expr) 
            return new_expr    
        print "Performing simplification on the metric. This may take some time ...."
        for i in range(0,dim):
            for j in range(0,dim):
                g[i,j] = symplify_expr(g[i,j])

    if write_to_file:
        f = open("metric.txt","w")
        f.write(str(g))
        f.close()
    return g

def toroidal_coordinates(form="simplified"):
    from sympy.abc import u, v, w, a
    from sympy import sin,cos,sinh,cosh

    x = (a*sinh(u)*cos(w))/(cosh(u) - cos(v))
    y = (a*sinh(u)*sin(w))/(cosh(u) - cos(v))  
    z = (a*sin(v))/(cosh(u) - cos(v))
    coord1 = (x,y,z)
    coord2 = (u,v,w)
    g = metric(coord1,coord2,form)
    diff_form  = [['du*du','du*dv','du*dw'],
                  ['dv*du','dv*dv','dv*dw'],
                  ['dw*du','dv*dw','dw*dw']]
    return g, diff_form

def cylindrical_coordinates(form="simplified"):
    from sympy.abc import u, v, w, x, y, z
    from sympy import sin,cos

    x = u*cos(v)
    y = u*sin(v)
    z = w 
    coord1 = (x,y,z)
    coord2 = (u,v,w)
    g = metric(coord1,coord2,form)
    diff_form  = [['du*du','du*dv','du*dw'],
                  ['dv*du','dv*dv','dv*dw'],
                  ['dw*du','dv*dw','dw*dw']]
    return g, diff_form

def spherical_coordinates(form="simplified"):
    from sympy.abc import u, v, w, x, y, z
    from sympy import sin,cos

    x = u*sin(v)*cos(w)
    y = u*sin(v)*sin(w)
    z = u*cos(v)
    coord1 = (x,y,z)
    coord2 = (u,v,w)
    g = metric(coord1,coord2,form)
    diff_form  = [['du*du','du*dv','du*dw'],
                  ['dv*du','dv*dv','dv*dw'],
                  ['dw*du','dv*dw','dw*dw']]
    return g, diff_form

def inverse_prolate_spheroidal_coordinates(form="usimp",write_to_file=True):
    from sympy.abc import u, v, w, a
    from sympy import sin,cos,sinh,cosh

    x = (a*sinh(u)*sin(v)*cos(w))/(cosh(u)**2 - sin(v)**2)
    y = (a*sinh(u)*sin(v)*sin(w))/(cosh(u)**2 - sin(v)**2)  
    z = (a*cosh(u)*cos(v))/(cosh(u)**2 - sin(v)**2)
    coord1 = (x,y,z)
    coord2 = (u,v,w)
    g = metric(coord1,coord2,form,write_to_file)
    diff_form  = [['du*du','du*dv','du*dw'],
                  ['dv*du','dv*dv','dv*dw'],
                  ['dw*du','dv*dw','dw*dw']]
    return g, diff_form

def cylindrical_catenoid_coordinates(form='simplified'):
    from sympy.abc import u, v, w
    from sympy import sin,cos
    x = cos(u) - v*sin(u)
    y = sin(u) + v*cos(u)
    z = v
    coord1 = (x,y,z)
    coord2 = (u,v,w)
    g = metric(coord1,coord2,form)
    g =  g[:2,:2] # 2-dimensional
    diff_form  = [['du*du','du*dv'],
                  ['dv*du','dv*dv']]
    return g, diff_form

def egg_carton_coordinates(form='simplified'):
    from sympy.abc import u, v, w
    from sympy import sin,cos
    x = u
    y = v
    z = sin(u)*cos(v)
    coord1 = (x,y,z)
    coord2 = (u,v,w)
    g = metric(coord1,coord2,form)
    g = g[:2,:2] # 2-dimensional
    diff_form  = [['du*du','du*dv'],
                  ['dv*du','dv*dv']]
    return g, diff_form

def analytical(k_value=0,form="simplified"): # k=0 gives flat space
    from sympy import sin
    from sympy.abc import u,v,k
    ds2 = '(1/(1 - k*u**2))*du**2 + u**2*dv**2 + u**2*sin(v)**2*dw**2'
    g = curve_to_metric(ds2,3)
    g = g.subs(k,k_value)  
    diff_form  = [['du*du','du*dv','du*dw'],
                  ['dv*du','dv*dv','dv*dw'],
                  ['dw*du','dv*dw','dw*dw']]
    return g, diff_form

def kerr_metric(): #in polar coordinates u,v,w, and t
    from sympy import symbols, simplify, cos, sin
    from sympy.abc import G,M,l,u,v,w #,c,J
    # from wikipedia :
    """
    ds2 = '(1 - us*u/p)*c**2*dt**2 - (p/l)*du**2 - p*dv**2 -\
          (u**2 + a**2 + (us*u*a**2/p**2)*sin(v)**2)*sin(v)**2*dw**2\
          + (2*us*u*a*sin(v)**2/p)*c*dt*dw'
    g = curve_to_metric(ds2,dim=4)
    
    us,p,a,l = symbols('us,p,a,l')
    g = g.subs({p:u**2 + a**2*cos(v)})
    g = g.subs({l:u**2 - us*u + a**2})
    g = g.subs({us:2*G*M/c**2})
    g = g.subs({a:J/(M*c)})
    """
    # from Thomas A. Moore (if a=0 ds2 reduces to Schwarzchild solution)
    ds2 = '-(1 - us*u/p)*dt**2 + (p/l)*du**2 + p*dv**2 \
           + (u**2 + a**2 + (us*u*a**2*sin(v)**2/p**2))*sin(v)**2*dw**2\
           - (2*us*u*a*sin(v)**2/p)*dt*dw'
    g = curve_to_metric(ds2,dim=4)
    us,p,a,l = symbols('us,p,a,l')
    g = g.subs({p:u**2 + a**2*cos(v)})
    g = g.subs({l:u**2 - us*u + a**2})
    g = g.subs({us:2*G*M})
    print "Performing simplification on the metric. This may take some time ...."
    g = simplify(g)
    diff_form  = [['du*du','du*dv','du*dw','dt*du'],
                  ['dv*du','dv*dv','dv*dw','dv*dt'],
                  ['dw*du','dv*dw','dw*dw','dw*dt'],
                  ['dt*du','dt*dv','dt*dw','dt*dt']]
    return g, diff_form

def kerr_3D_metric_time_independent(): # unphysical ?
    from sympy import symbols, simplify, cos, sin
    from sympy.abc import G,M,l,u,v,w 
    ds2 = '(p/l)*du**2 + p*dv**2 \
           + (u**2 + a**2 + (us*u*a**2*sin(v)**2/p**2))*sin(v)**2*dw**2'
    g = curve_to_metric(ds2,dim=3)
    us,p,a,l = symbols('us,p,a,l')
    g = g.subs({p:u**2 + a**2*cos(v)})
    g = g.subs({l:u**2 - us*u + a**2})
    g = g.subs({us:2*G*M})
    print "Performing simplification on the metric. This may take some time ...."
    g = simplify(g)
    diff_form  = [['du*du','du*dv','du*dw'],
                  ['dv*du','dv*dv','dv*dw'],
                  ['dw*du','dv*dw','dw*dw']]
    return g, diff_form

def kerr_3D_metric(): # one space component dropped : phi
    from sympy import symbols, simplify, cos, sin
    from sympy.abc import G,M,l,u,v,t
    ds2 = '-(1 - us*u/p)*dw**2 + (p/l)*du**2 + p*dv**2'
    g = curve_to_metric(ds2,dim=3)
    us,p,a,l = symbols('us,p,a,l')
    g = g.subs({p:u**2 + a**2*cos(v)})
    g = g.subs({l:u**2 - us*u + a**2})
    g = g.subs({us:2*G*M})
    print "Performing simplification on the metric. This may take some time ...."
    g = simplify(g)
    diff_form  = [['du*du','du*dv','du*dw'],
                  ['dv*du','dv*dv','dv*dw'],
                  ['dw*du','dv*dw','dw*dw']]
    return g, diff_form

def torus_metric(a=1,c=2,form='simplified'):
    from sympy.abc import u, v, w
    from sympy import sin,cos
    x = (c + a*cos(u))*cos(v)
    y = (c + a*cos(u))*sin(v)
    z = sin(u)
    coord1 = (x,y,z)
    coord2 = (u,v,w)
    g = metric(coord1,coord2,form)
    g = g[:2,:2]
    diff_form  = [['du*du','du*dv'],
                  ['dv*du','dv*dv']]
    return g, diff_form

def flat_sphere():
    ds2  = 'dv**2 + sin(v)**2*dw**2'
    diff_form = [['dv*dv','dv*dw'],['dw*dv','dw*dw']] 
    g = curve_to_metric(ds2,dim=2,diff_=diff_form)
    return g, diff_form

def mobius_strip(form='simplified'):
    from sympy.abc import u, v, w
    from sympy import sin,cos
    x = (1 + cos(u/2)*v/2)*cos(u)
    y = (1 + cos(u/2)*v/2)*sin(u)
    z = sin(u/2)*v/2
    coord1 = (x,y,z)
    coord2 = (u,v,w)
    g = metric(coord1,coord2,form)
    g = g[:2,:2]
    diff_form = [['du*du','du*dv'],['dv*du','dv*dv']] 
    return g, diff_form

if __name__ == "__main__":
    g1,diff_form = toroidal_coordinates()
    print "The toroidal metric" 
    print g1
    print 'with the corresponding differentials'
    print diff_form

    """
    g2,diff_form = cylindrical_coordinates()
    print "\nCylindrical" 
    print g2

    g3, diff_form = spherical_coordinates()
    print "\nSpherical" 
    print g3
    
    g4, diff_form = inverse_prolate_spheroidal_coordinates("usimp",1) 
    print "\nInverse prolate spheroidal coordinates - without simplified form"
    print g4
    """
