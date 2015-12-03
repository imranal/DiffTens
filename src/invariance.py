import sympy
from sympy import diff
import numpy

def msg(case,T,x):
    dim = len(x)
    T = numpy.array(T)
    if case:
        print "\nThe tensor" 
        print T
        if dim==2:
            print "has a degenerate point at (%d,%d)" %(x[0],x[1])
        if dim==3:  
            print "has a degenerate point at (%d,%d,%d)" %(x[0],x[1],x[2])
    else:
        print "\nThe Tensor"
        print T
        if dim==2:
            print "does not have a degenerate point at (%d,%d)" %(x[0],x[1])
        if dim==3:        
            print "does not have a degenerate point at (%d,%d,%d)" %(x[0],x[1],x[2])

def invariant2D(T,coords,info=False):
    """
    Function assumes that the tensor is given in analytical form as a
    sympy matrix. Further it only considers 2D second rank symmetric
    tensors, i.e of the form
                
                T = [T11 T12; T12 T22]
    
    The invariant of the tensor is determined : 
             delta = ad - bc
    where a = d/dx(T11 - T22), d = d/dy(T12), b = d/dy(T11 - T22),
    and c = d/dx(T12). If the invariant is found to be zero, the
    tensor has two equal eigenvalues, i.e the tensor is degenerate.

    This technique is based on PhD thesis of Delmarcelle - see Del94
    """
    x0 , y0 = coords[0], coords[1]
    
    T11 = 0.5*T[0,0]
    T12 = 0.5*T[0,1]
    T22 = 0.5*T[1,1]
    
    x, y = sympy.symbols('x y')

    delta =\
    diff(T11-T22,x).evalf(subs={x:x0, y:y0})*diff(T12,y).evalf(subs={x:x0, y:y0}) 
    -\
    diff(T11-T22,y).evalf(subs={x:x0, y:y0})*diff(T12,x).evalf(subs={x:x0, y:y0})

    eps = 1e-12
    if abs(delta) <= eps :
        if info:
            msg(1,T,coords)
        return 1
    else:
        if info:
            msg(0,T,coords)
        return 0

def invariant3D_discriminant(T,coords,info=True):
    """
    Function assumes that the tensor is given in analytical form as a
    sympy matrix. Further it only considers 3D second rank symmetric
    tensors, i.e of the formd 
                
                T = [T00 T01 T02; 
                     T01 T11 T12;
                     T02 T12 T22] 
    
    The discriminant of the tensor is evaluated. If it equals zero, this implies
    that the tensor has at least two equal eigenvalues, i.e the tensor is 
    degenerate.

    This technique is based on the article by Zheng et. al. - see ZTP06
    """
    x0, y0, z0 = coords[0], coords[1], coords[2]
    x,y,z = sympy.symbols('x y z')
    
    P = T[0,0] + T[1,1] + T[2,2]
    Q = sympy.det(T[:2,:2]) + sympy.det(T[1:,1:]) + T[2,2]*T[0,0] - T[0,2]*T[0,2]
    R = sympy.det(T)
    
    D = Q**2*P**2 - 4*R*P**3 - 4*Q**3 - 4*Q**3 + 18*P*Q*R - 27*R**2

    D_value = D.evalf(subs={x:x0,y:y0,z:z0})
    eps = 1e-12
    if abs(D_value) <= eps :
        if info:
            msg(1,T,coords)
        return 1
    else:
        if info:
            msg(0,T,coords)
        return 0

def invariant3D_constraint_functions(T,coords,info=True):
    """
    This function is another representation of the discriminant :
    see invariant3D_discriminant().

    Function assumes that the tensor is given in analytical form as a
    sympy matrix. Further it only considers 3D second rank symmetric
    tensors, i.e of the formd 
                
                T = [T00 T01 T02; 
                     T01 T11 T12;
                     T02 T12 T22] 
    
    The constaint function of the tensor is evaluated. If it equals zero, 
    this implies that the tensor has at least two equal eigenvalues, i.e 
    the tensor is degenerate.

    This technique is based on the article by Zheng et. al. - see ZTP06
    """
    x0, y0, z0 = coords[0], coords[1], coords[2]
    x,y,z = sympy.symbols('x y z')
    
    fx  = T[0,0]*(T[1,1]**2 - T[2,2]**2) + T[0,0]*(T[0,1]**2 - T[0,2]**2)\
        + T[1,1]*(T[2,2]**2 - T[0,0]**2) + T[1,1]*(T[1,2]**2 - T[0,1]**2)\
        + T[2,2]*(T[0,0]**2 - T[1,1]**2) + T[2,2]*(T[0,2]**2 - T[1,2]**2)
    
    fy1 = T[1,2]*(2*(T[1,2]**2 - T[0,0]**2) - (T[0,2]**2 + T[0,1]**2)\
        + 2*(T[1,1]*T[0,0] + T[2,2]*T[0,0] - T[1,1]*T[2,2]))\
        + T[0,1]*T[0,2]*(2*T[0,0] - T[2,2] - T[1,1])
    
    fy2 = T[0,2]*(2*(T[0,2]**2 - T[1,1]**2) - (T[0,1]**2 + T[1,2]**2)\
        + 2*(T[2,2]*T[1,1] + T[0,0]*T[1,1] - T[2,2]*T[0,0]))\
        + T[1,2]*T[0,1]*(2*T[1,1] - T[0,0] - T[2,2])

    fy3 = T[0,1]*(2*(T[0,1]**2 - T[2,2]**2) - (T[1,2]**2 + T[0,2]**2)\
        + 2*(T[0,0]*T[2,2] + T[1,1]*T[2,2] - T[0,0]*T[1,1]))\
        + T[0,2]*T[1,2]*(2*T[2,2] - T[1,1] - T[0,0])

    fz1 = T[1,2]*(T[0,2]**2 - T[0,1]**2) + T[0,1]*T[0,2]*(T[1,1] - T[2,2])

    fz2 = T[0,2]*(T[0,1]**2 - T[1,2]**2) + T[1,2]*T[0,1]*(T[2,2] - T[0,0])

    fz3 = T[0,1]*(T[1,2]**2 - T[0,2]**2) + T[0,2]*T[1,2]*(T[0,0] - T[1,1])

    D = fx**2 + fy1**2 + fy2**2 + fy3**2 + 15*fz1**2 + 15*fz2**2 + 15*fz3**2
    
    D_value = D.evalf(subs={x:x0,y:y0,z:z0})
    eps = 1e-12
    if abs(D_value) <= eps :
        if info:
            msg(1,T,coords)
        return 1
    else:
        if info:
            msg(0,T,coords)
        return 0        

if __name__ == "__main__":
    x,y,z = sympy.symbols('x y z')
    
    T1 = sympy.Matrix([[  0.5*x**2, -x**2+y**2                  ],
                       [-x**2+y**2, -0.5*x**2 - 2*x*y - 0.5*y**2]])
    x0 = 0
    y0 = 0
    coords = (x0,y0)
    invariant2D(T1,coords,info=True)
        
    x0 = 1
    y0 = 1
    z0 = 1
    coords = (x0,y0,z0)
    T2 = sympy.Matrix([[x**2,x*y,y**2],
                       [x*y,y**2,y*z],
                       [x**2,y*z,z**2]])
    discrim = invariant3D_discriminant(T2,coords,info=True)
    constrain = invariant3D_constraint_functions(T2,coords,info=True)

    if discrim == constrain:
        print "Both functions give same result!"
