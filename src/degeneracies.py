import numpy as np

def T1(x_):
    x = x_[0]
    y = x_[1]
    return np.array([[.5*x**2 + 2*x*y + .5*y**2,              -x**2 + y**2],
                     [             -x**2 + y**2, -.5*x**2 - 2*x*y - 5*y**2]])
def T2(x_):
    x = x_[0]
    y = x_[1]
    return np.array([[x**2 - y**2, 2*x*y],
                     [2*x*y, x**2 - y**2]])
def T3(x_):
    x = x_[0]
    y = x_[1]
    return np.array([[x**2 - 3*y**2, -5*x*y + 4*y**2],
                     [-5*x*y + 4*y**2, x**2 - 3*y**2]])
def T4(x_):
    x = x_[0]
    y = x_[1]
    return np.array([[x**4 - .5*x**2*y**2, 2*x**4 - 5*x**3*y - 9*x*y**3],
                     [2*x**4 - 5*x**3*y - 9*x*y**3, x**4 - .5*x**2*y**2]])

def T5(x_):
    x = x_[0]
    y = x_[1]
    return np.array([[-x**2 + y**2, -x**2 - 2*x*y + y**2],
                     [-x**2 - 2*x*y + y**2, -x**2 + y**2]])

def degenerate(T,tol=1e-12):
    """
    Function assumes that the tensor is given with the following form :
    
    If the tensor values are distributed over a 100x100 grid, then for 2D,
    (100,100,2,2) is the shape of the tensor T. Meaning, each grid point 
    contains a 2 dimensional second rank tensor. We assume that all such 
    tensors are symmetric. For a 3D tensor defined over a 16x50x16 grid, 
    T is of the shape (16,50,16,3,3). Here too we assume that all grid 
    tensors are symmetric.
    """
    if T.shape[-2:] == (2,2):
        if len(T.shape[:-2]) == 2:
            deg_points = _find_all_degen_points(T,dim=2,tol=tol)
            return deg_points
    elif T.shape[-2:] == (3,3):
        if len(T.shape[:-2]) == 3:
            deg_points = _find_all_degen_points(T,dim=3,tol=tol)
            return deg_points
    msg =  "Tensor has wrong shape.\n"
    msg += "Tensor shape must be either in 3D\n" 
    msg += "(data x, data y, data z, gridx, gridy, gridz)\n" 
    msg += "or in 2D\n(data x, data y, gridx, gridy)"
    raise IndexError(msg)

def _find_all_degen_points(T,dim,tol):
    if tol>1e-6:
        print "Warning : Tolerance value provided is too large."
    xdata = T.shape[0]
    ydata = T.shape[1]
    if dim == 3:
        zdata = T.shape[2]
    fdp = 0 # counter for the amount of times we find a degenerate point

    class SparseMatrix:
        def __init__(self):
            self.entries = {}

        def __call__(self, tuple, value=0):
            self.entries[tuple] = value

        def value(self, tuple):
            try:
                value = self.entries[tuple]
            except KeyError:
                value = 0
            return value

    deg_points = SparseMatrix()
    if dim == 2:
        """
        Solve following system of equations for a 2D tensor :
            T[i,j,0,0] - T[i,j,1,1] == 0
            T[i,j,0,1] == 0
        """
        found = np.zeros(2)
        for i in range(xdata):
            for j in range(ydata):
                found[0] = np.fabs(T[i,j,0,0] - T[i,j,1,1]) <= tol
                found[1] = np.fabs(T[i,j,0,1]) <= tol

                if np.all(found):
                    deg_points((i,j,k),1)
                    fdp = fdp + 1
    elif dim == 3:
        """
        Solve following system of equations for a 3D tensor :
            T[i,j,k,0,0] - T[i,j,1,1] == 0
            T[i,j,k,1,1] - T[i,j,2,2] == 0
            T[i,j,k,0,1] == 0
            T[i,j,k,0,2] == 0
            T[i,j,k,1,2] == 0
        """
        found = np.zeros(5)
        for i in range(xdata):
            for j in range(ydata):
                for k in range(zdata):
                    found[0] = np.fabs(T[i,j,k,0,0] - T[i,j,k,1,1]) <= tol
                    found[1] = np.fabs(T[i,j,k,1,1] - T[i,j,k,2,2]) <= tol
                    found[2] = np.fabs(T[i,j,k,0,1]) <= tol
                    found[3] = np.fabs(T[i,j,k,0,2]) <= tol
                    found[4] = np.fabs(T[i,j,k,1,2]) <= tol
                 
                    if np.all(found):
                        deg_points((i,j,k),1)
                        fdp = fdp + 1             
                         
    print "Found %d degenerate points in the tensor data." %(fdp)
    return deg_points, fdp
