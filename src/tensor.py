from __future__ import division
import sympy as sympy

class Riemann:
    """
    Used for defining a Riemann curvature tensor or Ricci tensor
    for a given metric between cartesian coordinates and another
    orthognal coordinate system.
    """
    def __init__(self, g, dim, sys_title="coordinate_system", user_coord = None, 
                       flat_diff = None):
        """
        Contructor __init__ initializes the object for a given
        metric g (symbolic matrix). The metric must be defined 
        with sympy variables u and v for the orthoganl basis in 
        the Cartesian coordinate system.

        g          : metric defined as nxn Sympy.Matrix object
        sys_titles : descriptive information about the coordinate system
                     besides Cartesian coordinates
        dim        : R^2, R^3, or R^4
        user_coord : User supplies their own set of coordinate symbols
        flat_diff  : Matrix with differentials if g is a flat metric
        """
        from sympy.diffgeom import Manifold, Patch
        self.dim = dim
        self.g = g
        if flat_diff is not None:
            self._set_flat_coordinates(sys_title,flat_diff)
        elif user_coord is None:
            self._set_coordinates(sys_title)
        else:
            self._set_user_coordinates(sys_title,user_coord)          
        self.metric = self._metric_to_twoform(g)

    def _set_flat_coordinates(self,sys_title,flat_diff):
        from sympy.diffgeom import CoordSystem, Manifold, Patch
        manifold = Manifold("M",self.dim)
        patch = Patch("P",manifold)
        flat_diff = sympy.Matrix(flat_diff)
        N = flat_diff.shape[0]
        coords = []
        for i in range(0,N):
            n = str(flat_diff[i,i]).find('*')
            coord_i = str(flat_diff[i,i])[1:n]
            coords.append(coord_i)
        if self.dim==4:
            system = CoordSystem(sys_title, patch, [str(coords[0]),str(coords[1]),\
                                                     str(coords[2]),str(coords[3])])
            u, v, w, t = system.coord_functions()
            self.w = w
            self.t = t

        if self.dim==3:   
            system = CoordSystem(sys_title, patch, [str(coords[0]),str(coords[1]),\
                                                    str(coords[2])])
            u, v, w = system.coord_functions()
            self.w = w
            
        if self.dim==2:
            system = CoordSystem(sys_title, patch, [str(coords[0]),str(coords[1])])
            u, v = system.coord_functions()
            
        self.u, self.v = u, v 
        self.system = system 

    def _set_user_coordinates(self,sys_title,user_coord):
        from sympy.diffgeom import CoordSystem, Manifold, Patch
        manifold = Manifold("M",self.dim)
        patch = Patch("P",manifold)
        if self.dim==4:
            system = CoordSystem(sys_title, patch, [str(user_coord[0]),str(user_coord[1]),\
                                                     str(user_coord[2]),str(user_coord[3])])
            u, v, w, t = system.coord_functions()
            self.w = w
            self.t = t

        if self.dim==3:   
            system = CoordSystem(sys_title, patch, [str(user_coord[0]),str(user_coord[1]), 
                                                     str(user_coord[2])])
            u, v, w = system.coord_functions()
            self.w = w
            
        if self.dim==2:
            system = CoordSystem(sys_title, patch, [str(user_coord[0]),str(user_coord[1])])
            u, v = system.coord_functions()
            
        self.u, self.v = u, v 
        self.system = system     

    def _set_coordinates(self,sys_title):
        from sympy.diffgeom import CoordSystem, Manifold, Patch
        manifold = Manifold("M",self.dim)
        patch = Patch("P",manifold)
        if self.dim==4:
            system = CoordSystem(sys_title, patch, ["u", "v", "w","t"])
            u, v, w, t = system.coord_functions()
            self.w = w
            self.t = t

        if self.dim==3:   
            system = CoordSystem(sys_title, patch, ["u", "v", "w"])
            u, v, w = system.coord_functions()
            self.w = w
            
        if self.dim==2:
            system = CoordSystem(sys_title, patch, ["u", "v"])
            u, v = system.coord_functions()
            
        self.u, self.v = u, v 
        self.system = system
    
    def _metric_to_twoform(self,g):
        dim = self.dim
        system = self.system
        diff_forms = system.base_oneforms()
        u_, v_ = self.u, self.v
        u = u_
        v = v_
        if dim >= 3:
            w_ = self.w
            w = w_
        if dim == 4:
            t_ = self.t
            t = t_

        from sympy import asin, acos, atan, cos, log, ln, exp, cosh, sin, sinh, sqrt, tan, tanh
        import sympy.abc as abc
        self._abc = abc
        self._symbols = ['*','/','(',')',"'",'"']
        self._letters = []
        g_ = sympy.Matrix(dim*[dim*[0]])
        # re-evaluate the metric for (u,v,w,t if 4D) which are Basescalar objects
        for i in range(dim):
            for j in range(dim):
                expr = str(g[i,j])
                self._try_expr(expr) # evaluate expr in a safe environment
                for letter in self._letters:
                    exec('from sympy.abc import %s'%letter)
                g_[i,j] = eval(expr) # this will now work for any variables defined in sympy.abc
                g_[i,j] = g_[i,j].subs(u,u_)
                g_[i,j] = g_[i,j].subs(v,v_)
                if dim >= 3:
                    g_[i,j] = g_[i,j].subs(w,w_)
                if dim == 4:
                    g_[i,j] = g_[i,j].subs(t,t_)
        from sympy.diffgeom import TensorProduct
        metric_diff_form = sum([TensorProduct(di, dj)*g_[i, j]
                               for i, di in enumerate(diff_forms) 
                               for j, dj in enumerate(diff_forms)])    
        return metric_diff_form

    def _try_expr(self,expr):
        """
        This is a help function used initially to evaluate the user-defined metric
        elements as a sympy expression : expr. The purpose of this method is to
        prevent the namespace of the user from being polluted by the command
        'from sympy.abc import *'.

        expr : a string object to be evaluated as a sympy expression
        """
        from sympy import asin, acos, atan, cos, log, ln, exp, cosh, sin, sinh, sqrt, tan, tanh
        letters = self._letters
        abc = self._abc
        try:
            for letter in letters:
                exec('from sympy.abc import %s'%letter)  # re-execute after finding each unknown variable    
            l_ = expr.count('(')
            r_ = expr.count(')')
            if l_ == r_: 
                eval(expr)
            elif l_ < r_:
                eval((r_-l_)*'('+expr)
        except NameError as err:
            msg = str(err)
            pos = msg.find("'")
            letter = msg[pos+1]
            pos = pos +1
            found = False
            symbols = self._symbols
            while (pos+1 < len(msg)) and (not found):
                more = msg[pos+1]
                for symb in symbols:
                    if more==symb or more.isdigit():
                        found = True
                        break
                if found is False:
                    letter = letter+more       
                    pos = pos + 1
            for alphabet in abc.__dict__:
                if letter == alphabet:
                    letters.append(alphabet)
                    self._try_expr(expr[expr.find(alphabet):]) # search for the next unknown variable

    def _tuple_to_list(self,t):
        """
        Recoursively turn a tuple to a list.
        """
        return list(map(self._tuple_to_list, t)) if isinstance(t, (list, tuple)) else t

    def _symplify_expr(self,expr): # this is a costly stage for complex expressions
            """
            Perform simplification of the provided expression.
            Method returns a SymPy expression.
            """
            expr = sympy.trigsimp(expr) 
            expr = sympy.simplify(expr) 
            return expr

    def metric_to_Christoffel_1st(self):
        from sympy.diffgeom import metric_to_Christoffel_1st
        return metric_to_Christoffel_1st(self.metric)

    def metric_to_Christoffel_2nd(self):
        from sympy.diffgeom import metric_to_Christoffel_2nd
        return metric_to_Christoffel_2nd(self.metric)

    def find_Christoffel_tensor(self,form="simplified"):
        """
        Method determines the Riemann-Christoffel tensor 
        for a given metric(which must be in two-form).
        
        form : default value - "simplified"
        If desired, a simplified form is returned. 

        The returned value is a SymPy Matrix.
        """
        from sympy.diffgeom import metric_to_Riemann_components
        metric = self.metric
        R = metric_to_Riemann_components(metric)
        simpR = self._tuple_to_list(R)
        dim = self.dim
        if form=="simplified":
            print 'Performing simplifications on each component....'
            for m in range(dim):
                for i in range(dim):
                    for j in range(dim):
                        for k in range(dim):
                            expr = str(R[m][i][j][k])
                            expr = self._symplify_expr(expr)
                            simpR[m][i][j][k] = expr
        self.Christoffel = sympy.Matrix(simpR)
        return self.Christoffel

    def find_Ricci_tensor(self,form="simplified"):
        """
        Method determines the Ricci curvature tensor for
        a given metric(which must be in two-form).
        
        form : default value - "simplified"
        If desired, a simplified form is returned. 

        The returned value is a SymPy Matrix.
        """
        from sympy.diffgeom import metric_to_Ricci_components
        metric = self.metric
        RR = metric_to_Ricci_components(metric)
        simpRR = self._tuple_to_list(RR)
        dim = self.dim
        if form=="simplified":
            print 'Performing simplifications on each component....'
            for m in range(dim):
                for i in range(dim):
                    expr = str(RR[m][i])
                    expr = self._symplify_expr(expr)
                    simpRR[m][i] = expr
        self.Ricci = sympy.Matrix(simpRR)      
        return self.Ricci

    def find_scalar_curvature(self):
        """
        Method performs scalar contraction on the Ricci tensor.
        """
        try:
            Ricci = self.Ricci
        except AttributeError:
            print "Ricci tensor must be determined first."
            return None
        g = self.g
        g_inv = self.g.inv()
        scalar_curv = sympy.simplify(g_inv*Ricci)
        scalar_curv = sympy.trace(scalar_curv)
        self.scalar_curv = scalar_curv
        return self.scalar_curv
           

if __name__ == "__main__":
    import find_metric
    k = -1
    g,diff_form = find_metric.analytical(k) # k=0 gives flat space
    R = Riemann(g,dim=3,sys_title="analytical")
    print R.metric
    from sympy import srepr
    print srepr(R.system)
    RC = R.find_Christoffel_tensor()
    RR = R.find_Ricci_tensor()
    scalarRR = R.find_scalar_curvature()

    print "\nThe analytical curve element has the following metric for k=%.1f"%k
    print g
    print "\nThe Ricci tensor is given as"
    print RR
    print "\nand the scalar curvature is"
    print scalarRR 
    
    """
    from sympy.abc import r,theta, phi, u,v
    g,diff_form = find_metric.flat_sphere()
    diff = [['dv*dv','dv*dw'],['dw*dv','dw*dw']]
    R = Riemann(g, dim=2, sys_title="flat_sphere",\
                flat_metric = True, flat_diff = diff)
    C = R.metric_to_Christoffel_2nd(R.metric)
    RC = R.find_Christoffel_tensor()
    RR = R.find_Ricci_tensor()
    scalarRR = R.find_scalar_curvature()

    print "\nThe 2D sphere has the following metric"
    print g
    print "\nThe Christoffel tensor is given as"
    for m in range(dim):
        for i in range(dim):
            print RC[m,i]
    print "\nThe Ricci tensor is given as"
    print RR
    print "\nand the scalar curvature is"
    print scalarRR 


    g,diff_form = find_metric.toroidal_coordinates()
    R = Riemann(g,dim=3,sys_title="toroidal")
    RC = R.find_Christoffel_tensor()
    RR = R.find_Ricci_tensor()
    print RC,"\n",RR
    
    g,diff_form = find_metric.spherical_coordinates()
    R = Riemann(g,dim=3,sys_title="spherical")
    RC = R.find_Christoffel_tensor()
    RR = R.find_Ricci_tensor()
    print RC,"\n",RR
    
    g,diff_form = find_metric.cylindrical_coordinates()
    R = Riemann(g=g,dim=3,sys_title="cylindrical")
    RC = R.find_Christoffel_tensor()
    RR = R.find_Ricci_tensor()
    print RC,"\n",RR      
    
    # Warning : This takes very long time (just to find g)!
    g,diff_form = find_metric.inverse_prolate_spheroidal_coordinates()
    R = Riemann(g,dim=3,sys_title="inv_prolate_sphere")
    RC = R.find_Christoffel_tensor()
    RR = R.find_Ricci_tensor()
    print RC,"\n",RR
    """
