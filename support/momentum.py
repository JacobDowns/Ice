from physical_constants    import *
from numpy import array
from dolfin import Constant,Max,sqrt
########################################################
#################   SUPPORT FUNCTIONS  #################
########################################################
class VerticalBasis(object):
    def __init__(self,u,coef,dcoef):
        self.u = u
        self.coef = coef
        self.dcoef = dcoef

    def __call__(self,s):
        return sum([u*c(s) for u,c in zip(self.u,self.coef)])

    def ds(self,s):
        return sum([u*c(s) for u,c in zip(self.u,self.dcoef)])

    def dx(self,s,x):
        return sum([u.dx(x)*c(s) for u,c in zip(self.u,self.coef)])

class VerticalIntegrator(object):
    def __init__(self,points,weights):
        self.points = points
        self.weights = weights
    def integral_term(self,f,s,w):
        return w*f(s)
    def intz(self,f):
        return sum([self.integral_term(f,s,w) for s,w in zip(self.points,self.weights)])

class Stresses(object):
    """
    The purpose of this class is to return the vertically integrated stresses.
    This is achived by accessing one of:
        tau_xx - longitudinal
        tau_xy - lateral drag
        tau_xz - vertical shear
        tau_d  - driving stress
        tau_b  - basal drag
    """

    def __init__(self, U, Hmid, H0, H, width, B, S, Phi):
        # Load physical constants
        n = pcs['n']
        rho = pcs['rho']
        rho_w = pcs['rho_w']
        g = pcs['g']
        n = pcs['n']
        A_s = pcs['A_s']
        mu = pcs['mu']
        b = pcs['b']
        m = pcs['m']
        thklim = pcs['thklim']
        eps_reg = Constant(1e-5)

        # These functions represent our prior about the vertical velocity profiles
        # Functional forms through the vertical, element 0) sliding, 1) deformation
        coef  = [lambda s:1.0, lambda s:1./4.*(5*s**4-1.)]
        dcoef = [lambda s:0.0, lambda s:5*s**3]           # Derivatives of above forms

        # This is a quadrature rule for vertical integration
        points  = array([0.0,0.4688,0.8302,1.0])
        weights = array([0.4876/2.,0.4317,0.2768,0.0476])

        # This object does the integration
        vi = VerticalIntegrator(points,weights)

        u    = VerticalBasis([U[0]  ,  U[1]],coef,dcoef)
        phi  = VerticalBasis([Phi[0],Phi[1]],coef,dcoef)

        def _dsdx(s):
          return 1./Hmid*(S.dx(0) - s*H0.dx(0))

        def _dsdz(s):
          return -1./Hmid

        def _eta_v(s):
          return b/2.*((u.dx(s,0) + u.ds(s)*_dsdx(s))**2 \
            +0.25*((u.ds(s)*_dsdz(s))**2) + eps_reg)**((1.-n)/(2*n))

        def _tau_d(s):
          return rho*g*Hmid*S.dx(0)*phi(s)

        def _tau_xx(s):
          return (phi.dx(s,0) + phi.ds(s)*_dsdx(s))*Hmid*_eta_v(s)\
          *(4*(u.dx(s,0) + u.ds(s)*_dsdx(s))) + (phi.dx(s,0) \
          + phi.ds(s)*_dsdx(s))*H0*_eta_v(s)*\
          (2*u(s)/width*width.dx(0))

        def _tau_xz(s):
          return _dsdz(s)**2*phi.ds(s)*Hmid*_eta_v(s)*u.ds(s)

        # These last two do not require vertical integration
        # Note the impact this has on the test functions associated with them.
        def _tau_xy():
          return 2.*Hmid*b/width*((n+2)/(width))**(1./n)*(abs(U[0])+1.0)**(1./n - 1)*U[0]

        def _tau_b():
          # Overburden, or hydrostatic pressure of ice
          P_i = rho*g*Hmid
          # Water pressure is either ocean pressure or zero
          P_w = Max(-rho_w*g*B,thklim)
          # Effective pressure is P_i - P_w, rho*g appears in A_s
          N   = Max((P_i - P_w)/(rho*g),thklim)
          normalx = (B.dx(0))/sqrt((B.dx(0)**2 + 1.0))
          normalz = sqrt(1 - normalx**2)
          return mu*A_s*(abs(u(1))+1.0)**(1./n-1)*u(1)*(abs(N)+1.0)**(1./m)*(1-normalx**2)


        # The stresses ready to be accessed
        self.tau_d       = vi.intz(_tau_d)
        self.tau_xx      = vi.intz(_tau_xx)
        self.tau_xz      = vi.intz(_tau_xz)
        self.tau_xy      = Phi[0] * _tau_xy()
        self.tau_b       = phi(1) * _tau_b()
        # Forms of stresses ammenable to plotting (no test functions)
        self.tau_b_plot  = _tau_b()
        self.tau_d_plot  = rho*g*Hmid*S.dx(0)
        self.tau_xx_plot = vi.intz(lambda s: _dsdx(s)*Hmid*_eta_v(s)*\
                                  (4*(u.dx(s,0) + u.ds(s)*_dsdx(s)))\
                                   +_dsdx(s)*H0*_eta_v(s)*\
                                   (2*u(s)/width*width.dx(0)))
        self.tau_xz_plot = vi.intz(lambda s:_dsdz(s)**2*Hmid*_eta_v(s)*u.ds(s))
        self.tau_xy_plot = _tau_xy()
