import numpy as np
from dolfin import *

### Support Functions
########################################################

class VerticalBasis(object):
    """
    Provides dolfin-like access to vertical derivatives.  Accepts
    nodal values (u), a list of test functions (coef), and their
    vertical derivatives (dcoef)
    """
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
    """
    Integrates a form in the vertical dimension
    """
    def __init__(self,points,weights):
        self.points = points
        self.weights = weights
    def integral_term(self,f,s,w):
        return w*f(s)
    def intz(self,f):
        return sum([self.integral_term(f,s,w) for s,w in zip(self.points,self.weights)])

class MomentumForm(object):
    """
    Set up the variational form for momentum balance equation. Computes vertically
    integrated stresses:
        tau_xx - longitudinal
        tau_xy - lateral drag
        tau_xz - vertical shear
        tau_d  - driving stress
        tau_b  - basal drag
    """

    def __init__(self, model):
        # Load physical constants
        n = model.constants['n']
        rho = model.constants['rho']
        g = model.constants['g']
        A_s = model.constants['A_s']
        mu = model.constants['mu']
        b = model.constants['b']
        m = model.constants['m']
        eps_reg = Constant(1e-5)

        # Continuous thickness
        H_c = model.H_c
        # Bed elevation
        B = model.B
        # Basal traction
        beta2 = model.beta2
        # Ice sheet length
        L = model.L
        # Surface
        S = model.S
        # Effective pressure
        N = model.N
        # Facet normal vector
        nhat = FacetNormal(model.mesh)

        # Sigma-coordinate jacobian terms
        def dsdx(s):
            return 1./H_c*(S.dx(0) - s*H_c.dx(0))

        def dsdz(s):
            return -1./H_c

        # vertical test functions, in this case a constant and a n+1 order polynomial
        coef = [lambda s:1.0, lambda s:1./4.*(5*s**4-1.)]
        dcoef = [lambda s:0.0, lambda s:5*s**3]

        # Make vertical basis from ubar and udef, the depth-average and
        # deformational velocities
        u_ = [model.ubar, model.udef]
        phi_ = [model.phibar, model.phidef]

        u = VerticalBasis(u_, coef, dcoef)
        phi = VerticalBasis(phi_, coef, dcoef)


        ### Below we define the various terms of the FO equations

        # Ice viscosity
        def eta_v(s):
            return Constant(b)/2.*(1./L**2*(u.dx(s,0) + u.ds(s)*dsdx(s))**2 \
                        +0.25*((u.ds(s)*dsdz(s))**2) \
                        + eps_reg)**((1.-n)/(2*n))

        # Longitudinal stress
        def membrane_xx(s):
            return 1./L**2*(phi.dx(s,0) + phi.ds(s)*dsdx(s))*H_c*eta_v(s)*(4*(u.dx(s,0) + u.ds(s)*dsdx(s)))# + 1./L**2*(phi.dx(s,0) + phi.ds(s)*dsdx(s))*H_c*eta_v(s)*(2*u(s)/width*width.dx(0))

        # Vertical shear stress
        def shear_xz(s):
            return dsdz(s)**2*phi.ds(s)*H_c*eta_v(s)*u.ds(s)

        # Driving stress
        def tau_dx(s):
            return 1./L*rho*g*H_c*S.dx(0)*phi(s)

        # Create a vertical integrator using gauss-legendre quadrature
        points = np.array([0.0,0.4688,0.8302,1.0])
        weights = np.array([0.4876/2.,0.4317,0.2768,0.0476])
        vi = VerticalIntegrator(points,weights)

        # Basal Shear stress (linear case)
        tau_b = beta2*N*u(1)

        # Residual of the first order equation
        R_momentum = (- vi.intz(membrane_xx) - vi.intz(shear_xz) - phi(1)*tau_b - vi.intz(tau_dx))*L*dx

        self.R_momentum = R_momentum
