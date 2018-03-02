import numpy as np
from dolfin import *

class MassForm(object):
    """
    Set up the variational form for the mass balance equation.
    """

    def __init__(self, model):

        # DG thickness
        H = model.H
        # Rate of change of H
        dHdt = model.dHdt
        # Ice sheet length
        L = model.L
        # Rate of change of L
        dLdt = model.dLdt
        # Surface mass balance
        adot = model.adot
        # Velocity
        ubar = model.ubar
        # DG test function
        xsi = model.xsi
        # Boundary measure
        ds1 = dolfin.ds(subdomain_data = model.boundaries)

        # Spatial coordinate
        x_spatial = SpatialCoordinate(model.mesh)
<<<<<<< HEAD
=======
        # SMB form
        adot_prime = (1.0 - adot * x_spatial[0])
        self.adot_prime = adot_prime
>>>>>>> a6039bd0e8d80499762f59d6a75d36d9d9349905
        # Grid velocity
        v = dLdt*x_spatial[0]
        # Flux velocity
        q_vel = ubar - v
        # Flux
        q_flux = q_vel * H
        # Inter element flux (upwind)
        uH = avg(q_vel)*avg(H) + 0.5*abs(avg(q_vel))*jump(H)


        ### Mass balance residual
        ########################################################################
<<<<<<< HEAD
        R_mass = (L*dHdt*xsi + H*dLdt*xsi - L*adot*xsi)*dx
=======
        R_mass = (L*dHdt*xsi + H*dLdt*xsi - L*adot_prime*xsi)*dx
>>>>>>> a6039bd0e8d80499762f59d6a75d36d9d9349905
        R_mass += uH*jump(xsi)*dS
        R_mass += (q_vel / sqrt(q_vel**2 + Constant(1e-10))) * q_flux*xsi*ds1(1)

        self.R_mass = R_mass
