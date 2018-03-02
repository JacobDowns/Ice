import numpy as np
from dolfin import *

class LengthForm(object):
    """
    Set up the variational form for length, or more specically, the H(x=1)=0
    boundary condition used to determine length.
    """

    def __init__(self, model):

        # DG thickness
        H = model.H
        # Real test function
        chi = model.chi
        # Mesh coordinates
        mesh_coords = model.mesh.coordinates()[:,0]
        # Spacing between cell center points
        delta_x = mesh_coords[-1] - mesh_coords[-2]
        # Midpoints for last three cells
        px2 = 1. - 0.5 * delta_x
        px1 = 1. - 1.5 * delta_x
        px0 = 1. - 2.5 * delta_x

        """
        To get thickness at the terminus, get a polynomial that interpolates the
        last three cell midpoints and evaluate it at x = 1.
        """

        # These coefficients are used to evaluate the polynomial at the terminus.
        C0 = Constant((1. - px1)*(1. - px2)) / ((px0 - px1)*(px0 - px2))
        C1 = Constant((1. - px0)*(1. - px2)) / ((px1 - px0)*(px1 - px2))
        C2 = Constant((1. - px0)*(1. - px1)) / ((px2 - px0)*(px2 - px1))

        # Indicator functions to pick out the last three thickness dofs
        ind0 = Function(model.V_dg)
        ind0.vector()[model.V_dg.dim()-3] = 1.
        ind1 = Function(model.V_dg)
        ind1.vector()[model.V_dg.dim()-2] = 1.
        ind2 = Function(model.V_dg)
        ind2.vector()[model.V_dg.dim()-1] = 1.


        ### Length residual
        ### ====================================================================

        R_length = (C0*(ind0*H) + C1*(ind1*H) + C2*(ind2*H))*chi*dx
        self.R_length = R_length
