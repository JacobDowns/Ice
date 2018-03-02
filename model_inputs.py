from dolfin import *

"""
Model inputs for inverse model. Initial thickness is specified. Bed elevation, basal
traction, and glacier length are specified through time. Surface mass balance is
unknown.
"""

class ModelInputs(object):

    def __init__(self, input_file_name, k = 1.0):
        # Rate of change of length (m / a)
        self.dLdt = k
        # Load the mesh for the model
        self.mesh = Mesh()
        self.input_file  = HDF5File(self.mesh.mpi_comm(), input_file_name, "r")
        self.input_file.read(self.mesh, "/mesh", False)


        # Create function space for input data
        self.E_cg = FiniteElement("CG", self.mesh.ufl_cell(), 1)
        self.E_dg = FiniteElement("DG", self.mesh.ufl_cell(), 0)
        self.E_r = FiniteElement("R",  self.mesh.ufl_cell(), 0)
        self.V_cg = FunctionSpace(self.mesh, self.E_cg)
        self.V_dg = FunctionSpace(self.mesh, self.E_dg)
        self.V_r = FunctionSpace(self.mesh, self.E_r)

        # Bed elevation
        class B(Expression):
            def __init__(self, L_initial, degree=1):
                self.L = L_initial
                self.degree=degree

            def eval(self, values, x):
                values[0] = 0.0

        # Basal traction
        class Beta2(Expression):
            def __init__(self, L_initial, degree=1):
                self.L = L_initial
                self.degree=degree

            def eval(self,values,x):
                values[0] = 1e-3

        # Functions for storing inputs
        self.H0 = Function(self.V_dg)
        self.H0_c = Function(self.V_cg)
        self.B = Function(self.V_cg)
        self.beta2 = Function(self.V_cg)
        self.L0 = Function(self.V_r)

        # Load initial thickness and length from a file
        self.input_file.read(self.H0, "/H0")
        self.input_file.read(self.H0_c, "/H0_c")
        self.input_file.read(self.L0, "/L0")
        self.L_init = float(self.L0)

        # Input expressions
        self.B_exp = B(self.L_init, degree = 1)
        self.beta2_exp = Beta2(self.L_init, degree = 1)

        # Create boundary facet function
        self.boundaries = FacetFunctionSizet(self.mesh, 0)

        for f in facets(self.mesh):
            if near(f.midpoint().x(), 1):
                # Terminus
               self.boundaries[f] = 1
            if near(f.midpoint().x(), 0):
               # Divide
               self.boundaries[f] = 2

        # Set function values
        self.update_L(self.L_init)


    # Update time to determine inputs
    def update(self, t):
        self.dLdt = (2.0 / 100.0) * self.t
        self.L = self.L_init +  (1. / 100.0) * self.t**2

        self.B_exp.L = self.L
        self.beta2_exp.L = self.L

        self.B.interpolate(self.B_exp)
        self.beta2.interpolate(self.beta2_exp)
