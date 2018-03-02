#import h5py
from dolfin import *
from support.physical_constants import *
from support.momentum_form import *
from support.mass_form import *
from support.length_form import *

parameters['form_compiler']['cpp_optimize'] = True
parameters["form_compiler"]["representation"] = "uflacs"
parameters['form_compiler']['quadrature_degree'] = 4
parameters['allow_extrapolation'] = True

class ForwardIceModel(object):

    def __init__(self, model_inputs, out_dir, checkpoint_file):

        # Model inputs object
        self.model_inputs = model_inputs
        # Mesh
        self.mesh = model_inputs.mesh
        # Model time
        self.t = 0.
        # Physical constants / parameters
        self.constants = pcs


        #### Function spaces
        ########################################################################

        # Define finite element function spaces.  Here we use CG1 for
        # velocity computations, DG0 (aka finite volume) for mass cons,
        # and "Real" (aka constant) elements for the length

        E_cg = self.model_inputs.E_cg
        E_dg = self.model_inputs.E_dg
        E_r = FiniteElement("R", self.mesh.ufl_cell(), 0)
        E_V = MixedElement(E_cg, E_cg, E_cg, E_dg, E_r)

        V_cg = self.model_inputs.V_cg
        V_dg = self.model_inputs.V_dg
        V_r = FunctionSpace(self.mesh, E_r)
        V = FunctionSpace(self.mesh, E_V)

        # For moving data between vector functions and scalar functions
        self.assigner_inv = FunctionAssigner([V_cg, V_cg, V_cg, V_dg, V_r], V)
        self.assigner     = FunctionAssigner(V, [V_cg, V_cg, V_cg, V_dg, V_r])

        self.V_cg = V_cg
        self.V_dg = V_dg
        self.V_r = V_r
        self.V = V


        ### Model unknowns + trial and test functions
        ########################################################################

        # U contains both velocity components, the DG thickness, the CG-projected thickness,
        # and the length
        U = Function(V)
        # Trial Function
        dU = TrialFunction(V)
        # Test Function
        Phi = TestFunction(V)

        # Split vector functions into scalar components
        ubar, udef, H_c, H, L = split(U)
        phibar, phidef, xsi_c, xsi, chi = split(Phi)

        # Values of model variables at previous time step
        un = Function(V_cg)
        u2n = Function(V_cg)
        H0_c = Function(V_cg)
        H0 = Function(V_dg)
        L0 = Function(V_r)

        self.ubar = ubar
        self.udef = udef
        self.H_c = H_c
        self.H = H
        self.L = L
        self.phibar = phibar
        self.phidef = phidef
        self.xsi_c = xsi_c
        self.xsi = xsi
        self.chi = chi
        self.U = U
        self.Phi = Phi
        self.un = un
        self.u2n = u2n
        self.H0_c = H0_c
        self.H0 = H0
        self.L0 = L0
        # Time step
        dt = Constant(1.0)
        self.dt = dt
        # 0 function used as an initial velocity guess if velocity solve fails
        self.zero_guess = Function(V_cg)


        ### Input functions
        ########################################################################

        # Bed elevation
        B = Function(V_cg)
        # Basal traction
        beta2 = Function(V_cg)
        # SMB
        adot = Function(V_cg)

        self.B = B
        self.beta2 = beta2
        self.adot = adot
        # Facet function marking divide and margin boundaries
        self.boundaries = model_inputs.boundaries


        ### Function initialization
        ########################################################################

        # Assign initial ice sheet length from data
        L0.vector()[:] = model_inputs.L_init
        # Initialize initial thickness
        H0.assign(model_inputs.H0)
        #H0.vector()[:] += 1.0
        H0_c.assign(model_inputs.H0_c)
        # Initialize guesses for unknowns
        self.assigner.assign(U, [self.zero_guess, self.zero_guess, H0_c, H0, L0])


        ### Derived expressions
        ########################################################################

        # Ice surface
        S = B + H_c
        # Time derivatives
        dLdt = (L - L0) / dt
        dHdt = (H - H0) / dt
        # Overburden pressure
        P_0 = Constant(self.constants['rho']*self.constants['g'])*H_c
        # Water pressure
        P_w = Constant(self.constants['rho_w']*self.constants['g'])*B
        # Effective pressure
        N = P_0 - P_w

        self.S = S
        self.dLdt = dLdt
        self.dHdt = dHdt
        self.dt = dt
        self.P_0 = P_0
        self.P_w = P_w
        self.N = N


        ### Variational Forms
        ########################################################################

        # Momentum balance residual
        momentum_form = MomentumForm(self)
        R_momentum = momentum_form.R_momentum

        # Continuous thickness residual
        R_thickness = (H_c - H)*xsi_c*dx

        # Mass balance residual
        mass_form = MassForm(self)
        R_mass = mass_form.R_mass

        # Length residual
        length_form = LengthForm(self)
        R_length = length_form.R_length

        # Total residual
        R = R_momentum + R_thickness + R_mass + R_length
        J = derivative(R, U, dU)


        ### Variational solver
        ########################################################################

        # Define variational problem subject to no Dirichlet BCs, but with a
        # thickness bound, plus form compiler parameters for efficiency.
        ffc_options = {"optimize": True}
        problem = NonlinearVariationalProblem(R, U, bcs=[], J=J, form_compiler_parameters = ffc_options)

        # Solver parameters
        self.snes_params = {'nonlinear_solver': 'snes',
                      'snes_solver': {
                      'relative_tolerance' : 1e-13,
                      'absolute_tolerance' : 1e-5,
                       'linear_solver': 'mumps',
                       'maximum_iterations': 75,
                       'report' : False
                       }}

        self.problem = problem


        ### Setup the iterator for replaying a run
        ########################################################################

        # Get the time step from input file
        dt = self.model_inputs.dt
        self.dt.assign(dt)
        # Number of steps
        self.N = self.model_inputs.N
        # Iteration count
        self.i = 0


        ### Output files
        ########################################################################
        #self.out_file = HDF5File(mpi_comm_world(), out_dir + checkpoint_file + ".hdf5", 'w')
        #self.out_file.write(self.H0, "H_init")
        #self.out_file.write(self.get_S(), "S_init")
        #self.out_file.write(self.L0, "L_init")
        #self.out_file.close()


    # Assign input functions from model_inputs
    def update_inputs(self, i, L):
        self.model_inputs.assign_inputs(i, L)
        self.B.assign(self.model_inputs.B)
        self.beta2.assign(self.model_inputs.beta2)
        self.adot.assign(self.model_inputs.adot)


    def step(self):
        if self.i < self.N:
            self.update_inputs(self.i, float(self.L0))

            try:
                print float(self.adot0)
                solver = NonlinearVariationalSolver(self.problem)
                solver.parameters.update(self.snes_params)
                solver.solve()
            except:
                solver = NonlinearVariationalSolver(self.problem)
                solver.parameters.update(self.snes_params)
                solver.parameters['snes_solver']['error_on_nonconvergence'] = False
                self.assigner.assign(self.U, [self.zero_guess, self.zero_guess,self.H0_c, self.H0, self.L0])
                solver.solve()

            # Update previous solutions
            self.assigner_inv.assign([self.un,self.u2n, self.H0_c, self.H0, self.L0], self.U)
            # Print current time, max thickness, and adot parameter
            print self.t, self.H0.vector().max(), float(self.L0)
            # Update time
            self.t += float(self.dt)
            self.i += 1

        return (self.i == self.N)


    # Write out a steady state file
    def write_steady_file(self, output_file_name):
      output_file = HDF5File(mpi_comm_world(), output_file_name + '.hdf5', 'w')

      ### Write variables
      output_file.write(self.mesh, "mesh")
      output_file.write(self.H0, "H0")
      output_file.write(self.H0_c, "H0_c")
      output_file.write(self.L0, "L0")
      output_file.write(self.un, "u0")
      output_file.write(self.u2n, "u20")
      output_file.write(self.boundaries, "boundaries")
      output_file.flush()
      output_file.close()
