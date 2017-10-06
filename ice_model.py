#import h5py
from dolfin import *
from support.expressions import *
from support.fenics_optimizations import *
from support.momentum import *
from support.physical_constants import *

class IceModel(object):

    def __init__(self, model_inputs, out_dir, checkpoint_file):

        # Model inputs object
        self.model_inputs = model_inputs
        # Model time
        self.t = 0.


        ### Function spaces
        #######################################################################

        mesh = model_inputs.mesh
        Q = self.model_inputs.Q
        E_Q = self.model_inputs.E_Q
        E_V = MixedElement(E_Q, E_Q, E_Q)
        V   = FunctionSpace(mesh, E_V)
        # Assigners to and from mixed function space
        assigner_inv = FunctionAssigner([Q, Q, Q], V)
        assigner     = FunctionAssigner(V, [Q, Q, Q])


        ### Data / input  fields
        #######################################################################

        # Surface
        S0 = Function(Q)
        # Bed elevation
        B  = Function(Q)
        # Thickness
        H0 = Function(Q)
        # SMB data
        smb  = Function(Q)

        self.S0 = S0
        self.H0 = H0
        self.B = B
        self.smb = smb
        self.load_from_model_inputs()


        ### Set up mixed problem for H and velocity
        #######################################################################

        # Mixed unknown
        U   = Function(V)
        dU  = TrialFunction(V)
        Phi = TestFunction(V)
        # H will be the thickness at current time
        u, u2, H   = split(U)
        # Individual test functions needed for forms
        phi, phi1, xsi = split(Phi)
        # Placeholders needed for assignment
        un  = Function(Q)
        u2n = Function(Q)
        # Zeros, an initial guess for velocity for when solver fails
        zero_sol = Function(Q)
        # A generalization of the Crank-Nicolson method, which is theta = .5
        theta = 0.5
        Hmid = Constant(theta)*H + Constant(1.0 - theta)*H0
        # Define surface elevation
        S = B + Hmid
        width = interpolate(Width(degree = 2),Q)


        ### Momentum conservation form
        #######################################################################

        # This object stores the stresses
        strs = Stresses(U, Hmid, H0, H, width, B, S, Phi)
        # Conservation of momentum form:
        R = -(strs.tau_xx + strs.tau_xz + strs.tau_b + strs.tau_d + strs.tau_xy)*dx


        ### Mass conservation form
        #######################################################################

        h = CellSize(mesh)
        D = h*abs(U[0])/2.
        area = Hmid*width
        dt = Constant(1.0)

        mesh_min = mesh.coordinates().min()
        mesh_max = mesh.coordinates().max()

        # Define boundaries
        ocean = FacetFunctionSizet(mesh,0)
        ds1 = ds(subdomain_data=ocean)

        for f in facets(mesh):
            if near(f.midpoint().x(),mesh_max):
              ocean[f] = 1
            if near(f.midpoint().x(),mesh_min):
              ocean[f] = 2

        # Directly write the form, with SPUG and area correction,
        R += ((H-H0)/dt*xsi - xsi.dx(0)*U[0]*Hmid + D*xsi.dx(0)*Hmid.dx(0) - (smb - U[0]*H/width*width.dx(0))*xsi)*dx\
             + U[0]*area*xsi*ds1(1) - U[0]*area*xsi*ds1(0)


        ### Solver setup
        #######################################################################

        # Bounds
        thklim = pcs['thklim']
        l_thick_bound = project(Constant(thklim),Q)

        u_thick_bound = project(Constant(1e4),Q)
        l_v_bound = project(-10000.0, Q)
        u_v_bound = project(10000.0, Q)
        l_bound = Function(V)
        u_bound = Function(V)
        assigner.assign(l_bound,[l_v_bound]*2+[l_thick_bound])
        assigner.assign(u_bound,[u_v_bound]*2+[u_thick_bound])

        # This should set the velocity at the divide (left) to zero
        dbc0 = DirichletBC(V.sub(0),0,lambda x,o:near(x[0],mesh_min) and o)
        # Set the velocity on the right terminus to zero
        dbc1 = DirichletBC(V.sub(0),0,lambda x,o:near(x[0],mesh_max) and o)
        # overkill?
        #dbc2 = DirichletBC(V.sub(1),0,lambda x,o:near(x[0],mesh_max) and o)
        # set the thickness on the right edge to thklim
        dbc3 = DirichletBC(V.sub(2),thklim,lambda x,o:near(x[0],mesh_max) and o)

        #Define variational solver for the mass-momentum coupled problem
        J = derivative(R, U, dU)
        coupled_problem = NonlinearVariationalProblem(R, U, bcs=[dbc0, dbc1, dbc3], J=J)
        coupled_problem.set_bounds(l_bound, u_bound)


        ### Output files
        #######################################################################
        self.out_file = HDF5File(mpi_comm_world(), out_dir + checkpoint_file + ".hdf5", 'w')
        self.out_file.write(self.H0, "H_init")
        self.out_file.write(self.smb, "smb_init")
        self.out_file.write(self.B, "B")
        self.out_file.write(self.S0, "S_init")
        self.out_file.close()


        ### Set some local vars
        #########################################################################

        self.Q = Q
        self.dt = dt
        self.un = un
        self.u2n = u2n
        self.H0 = H0
        self.U = U
        self.assigner_inv = assigner_inv
        self.assigner = assigner
        self.zero_sol = zero_sol
        self.R = R
        self.U = U
        self.dbc0 = dbc0
        self.dbc1 = dbc1
        self.dbc3 = dbc3
        self.J = J
        self.l_bound = l_bound
        self.u_bound = u_bound
        self.coupled_problem = coupled_problem
        self.mesh = mesh
        self.S = S
        self.H = H


    # Load S, B, H, smb from file
    def load_from_model_inputs(self):
        self.S0.assign(self.model_inputs.S0)
        self.H0.assign(self.model_inputs.H0)
        self.B.assign(self.model_inputs.B)
        self.smb.assign(self.model_inputs.smb)


    # Step the model forward
    def step(self, dt):
        # time0 = time.time()
        #print( "Solving for time: ", t)

        # Accquire the optimizations in fenics_optimizations
        try:
            coupled_solver = NonlinearVariationalSolver(self.coupled_problem)
            set_solver_options(coupled_solver)
            coupled_solver.solve()
        except:
            coupled_solver = NonlinearVariationalSolver(self.coupled_problem)
            set_solver_options(coupled_solver)
            coupled_solver.parameters['snes_solver']['error_on_nonconvergence'] = False
            self.assigner.assign(self.U,[self.zero_sol,self.zero_sol,self.H0])
            coupled_solver.solve()
            coupled_solver.parameters['snes_solver']['error_on_nonconvergence'] = True

        self.assigner_inv.assign([self.un,self.u2n,self.H0],self.U)
        self.t += dt


    def checkpoint(self):
        self.out_file.write(self.H, "H", self.t)
        self.out_file.write(interpolate(self.B + self.H, self.Q), "S", self.t)
