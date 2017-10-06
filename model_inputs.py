from dolfin import *

class ModelInputs(object):

    def __init__(self, input_file_name, load_from_file = True, load_callback = None):

        self.input_file_name = input_file_name
        self.load_from_file = load_from_file
        self.load_callback = load_callback

        # Load the mesh for the model
        self.mesh = Mesh()
        self.input_file  = HDF5File(self.mesh.mpi_comm(), input_file_name, "r")
        self.input_file.read(self.mesh, "/mesh", False)
        self.load_from_file = load_from_file

        # Create function space for input data
        self.E_Q = FiniteElement("CG", self.mesh.ufl_cell(), 1)
        self.Q = FunctionSpace(self.mesh, self.E_Q)

        # Load inputs
        self.load_inputs()



    # Load all model inputs
    def load_inputs(self):
        if self.load_from_file:
            # Surface
            self.S0 = Function(self.Q)
            # Bed elevation
            self.B  = Function(self.Q)
            # Thickness
            self.H0 = Function(self.Q)
            # SMB data
            self.smb  = Function(self.Q)

            self.input_file.read(self.S0.vector(), "/surface", True)
            self.input_file.read(self.B.vector(), "/bed", True)
            self.input_file.read(self.smb.vector(), "/smb", True)
            self.H0.assign(self.S0 - self.B)

            print self.smb.vector().array()
        else:
            self.load_callback(self)

        self.input_file.close()
