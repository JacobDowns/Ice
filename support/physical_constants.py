from dolfin import Constant

pcs = {}
# Seconds per year
pcs['spy'] = 60**2*24*365
# Ice density
pcs['rho'] = 917.
# Seawater density
pcs['rho_w'] = 1029.0
# Gravitational acceleration
pcs['g'] = 9.81
# Glen's exponent
pcs['n'] = 3.0
# Sliding law exponent
pcs['m'] = 3.0
# Ice hardness
pcs['b'] = 1e-17**(-1./pcs['n'])
# Basal sliding law constants:
pcs['mu'] = Constant(1.0)
pcs['A_s'] = Constant(pcs['rho']*pcs['g']*315.0/500.)
# Minimum thickness
pcs['thklim'] = 1.0
