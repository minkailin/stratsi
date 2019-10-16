'''
test script for learning dedalus. solve

d(log rho)/dz = -z 

solution is rho = rhog*exp(-z^2/2)

'''

import numpy as np
import matplotlib.pyplot as plt
from dedalus import public as de

import time

import logging
logger = logging.getLogger(__name__)

matplotlib_logger = logging.getLogger('matplotlib')
matplotlib_logger.setLevel(logging.WARNING)

rhog0 = 1.0
zmin  = 0.0
zmax  = 5.0
nz    = 128 


'''
analytical equilibria
'''

def rhog_analytic(z):
    return rhog0*np.exp(-0.5*z*z)

#z_basis = de.Chebyshev('z', 128, interval=(0,5))
z_basis = de.Hermite('z', 128, interval=(0,5))

domain = de.Domain([z_basis], np.float64)

problem = de.LBVP(domain, variables=['ln_rhog'])
problem.meta[:]['z']['envelope'] = True

problem.parameters['ln_rhog0']    = 0.0 

problem.add_equation("dz(ln_rhog) = -z")
problem.add_bc("left(ln_rhog)      = ln_rhog0")

solver = problem.build_solver()
solver.solve()

z            = domain.grid(0)
ln_rhog      = solver.state['ln_rhog']

do_plot = True

if do_plot:
    fig, ax = plt.subplots()
    ax2 = ax.twinx()

if do_plot:

    ax.plot(z, np.exp(ln_rhog['g']), label='rhog')
    ax.set_ylabel("rhog")
    ax.plot(z, rhog_analytic(z), label='rhog_analytic',linestyle='dashed')
  
plt.show()

