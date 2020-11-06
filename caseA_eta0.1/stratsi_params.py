'''
common parameters and functions for stratified streaming instability 

natural units: Hgas=Omega=1, so cs=1
'''
import sys
import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt
from dedalus import public as de
import h5py
import argparse
import time
from scipy.integrate import quad
from scipy.optimize import broyden1

import logging
logger = logging.getLogger(__name__)

matplotlib_logger = logging.getLogger('matplotlib')
matplotlib_logger.setLevel(logging.WARNING)

comm = MPI.COMM_WORLD

'''
disk parameters
'''
rhog0    = 1.0      #midplane gas density, density normalization 
alpha    = 1e-6     #alpha viscosity value, assumed constant
eta_hat  = 0.1     #dimensionless radial pressure gradient 

'''
dust parameters
'''
dg0      = 2.0     #midplane d/g ratio
metal    = 0.03    #metallicity  
stokes   = 1e-2    #assume a constant stokes number throughout 

delta    = alpha*(1.0 + stokes + 4.0*stokes*stokes)/(1.0+stokes*stokes)**2
beta     = (1.0/stokes - (1.0/stokes)*np.sqrt(1.0 - 4.0*stokes**2))/2.0

'''
grid parameters
'''
zmin    = 0.0
zmax    = 0.05
nz_vert = 1024

'''
mode parameters
'''
kx     = 400.0
kx_min = 1e2
kx_max = 1e4
nkx    = 10

'''
vertical resolution
'''
nz_waves = 384

'''
physics options 
'''
fix_metal     = True
viscosity_eqm = False
viscosity_pert= False
diffusion     = True
backreaction  = True

'''
numerical options
'''
all_solve_dense   = True #solve for all eigenvals for all kx
first_solve_dense = True #use the dense solver for very first eigen calc
Neig = 10 #number of eigenvalues to get for sparse solver
eigen_trial = 3.902597e-1-4.110389e-3*1j #0.336815 -1j*0.020939 #trial eigenvalue
sig_filter = 1e10 #mode filter, only allow |sigma| < sig_filter
tol = 1e-12

'''
analytic vertical profiles for d/g, vdz, rhog assuming constant stokes number 
'''
def epsilon(z):
    return dg0*np.exp(-0.5*beta*z*z/delta)

def rhog(z):
    return rhog0*np.exp( (delta/stokes)*(epsilon(z) - dg0) - 0.5*z*z)

def integrand_rhog(z, dg):
    return np.exp( dg*(delta/stokes)*(np.exp(-0.5*beta*z*z/delta) - 1.0) - 0.5*z*z)

def integrand_rhod(z, dg):
    rg  = integrand_rhog(z, dg)
    eps = dg*np.exp(-0.5*beta*z*z/delta)
    return eps*rg

def sigma_g(dg):    
    I = quad(integrand_rhog, 0.0, np.inf, args=(dg))
    return I[0]

def sigma_d(dg):
    I = quad(integrand_rhod, 0.0, np.inf, args=(dg))    
    return I[0]

def vdz(z):
    return -beta*z

def dvdz(z):
    return -beta

def ln_epsilon(z):
    eps = epsilon(z)
    return np.log(eps)

def dln_epsilon(z):
    return -beta*z/delta

def d2ln_epsilon(z):
    return -beta/delta

def depsilon(z):
    eps = epsilon(z)
    dln_eps = dln_epsilon(z)
    return eps*dln_eps

def d2epsilon(z):
    eps = epsilon(z)    
    deps = depsilon(z)
    dln_eps  = dln_epsilon(z)
    d2ln_eps = d2ln_epsilon(z)
    return deps*dln_eps + eps*d2ln_eps

def dln_rhog(z):
    deps = depsilon(z)
    return (delta/stokes)*deps - z

def d2ln_rhog(z):
    d2eps = d2epsilon(z)
    return (delta/stokes)*d2eps - 1.0 

def dln_rhod(z):
    return dln_rhog(z) + dln_epsilon(z)

def metallicity_error(dg):
    sigg = sigma_g(dg)
    sigd = sigma_d(dg)
    Z = sigd/sigg
    return Z - metal

def get_dg0_from_metal():
    Hd      = np.sqrt(delta/(stokes+delta))
    dgguess = metal/Hd
    #print("dgguess=",dgguess)
    sol     = broyden1(metallicity_error,[dgguess],f_tol=1e-16)
    return sol[0]

def Nz2(z):
    eps  = epsilon(z)
    deps = depsilon(z)
    dlnP = dln_rhog(z) #isothermal gas 

    return dlnP*deps/(1.0+eps)**2

if fix_metal == True:
    dg0 = get_dg0_from_metal()
    print("adjust midplane d/g={0:4.2f} to satisfy Z={1:4.2f}".format(dg0, metal))
