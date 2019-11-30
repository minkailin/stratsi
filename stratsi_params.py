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
eta_hat  = 0.05     #dimensionless radial pressure gradient 

'''
dust parameters
'''
dg0      = 3.0      #midplane d/g ratio
stokes   = 0.01    #assume a constant stokes number throughout 

delta    = alpha*(1.0 + stokes + 4.0*stokes*stokes)/(1.0+stokes*stokes)**2
beta     = (1.0/stokes - (1.0/stokes)*np.sqrt(1.0 - 4.0*stokes**2))/2.0

'''
grid parameters
'''
zmin    = 0.0
zmax    = 0.05
nz_vert = 256 

'''
mode parameters
'''
kx       = 400.0
nz_waves = 64
    
'''
physics options 
'''
viscosity_eqm = False
viscosity_pert= False
diffusion     = True
backreaction  = True

'''
analytic vertical profiles for d/g, vdz, rhog assuming constant stokes number 
'''
def epsilon(z):
    return dg0*np.exp(-0.5*beta*z*z/delta)

def rhog(z):
    return rhog0*np.exp( (delta/stokes)*(epsilon(z) - dg0) - 0.5*z*z)

def epsilon_Z(z, dgmid):
    return dgmid*np.exp(-0.5*beta*z*z/delta)

def rhog_Z(z, dgmid):
    return np.exp( (delta/stokes)*(epsilon_Z(z,dgmid) - dgmid) - 0.5*z*z)

def rhod_Z(z, dgmid):
    return epsilon(z, dgmid)*rhog_Z(z, dgmid)

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

def metallicity(dgmid):
    
    
