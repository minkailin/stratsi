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
eta_hat  = 0.05     #dimensionless radial pressure gradient 

'''
dust parameters
'''
dg0      = 2.0     #midplane d/g ratio
metal    = 0.03    #metallicity  
stokes   = 0.1    #assume a constant stokes number throughout 

delta    = alpha*(1.0 + stokes + 4.0*stokes*stokes)/(1.0+stokes*stokes)**2
beta     = (1.0/stokes - (1.0/stokes)*np.sqrt(1.0 - 4.0*stokes**2))/2.0

'''
grid parameters
'''
zmin    = 0.0
zmax    = 0.0158
nz_vert = 1024

'''
mode parameters
'''
kx     = 400.0
kx_min = 1e2
kx_max = 1e4
nkx    = 25

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
eigen_trial = 1.719138+18.379114*1j #0.336815 -1j*0.020939 #trial eigenvalue
growth_filter = 100.0 #mode filter, only allow growth rates < growth_filter
osc_filter    = 0.05
tol = 1e-12

zstar = 0.9*zmax 

'''
analytic vertical profiles for d/g, vdz, rhog assuming constant stokes number 
'''
def epsilon(z):
    return dg0*np.exp(-0.5*beta*z*z/delta)

def fz(z):
    return 1.0 - ((z-zstar)/(zmax-zstar))**2.0

def dfz(z):
    return -2.0*(z-zstar)/(zmax-zstar)**2.0

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
    result = -beta*z
    fzarr  = fz(z)
    result[z>zstar] *= fzarr[z>zstar]
    return result

def dvdz(z):
    result = -beta*np.ones(np.size(z))
    dfzarr = fz(z) + z*dfz(z)
    result[z>zstar] *= dfzarr[z>zstar]
    return result

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

def d3epsilon(z):
    eps = epsilon(z)
    deps = depsilon(z)
    d2eps= d2epsilon(z)
    dln_eps  = dln_epsilon(z)
    d2ln_eps = d2ln_epsilon(z)

    return d2eps*dln_eps + deps*d2ln_eps + deps*d2ln_eps

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

def Delta_vx1f(z):
    eps  = epsilon(z)
    result = -2.0*eta_hat*stokes/(1.0+eps)
#    result[z>0.015] = -2.0*eta_hat*stokes 
    return result

def dDelta_vx1f(z):
    eps  = epsilon(z)
    deps = depsilon(z)
    result = 2.0*eta_hat*stokes*deps/(1.0+eps)**2
#    result[z>0.015] = 0.0
    return result

def d2Delta_vx1f(z):
    eps  = epsilon(z)
    deps = depsilon(z)
    d2eps= d2epsilon(z)
    result = 2.0*eta_hat*stokes*(d2eps*(1.0+eps) - 2.0*deps*deps)/(1.0+eps)**3
#    result[z>0.015]=0.0
    return result    

def vx1f(z):
    eps  = epsilon(z)
    deps = depsilon(z)
    result = 2.0*eta_hat*eps*deps*stokes*z/(1.0+eps)**3
    result[z>0.015]=0.0 #this seems to help
    return result

def dvx1f(z):
    eps  = epsilon(z)
    deps = depsilon(z)
    d2eps= d2epsilon(z)
    
    top = (1.0+eps)*(eps*d2eps*z + eps*deps) + deps*deps*z*(1.0-2.0*eps)
    bot = (1.0+eps)**4    

    result = 2.0*eta_hat*stokes*top/bot
    result[z>0.015] = 0.0
    return result

def d2vx1f(z):
    eps  = epsilon(z)
    deps = depsilon(z)
    d2eps= d2epsilon(z)
    d3eps= d3epsilon(z)

    top = eps*(1.0+eps)*(d2eps*z + deps) + deps*deps*z*(1.0-2.0*eps)
    dtop= deps*(1.0+2.0*eps)*(d2eps*z + deps) + eps*(1.0+eps)*(d3eps*z + 2.0*d2eps) + \
          deps*(2.0*d2eps*z + deps)*(1.0 - 2.0*eps) - 2.0*deps**3*z
          
    bot = (1.0+eps)**4
    dbot=4.0*(1.0+eps)**3*deps

    result = 2.0*eta_hat*stokes*(dtop*bot - dbot*top)/bot**2
#    result[z>0.015] = 0.0
    return result

def vy1f(z):
    eps  = epsilon(z)
    result = -eta_hat/(1.0+eps)
#    result[z>0.015] = -eta_hat
    return result
    
def dvy1f(z):
    eps  = epsilon(z)
    deps = depsilon(z)
    result = eta_hat*deps/(1.0+eps)**2
#    result[z>0.015] = 0.0
    return result

def d2vy1f(z):
    eps  = epsilon(z)
    deps = depsilon(z)
    d2eps= d2epsilon(z)
    result = eta_hat*(d2eps*(1.0+eps) - 2.0*deps*deps)/(1.0+eps)**3
#    result[z>0.015] = 0.0
    return result

def vdx_eqm(z):
    eps  = epsilon(z)

    return vx1f(z) + Delta_vx1f(z)/(1.0+eps)

def dvdx_eqm(z):
    eps  = epsilon(z)
    deps = depsilon(z)

    return dvx1f(z) + dDelta_vx1f(z)/(1.0+eps) - Delta_vx1f(z)*deps/(1.0+eps)**2

def d2vdx_eqm(z):
    eps  = epsilon(z)
    deps = depsilon(z)
    d2eps= d2epsilon(z)

    return d2vx1f(z) + d2Delta_vx1f(z)/(1.0+eps) - dDelta_vx1f(z)*deps/(1.0+eps)**2 -\
           dDelta_vx1f(z)*deps/(1.0+eps)**2 - Delta_vx1f(z)*(d2eps*(1.0+eps) - 2.0*deps*deps)/(1.0+eps)**3

def vgx_eqm(z):
      
    return vdx_eqm(z) - Delta_vx1f(z) 

def dvgx_eqm(z):

    return dvdx_eqm(z) - dDelta_vx1f(z)

def d2vgx_eqm(z):

    return d2vdx_eqm(z) - d2Delta_vx1f(z)

if fix_metal == True:
    dg0 = get_dg0_from_metal()
    print("adjust midplane d/g={0:4.2f} to satisfy Z={1:4.2f}".format(dg0, metal))
