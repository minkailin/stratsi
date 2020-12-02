import numpy as np
import scipy.optimize 


delta = 1e-6
stokes= 0.01
hd    = np.sqrt(delta/(stokes+delta))

metal = 0.03
dgmid = metal/hd

eta_hat=0.05


def func(x):
   return  (x - 1)/(x + 1) - dgmid*np.exp(-x/2.0)

def epsilon(z):
    return dgmid*np.exp(-stokes*z*z/(2.0*delta))

def depsilon(z):
    return -stokes*z*epsilon(z)/delta

def dlnepsilon(z):
    return -stokes*z/delta

def vy(z):
    return -eta_hat/(1 + epsilon(z))

def dvy(z):
    return (eta_hat/(1.0 + epsilon(z))**2)*depsilon(z)

def d2vy(z):
    eps = epsilon(z)
    deps= depsilon(z)
    return -(stokes/delta)*eta_hat*((eps + z*deps)*(1.0 + eps) - 2.0*deps*z*eps)/(1.0+eps)**3
    
chi           = scipy.optimize.broyden1(func, 1.0)
z_maxvshear   = np.sqrt(chi*delta/stokes)
maxvshear     = np.abs(dvy(z_maxvshear))

#z = scipy.optimize.broyden1(d2vy, 0.01)

print('max vert shear at z=', z_maxvshear)
print('max vert shear =', maxvshear)
