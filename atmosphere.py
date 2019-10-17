"""
radially local, vertically global basic state of a stratified dusty disk
dust and gas treated as two-fluid system 
horizontal velocities and vertical structure equations are decoupled

assume constant stokes number and particle diffusion coefficient to solve vertical equations exactly,
then use analytic solutions in horizontal equations as non-constant coefficients of the ODEs

horizontal velocity ODEs for vgx (2nd order), vgy (2nd order), vdx, vdy
assume at z=0 we recover the well-known unstratified solutions 

"""

import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt
from dedalus import public as de
import h5py

import time

import logging
logger = logging.getLogger(__name__)

matplotlib_logger = logging.getLogger('matplotlib')
matplotlib_logger.setLevel(logging.WARNING)

comm = MPI.COMM_WORLD

'''
physical parameters
'''
rhog0    = 1.0   #midplane gas density, density normalization 
eta_hat0 = 0.05  #midplane radial pressure gradient 
alpha0   = 1e-3  #midplane alpha viscosity value
epsilon0 = 1.0   #midplane d/g ratio
st0      = 1e-3  #assume a constant stokes number throughout 
 
'''
assume a constant diffusion coefficient throughout. 
slight inconsistency here because gas visc depends on height, but not particle diffusion (for simplicity)
'''
delta0   = alpha0*(1.0 + st0 + 4.0*st0*st0)/(1.0+st0*st0)**2
Delta2   = st0*st0 + (1.0 + epsilon0)**2 
beta     =(1.0/st0 - (1.0/st0)*np.sqrt(1.0 - 4.0*st0**2))/2.0
    
'''
grid parameters
'''
zmin     = 0.0
zmax     = 0.5
nz       = 256

'''
numerical parameters
'''
ncc_cutoff = 1e-13
tolerance  = 1e-13

'''
plotting parameters
'''
fontsize= 24
nlev    = 128
nclev   = 6
cmap    = plt.cm.inferno

'''
analytical equilibria for constant stokes number and diffusion 
'''
def epsilon_analytic(z):
    return epsilon0*np.exp(-0.5*beta*z*z/delta0)

def rhog_analytic(z):
    return rhog0*np.exp( (delta0/st0)*(epsilon_analytic(z) - epsilon0) - 0.5*z*z)

def rhog_analytic_dustfree(z):
    return rhog0*np.exp(-0.5*z*z)
    
def vdust_analytic(z):
    return -beta*z

def visc(z):
    #return 1.0e-6
    return alpha0*rhog0/rhog_analytic(z) #this is needed because we assume the dynamical viscosity = nu*rhog = constant throughout 

def eta_hat(z):
    return eta_hat0

'''
specify vdx, vdy, vgx, vgy at the midplane to be that for the unstratified disk 
'''

vdx0 = -2.0*eta_hat0*st0/Delta2
vdy0 = -(1.0+epsilon0)*eta_hat0/Delta2
vgx0 = 2.0*eta_hat0*epsilon0*st0/Delta2
vgy0 = -(1.0 + epsilon0 + st0*st0)*eta_hat0/Delta2

print('vgx, vgy, vdx, vdy', vgx0, vgy0, vdx0, vdy0)


'''
setup grid and problem 
'''
z_basis = de.Chebyshev('z', nz, interval=(zmin,zmax), dealias=2)

domain  = de.Domain([z_basis], np.float64, comm=MPI.COMM_SELF)
problem = de.LBVP(domain, variables=['vgx', 'vgx_prime', 'vgy', 'vgy_prime', 'vdx', 'vdy'], ncc_cutoff=ncc_cutoff)

'''
non-constant coefficients:
equilibrium d/g ratio, dust vert velocity, radial pressure gradient, viscosity  
'''
z                = domain.grid(0)

eps_profile      = domain.new_field(name='eps_profile')
eps_profile['g'] = epsilon_analytic(z)

vzd_profile      = domain.new_field(name='vzd_profile')
vzd_profile['g'] = vdust_analytic(z)

eta_profile      = domain.new_field(name='eta_profile')
eta_profile['g'] = eta_hat(z)

visc_profile     = domain.new_field(name='visc_profile')
visc_profile['g']= visc(z)

problem.parameters['eps_profile'] = eps_profile
problem.parameters['vzd_profile'] = vzd_profile
problem.parameters['eta_profile'] = eta_profile
problem.parameters['visc_profile']= visc_profile

problem.parameters['st0']   = st0
problem.parameters['eta_hat_top'] = eta_hat(zmax)
problem.parameters['vgx0']  = vgx0
problem.parameters['vgy0']  = vgy0
problem.parameters['vdx0']  = vdx0
problem.parameters['vdy0']  = vdy0

'''
equilibrium equations 
'''

problem.add_equation("visc_profile*dz(vgx_prime) + 2*vgy - eps_profile*(vgx - vdx)/st0 = -2*eta_profile")
problem.add_equation("dz(vgx) - vgx_prime = 0")
problem.add_equation("visc_profile*dz(vgy_prime) - 0.5*vgx - eps_profile*(vgy - vdy)/st0 = 0")
problem.add_equation("dz(vgy) - vgy_prime = 0")
problem.add_equation("vzd_profile*dz(vdx) - 2*vdy + (vdx - vgx)/st0 = 0")
problem.add_equation("vzd_profile*dz(vdy) + 0.5*vdx + (vdy - vgy)/st0 = 0")

#problem.add_equation("visc_profile*dz(vgx_prime)*st0 + 2*vgy*st0 - eps_profile*(vgx - vdx) = -2*eta_profile*st0")
#problem.add_equation("dz(vgx) - vgx_prime = 0")
#problem.add_equation("visc_profile*dz(vgy_prime)*st0 - 0.5*vgx*st0 - eps_profile*(vgy - vdy) = 0")
#problem.add_equation("dz(vgy) - vgy_prime = 0")
#problem.add_equation("vzd_profile*dz(vdx)*st0 - 2*vdy*st0 + (vdx - vgx) = 0")
#problem.add_equation("vzd_profile*dz(vdy)*st0 + 0.5*vdx*st0 + (vdy - vgy) = 0")

'''
initial conditions 
'''
#problem.meta[:]['z']['dirichlet'] = True
problem.add_bc("left(vgx)            = vgx0")
#problem.add_bc("right(vgx)            = 0")
problem.add_bc("left(vgx_prime)      = 0")
problem.add_bc("left(vgy)            = vgy0")
#problem.add_bc("right(vgy)            =-eta_hat_top")
problem.add_bc("left(vgy_prime)      = 0")
problem.add_bc("left(vdx)            = vdx0")
problem.add_bc("left(vdy)            = vdy0")

'''
solve equations 
'''
solver = problem.build_solver()
solver.solve()

'''
extract solutions (to fine grid)
'''

z    = domain.grid(0, scales=domain.dealias)
vgx  = solver.state['vgx']
vgy  = solver.state['vgy']
vdx  = solver.state['vdx']
vdy  = solver.state['vdy']

vgx.set_scales(domain.dealias)
vgy.set_scales(domain.dealias)
vdx.set_scales(domain.dealias)
vdy.set_scales(domain.dealias)

'''
plot equilibrium velocities 
'''
fig = plt.figure(figsize=(8,4.5))
ax = fig.add_subplot()
plt.subplots_adjust(left=0.18, right=0.95, top=0.95, bottom=0.2)

xbounds = np.array([0,0.1])

#if(xbounds == None):
#    xmin  = zmin
#    xmax  = zmax
#else:
xmin  = xbounds[0]
xmax  = xbounds[1]

x1 = np.argmin(np.absolute(z - xmin))
x2 = np.argmin(np.absolute(z - xmax))

#if(plotrange == None):
ymin = np.amin(vdx['g'][x1:x2])
ymax = np.amax(vdx['g'][x1:x2])
#else:
#    ymin = plotrange[0]
#    ymax = plotrange[1]

#plt.ylim(ymin,ymax)
#plt.xlim(xmin,xmax)

#plt.plot(z, vgx['g'], linewidth=2, label=r'$v_{gx}$')
#plt.plot(z, vgy['g'], linewidth=2, label=r'$v_{gy}$')
plt.plot(z, vdx['g'], linewidth=2, label=r'$v_{dx}$')
#plt.plot(z, vdy['g'], linewidth=2, label=r'$v_{dy}$')

plt.rc('font',size=fontsize,weight='bold')

lines1, labels1 = ax.get_legend_handles_labels()

legend=ax.legend(lines1, labels1, loc='upper right', frameon=False, ncol=1, fontsize=fontsize/2)
#  legend.get_frame().set_linewidth(0.0)

#plt.title(title,weight='bold')

plt.xticks(fontsize=fontsize,weight='bold')
plt.xlabel('$z/H_g$',fontsize=fontsize)

plt.yticks(fontsize=fontsize,weight='bold')
plt.ylabel(r'$velocities$', fontsize=fontsize)

fname = 'eqm_velocity'
plt.savefig(fname,dpi=150)

dvdx = domain.new_field()
vdx.differentiate('z',out=dvdx)
print(dvdx.interpolate(z=0)['g'][0])
                    
#plt.show()

