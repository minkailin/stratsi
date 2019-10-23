"""
radially local, vertically global basic state of a stratified dusty disk
dust and gas treated as two-fluid system 
horizontal velocities and vertical structure equations are decoupled

assume constant stokes number and particle diffusion coefficient to solve vertical equations exactly,
then use analytic solutions in horizontal equations as non-constant coefficients of the ODEs

horizontal velocity ODEs for vgx (2nd order), vgy (2nd order), vdx, vdy
assume at z=0 we recover the well-known unstratified solutions 

"""
import sys
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
read in parameters from 'eqm_vert.py'
'''
output_file = h5py.File('./eqm_vert.h5', 'r')
rhog0    = output_file['rhog0'][()]
alpha0   = output_file['alpha0'][()]
epsilon0 = output_file['epsilon0'][()]
st0      = output_file['st0'][()]
eta_hat0 = output_file['eta_hat0'][()]
delta0   = output_file['delta0'][()] 
beta     = output_file['beta'][()]
zmin     = output_file['zmin'][()]
zmax     = output_file['zmax'][()]
nz_data  = output_file['nz'][()]

zaxis    = output_file['z'][:]
epsilon  = output_file['epsilon'][:]
rhog     = output_file['rhog'][:]
vdz      = output_file['vdz'][:]
output_file.close()

Delta2   = st0*st0 + (1.0 + epsilon0)**2 

'''
parameters for this calculation
'''
nz = 32#nz_data 

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

'''
assume a constant diffusion coefficient throughout. 
slight inconsistency here because gas visc depends on height, but not particle diffusion (for simplicity)
'''

def visc(rhog_profile):
    #return 0.0
    return alpha0*rhog0/rhog_profile['g'] #this is needed because we assume the dynamical viscosity = nu*rhog = constant throughout 

def eta_hat(z):
    return eta_hat0

'''
vdx, vdy, vgx, vgy at the midplane of an unstratified disk 
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

#problem = de.LBVP(domain, variables=['vgx', 'vgx_prime', 'vgy', 'vgy_prime', 'vdx', 'vdy'], ncc_cutoff=ncc_cutoff)
problem = de.LBVP(domain, variables=['vgx', 'vgy', 'vdx', 'vdy'], ncc_cutoff=ncc_cutoff)
#problem = de.NLBVP(domain, variables=['vgx', 'vgx_prime', 'vgy', 'vgy_prime', 'vdx', 'vdy'], ncc_cutoff=ncc_cutoff)

'''
non-constant coefficients:
equilibrium d/g ratio and dust vert velocity are extracted from data 
radial pressure gradient, viscosity are prescribed 
'''
eps_profile      = domain.new_field(name='eps_profile')
rhog_profile     = domain.new_field(name='rhog_profile')
vdz_profile      = domain.new_field(name='vdz_profile')
eta_profile      = domain.new_field(name='eta_profile')
visc_profile     = domain.new_field(name='visc_profile')

scale            = nz_data/nz 

eps_profile.set_scales(scale)
rhog_profile.set_scales(scale)
vdz_profile.set_scales(scale)

eps_profile['g'] = epsilon
rhog_profile['g']= rhog 
vdz_profile['g'] = vdz

eps_profile.set_scales(1,keep_data=True)
rhog_profile.set_scales(1,keep_data=True)
vdz_profile.set_scales(1,keep_data=True)

z                = domain.grid(0)
#eps_profile['g'] = epsilon_analytic(z)
#rhog_profile['g']= rhog_analytic(z)
#vdz_profile['g'] = vdust_analytic(z)
eta_profile['g'] = eta_hat(z)
visc_profile['g']= visc(rhog_profile)

#print(eps_profile['g'])
#sys.exit()

problem.parameters['eps_profile'] = eps_profile
problem.parameters['vdz_profile'] = vdz_profile
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
'''
full equations
'''
'''
problem.add_equation("visc_profile*dz(vgx_prime) + 2*vgy - eps_profile*(vgx - vdx)/st0 = -2*eta_profile")
problem.add_equation("dz(vgx) - vgx_prime = 0")
problem.add_equation("visc_profile*dz(vgy_prime) - 0.5*vgx - eps_profile*(vgy - vdy)/st0 = 0")
problem.add_equation("dz(vgy) - vgy_prime = 0")
problem.add_equation("vdz_profile*dz(vdx) - 2*vdy + (vdx - vgx)/st0 = 0")
problem.add_equation("vdz_profile*dz(vdy) + 0.5*vdx + (vdy - vgy)/st0 = 0")
'''
'''
equations without gas viscosity
'''
problem.add_equation("2*vgy - eps_profile*(vgx - vdx)/st0 = -2*eta_profile")
problem.add_equation("-0.5*vgx - eps_profile*(vgy - vdy)/st0 = 0")
problem.add_equation("vdz_profile*dz(vdx) - 2*vdy + (vdx - vgx)/st0 = 0")
problem.add_equation("vdz_profile*dz(vdy) + 0.5*vdx + (vdy - vgy)/st0 = 0")

'''
boundary conditions for full problem
'''
'''
problem.add_bc("left(vgx)            = vgx0")
#problem.add_bc("right(vgx)            = 0")
problem.add_bc("left(vgx_prime)      = 0")
problem.add_bc("left(vgy)            = vgy0")
#problem.add_bc("right(vgy)            =-eta_hat_top")
problem.add_bc("left(vgy_prime)      = 0")
problem.add_bc("left(vdx)            = vdx0")
problem.add_bc("left(vdy)            = vdy0")
'''

'''
boundary conditions for inviscid case
'''
problem.add_bc("left(vdx)            = vdx0")
problem.add_bc("left(vdy)            = vdy0")

'''
build problem and solver
'''
solver = problem.build_solver()

'''
initial guess for solving as nonlinear problem
set as unstratified solution but use epsilon(z) and eta_hat(z)
'''
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

Delta2_var = st0*st0 + (1.0 + epsilon_analytic(z))**2
vgx['g'] = -2.0*eta_hat(z)*st0/Delta2_var
vdy['g'] = -(1.0+epsilon_analytic(z))*eta_hat(z)/Delta2_var
vgx['g'] = 2.0*eta_hat(z)*epsilon_analytic(z)*st0/Delta2_var
vgy['g'] = -(1.0 + epsilon_analytic(z) + st0*st0)*eta_hat(z)/Delta2_var
'''

'''
solve equations (linear problem)
'''
solver.solve()

'''
solve as nonlinear problem
'''
'''
pert = solver.perturbations.data
pert.fill(1+tolerance)

iter = 0
start_time = time.time()
while np.sum(np.abs(pert)) > tolerance and np.sum(np.abs(pert)) < 1e6:
    solver.newton_iteration()
    iter += 1
end_time = time.time()
'''

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
plt.xlabel(r'$z/H_g$',fontsize=fontsize)

plt.yticks(fontsize=fontsize,weight='bold')
plt.ylabel(r'$velocities$', fontsize=fontsize)

fname = 'eqm_velocity'
plt.savefig(fname,dpi=150)


#dvdx = domain.new_field()
#vdx.differentiate('z',out=dvdx)
#print(dvdx.interpolate(z=zmin)['g'][0])



#plt.show()

