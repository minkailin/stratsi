"""
ONE FLUID APPROX

radially local, vertically global basic state of a stratified dusty disk
dust and gas treated as two-fluid system 
horizontal velocities and vertical structure equations are decoupled

required vertical profiles of epsilon, rhogas, stokes, eta, viscosity...etc are all read in from
eqm_vert_1fluid.py's output.

option to ignore quadratic terms in velocities 

if everything is included then we get 
horizontal velocity ODEs for vx and vy

impose pure gas solution at zmax. 
physically this requires zmax to be large enough so that epsilon -> 0 there

assume Omega=1, H=1, so cs=1

"""
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
normalizations. these are subsumed in some places, namely in the ODEs
'''
Omega  = 1.0
Hgas   = 1.0
cs     = Hgas*Omega

'''
process command line arguements
'''
parser = argparse.ArgumentParser()
parser.add_argument("--xlim", "-xl", nargs='*', help="set horizontal axis range")
parser.add_argument("--ylim", "-yl", nargs='*', help="set vertical axis range")
args = parser.parse_args()
if(args.xlim):
    xbounds = np.array(args.xlim).astype(np.float) 

if(args.ylim):
    ybounds = np.array(args.ylim).astype(np.float) 
        
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
ln_P     = output_file['ln_P'][:]
vz       = output_file['vz'][:]
stokes   = output_file['stokes'][:]
eta      = output_file['eta'][:]

output_file.close()

'''
parameters for this calculation
'''
nz = nz_data 
vsq_terms = False #include quadratic terms in velocity? 

'''
numerical parameters
'''
ncc_cutoff = 1e-12
tolerance  = 1e-12

'''
setup grid and problem 
'''
z_basis = de.Chebyshev('z', nz, interval=(zmin,zmax), dealias=2)
domain  = de.Domain([z_basis], np.float64, comm=MPI.COMM_SELF)

problem = de.LBVP(domain, variables=['vx', 'vy'], ncc_cutoff=ncc_cutoff)

'''
non-constant coefficients:
equilibrium d/g ratio and vert velocity are extracted from data 
radial pressure gradient, viscosity are prescribed (but also read from data)
'''
eps_profile      = domain.new_field(name='eps_profile')
vz_profile       = domain.new_field(name='vz_profile')
eta_profile      = domain.new_field(name='eta_profile')

scale            = nz_data/nz 

eps_profile.set_scales(scale)
vz_profile.set_scales(scale)
eta_profile.set_scales(scale)

eps_profile['g']  = epsilon
vz_profile['g']   = vz
eta_profile['g']  = eta

eps_profile.set_scales(1,keep_data=True)
vz_profile.set_scales(1,keep_data=True)
eta_profile.set_scales(1,keep_data=True)

problem.parameters['eps_profile'] = eps_profile
problem.parameters['vz_profile'] = vz_profile
problem.parameters['eta_profile'] = eta_profile

'''
equilibrium equations 
'''
if vsq_terms == True:
    problem.add_equation("vz_profile*dz(vx) - 2.0*vy  = -2*eta_profile/(1 + eps_profile)")
    problem.add_equation("vz_profile*dz(vy) + 0.5*vx  = 0")
if vsq_terms == False:
    problem.add_equation("-2.0*vy  = -2*eta_profile/(1 + eps_profile)")
    problem.add_equation("0.5*vx    = 0")
           
'''
boundary conditions for full problem
use analytic solution at z=infinity (pure gas disk) to set vgx, vgy, vdx, vdy at infinity
'''
    
if vsq_terms == True: #impose velocities at zmax to be that of a pure gas
    problem.add_bc("right(vx)            = 0")
    problem.add_bc("right(vy)            =-right(eta_profile)")

'''
build problem and solver
'''
solver = problem.build_solver()

'''
solve equations (linear problem)
'''
solver.solve()

'''
extract solutions (to fine grid)
'''

z    = domain.grid(0, scales=domain.dealias)
vx  = solver.state['vx']
vy  = solver.state['vy']


vx.set_scales(domain.dealias)
vy.set_scales(domain.dealias)

#velocities are normalized by cs
vx_norm = vx['g']
vy_norm = vy['g']


'''
plot equilibrium velocities 
'''

fontsize= 24
nlev    = 128
nclev   = 6
cmap    = plt.cm.inferno

fig = plt.figure(figsize=(8,4.5))
ax = fig.add_subplot()
plt.subplots_adjust(left=0.18, right=0.95, top=0.95, bottom=0.2)

if(args.xlim):
    xmin  = xbounds[0]
    xmax  = xbounds[1]
else:
    xmin  = zmin
    xmax  = zmax

if(args.ylim):
    ymin = ybounds[0]
    ymax = ybounds[1]
else:
    x1 = np.argmin(np.absolute(z - xmin))
    x2 = np.argmin(np.absolute(z - xmax))    
    arr = np.append(vx_norm[x1:x2],[vy_norm[x1:x2]])
    ymin = np.amin(arr)
    ymax = np.amax(arr)


plt.ylim(ymin,ymax)
plt.xlim(xmin,xmax)

plt.plot(z, vx_norm, linewidth=2, label=r'$v_{x}/c_s$')
plt.plot(z, vy_norm, linewidth=2, label=r'$v_{y}/c_s$')

plt.rc('font',size=fontsize,weight='bold')

lines1, labels1 = ax.get_legend_handles_labels()

legend=ax.legend(lines1, labels1, loc='lower left', frameon=False, ncol=1, fontsize=fontsize/2)
#  legend.get_frame().set_linewidth(0.0)

#plt.title(title,weight='bold')

plt.xticks(fontsize=fontsize,weight='bold')
plt.xlabel(r'$z/H_g$',fontsize=fontsize)

plt.yticks(fontsize=fontsize,weight='bold')
plt.ylabel(r'$velocities$', fontsize=fontsize)

fname = 'eqm_velocity_1fluid'
plt.savefig(fname,dpi=150)


'''
output vertical profiles of horizontal velocites
'''
output_file = h5py.File('./eqm_horiz.h5','w')
zaxis = domain.grid(0,scales=1)
vx.set_scales(1, keep_data=True)
vy.set_scales(1, keep_data=True)

output_file['nz']  = nz
output_file['z']   = zaxis
output_file['vx'] = vx['g']
output_file['vy'] = vy['g']
output_file.close()
