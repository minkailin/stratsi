"""
radially local, vertically global basic state of a stratified dusty disk
dust and gas treated as two-fluid system 
horizontal velocities and vertical structure equations are decoupled

required vertical profiles of epsilon, rhogas, stokes, eta, viscosity...etc are all read in from
eqm_vert.py's output.

the only inputs here are whether or not to artifically ignore gas viscosity and/or quadratic terms in dust velocities 

if everything is included then we get 
horizontal velocity ODEs for vgx (2nd order), vgy (2nd order), vdx (1st order), vdy (1st order)

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
normalizations 
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
#    print(args.xlim)
    #xlim_float = [float(x) for x in args.xlim]
    #xbounds = np.array(xlim_float) 
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
ln_epsilon  = output_file['ln_epsilon'][:]
ln_rhog     = output_file['ln_rhog'][:]
vdz      = output_file['vdz'][:]
stokes   = output_file['stokes'][:]
eta      = output_file['eta'][:]

output_file.close()

Delta2   = st0*st0 + (1.0 + epsilon0)**2 

'''
parameters for this calculation
'''
nz = nz_data 
viscosity  = False #include viscosity in gas when calculating horizontal velocities?
vdsq_terms = True #include quadratic terms in dust velocity? would remove ODE for dust variables

'''
numerical parameters
'''
ncc_cutoff = 1e-12
tolerance  = 1e-12

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
vdx, vdy, vgx, vgy at the midplane of an unstratified disk 
'''

vdx0 = -2.0*eta_hat0*st0/Delta2
vdy0 = -(1.0+epsilon0)*eta_hat0/Delta2
vgx0 = 2.0*eta_hat0*epsilon0*st0/Delta2
vgy0 = -(1.0 + epsilon0 + st0*st0)*eta_hat0/Delta2

#print('vgx, vgy, vdx, vdy', vgx0, vgy0, vdx0, vdy0)

'''
setup grid and problem 
'''
z_basis = de.Chebyshev('z', nz, interval=(zmin,zmax), dealias=2)
domain  = de.Domain([z_basis], np.float64, comm=MPI.COMM_SELF)

if viscosity == True:
    problem = de.LBVP(domain, variables=['vgx', 'vgx_prime', 'vgy', 'vgy_prime', 'vdx', 'vdy'], ncc_cutoff=ncc_cutoff)
if viscosity == False:
    problem = de.LBVP(domain, variables=['vgx', 'vgy', 'vdx', 'vdy'], ncc_cutoff=ncc_cutoff)

'''
non-constant coefficients:
equilibrium d/g ratio and dust vert velocity are extracted from data 
radial pressure gradient, viscosity are prescribed 
'''
eps_profile      = domain.new_field(name='eps_profile')
ln_rhog_profile  = domain.new_field(name='ln_rhog_profile')
dln_rhog_profile = domain.new_field(name='dln_rhog_profile')
vdz_profile      = domain.new_field(name='vdz_profile')
stokes_profile   = domain.new_field(name='stokes_profile')
eta_profile      = domain.new_field(name='eta_profile')

scale            = nz_data/nz 

eps_profile.set_scales(scale)
ln_rhog_profile.set_scales(scale)
dln_rhog_profile.set_scales(scale)
vdz_profile.set_scales(scale)
stokes_profile.set_scales(scale)
eta_profile.set_scales(scale)

eps_profile['g']  = np.exp(ln_epsilon)
ln_rhog_profile['g'] = ln_rhog
ln_rhog_profile.differentiate('z', out=dln_rhog_profile)
vdz_profile['g']  = vdz
stokes_profile['g'] = stokes
eta_profile['g'] = eta


eps_profile.set_scales(1,keep_data=True)
ln_rhog_profile.set_scales(1,keep_data=True)
dln_rhog_profile.set_scales(1,keep_data=True)
vdz_profile.set_scales(1,keep_data=True)
stokes_profile.set_scales(1,keep_data=True)
eta_profile.set_scales(1,keep_data=True)


'''
don't delete the following. can use it to set non-constant parameters exactly if we decide to
(e.g. for analytic equilibrium with constant st0)
'''
#z                = domain.grid(0)
#eta_profile['g'] = eta_hat0
#visc_profile['g']= visc(rhog_profile)

problem.parameters['eps_profile'] = eps_profile
problem.parameters['vdz_profile'] = vdz_profile
problem.parameters['stokes_profile'] = stokes_profile
problem.parameters['eta_profile'] = eta_profile
problem.parameters['dln_rhog_profile']=dln_rhog_profile

problem.parameters['st0']   = st0
problem.parameters['alpha0']= alpha0
problem.parameters['vgx0']  = vgx0
problem.parameters['vgy0']  = vgy0
problem.parameters['vdx0']  = vdx0
problem.parameters['vdy0']  = vdy0

'''
equilibrium equations 
'''
'''
full gas equations
'''
if viscosity ==  True:
    problem.add_equation("alpha0*dz(vgx_prime) + alpha0*dln_rhog_profile*vgx_prime + 2*vgy - eps_profile*(vgx - vdx)/stokes_profile = -2*eta_profile")
    problem.add_equation("dz(vgx) - vgx_prime = 0")
    problem.add_equation("alpha0*dz(vgy_prime) + alpha0*dln_rhog_profile*vgy_prime - 0.5*vgx - eps_profile*(vgy - vdy)/stokes_profile = 0")
    problem.add_equation("dz(vgy) - vgy_prime = 0")
    
'''
gas equations in the inviscid limit
'''
if viscosity == False:
    problem.add_equation("2*vgy - eps_profile*(vgx - vdx)/stokes_profile = -2*eta_profile")
    problem.add_equation("-0.5*vgx - eps_profile*(vgy - vdy)/stokes_profile = 0")

'''
dust equations depending on if we ignore the v*dv/dz terms  
result: since these terms are small, they don't make a practical difference.
''' 
if vdsq_terms == True:
    problem.add_equation("vdz_profile*dz(vdx) - 2.0*vdy + (vdx - vgx)/stokes_profile = 0")
    problem.add_equation("vdz_profile*dz(vdy) + 0.5*vdx + (vdy - vgy)/stokes_profile = 0")
if vdsq_terms == False:
    problem.add_equation("-2.0*vdy + (vdx - vgx)/stokes_profile = 0")
    problem.add_equation("0.5*vdx + (vdy - vgy)/stokes_profile = 0")
           
'''
boundary conditions for full problem
use analytic solution at z=infinity (pure gas disk) to set vgx, vgy, vdx, vdy at infinity
set vgx and vgy to be symmetric about midplane
'''

if viscosity == True: #full 2nd order ODE in gas velocities 
    problem.add_bc("right(vgx)           = 0")
    problem.add_bc("left(vgx_prime)      = 0")
    problem.add_bc("right(vgy)           =-right(eta_profile)")
    problem.add_bc("left(vgy_prime)      = 0")
    
if vdsq_terms == True: #then get 1st order ODE in dust velocities, need to impose BC 
    problem.add_bc("right(vdx)            = -2.0*right(stokes_profile*eta_profile)/(1.0 + right(stokes_profile**2.0))")
    problem.add_bc("right(vdy)            = -right(eta_profile)/(1.0 + right(stokes_profile**2.0))")

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
vgx  = solver.state['vgx']
vgy  = solver.state['vgy']
vdx  = solver.state['vdx']
vdy  = solver.state['vdy']

vgx.set_scales(domain.dealias)
vgy.set_scales(domain.dealias)
vdx.set_scales(domain.dealias)
vdy.set_scales(domain.dealias)


# vgx_norm = vgx['g']/np.abs(vgx0)
# vgy_norm = vgy['g']/np.abs(vgy0)
# vdx_norm = vdx['g']/np.abs(vdx0)
# vdy_norm = vdy['g']/np.abs(vdy0)

vgx_norm = vgx['g']
vgy_norm = vgy['g']
vdx_norm = vdx['g']
vdy_norm = vdy['g']

'''
plot equilibrium velocities 
'''
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
    arr = np.append(vgx_norm[x1:x2],[vgy_norm[x1:x2],vdx_norm[x1:x2],vdy_norm[x1:x2]])
    ymin = np.amin(arr)
    ymax = np.amax(arr)


plt.ylim(ymin,ymax)
plt.xlim(xmin,xmax)

# plt.plot(z, vgx_norm, linewidth=2, label=r'$v_{gx}/|v_{gx0}|$')
# plt.plot(z, vgy_norm, linewidth=2, label=r'$v_{gy}/|v_{gy0}|$')
# plt.plot(z, vdx_norm, linewidth=2, label=r'$v_{dx}/|v_{dx0}|$',linestyle='dashed')
# plt.plot(z, vdy_norm, linewidth=2, label=r'$v_{dy}/|v_{dy0}|$',linestyle='dashed')

plt.plot(z, vgx_norm, linewidth=2, label=r'$v_{gx}/c_s$')
plt.plot(z, vgy_norm, linewidth=2, label=r'$v_{gy}/c_s$')
plt.plot(z, vdx_norm, linewidth=2, label=r'$v_{dx}/c_s$',linestyle='dashed')
plt.plot(z, vdy_norm, linewidth=2, label=r'$v_{dy}/c_s$',linestyle='dashed')



plt.rc('font',size=fontsize,weight='bold')

lines1, labels1 = ax.get_legend_handles_labels()

legend=ax.legend(lines1, labels1, loc='lower left', frameon=False, ncol=1, fontsize=fontsize/2)
#  legend.get_frame().set_linewidth(0.0)

#plt.title(title,weight='bold')

plt.xticks(fontsize=fontsize,weight='bold')
plt.xlabel(r'$z/H_g$',fontsize=fontsize)

plt.yticks(fontsize=fontsize,weight='bold')
plt.ylabel(r'$velocities$', fontsize=fontsize)

# unity = np.zeros(len(z))
# unity[:] = 1.0
# plt.plot(z, unity, linewidth=1, linestyle='dotted',color='black')
# plt.plot(z, -unity, linewidth=1, linestyle='dotted',color='black')

fname = 'eqm_velocity'
plt.savefig(fname,dpi=150)


'''
output vertical profiles of horizontal velocites
'''
output_file = h5py.File('./eqm_horiz.h5','w')
zaxis = domain.grid(0,scales=1)
vgx.set_scales(1, keep_data=True)
vgy.set_scales(1, keep_data=True)
vdx.set_scales(1, keep_data=True)
vdy.set_scales(1, keep_data=True)

output_file['nz']  = nz
output_file['z']   = zaxis
output_file['vgx'] = vgx['g']
output_file['vgy'] = vgy['g']
output_file['vdx'] = vdx['g']
output_file['vdy'] = vdy['g']
output_file.close()

#print(np.abs((vdx.interpolate(z=0)['g'][0]-vdx0)/vdx0))
#print(np.abs((vdy.interpolate(z=0)['g'][0]-vdy0)/vdy0))
