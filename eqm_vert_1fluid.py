"""
test dedalus ODE solver setup by solving the vertical structure with known analytic solution
assume constant stokes number and diffusion throughout domain 

this version utilizes the one-fluid approximation 
assumes particle stokes number is strictly constant

"""

import numpy as np
#from mpi4py import MPI
import matplotlib.pyplot as plt
from dedalus import public as de
import h5py

import time

import logging
logger = logging.getLogger(__name__)

matplotlib_logger = logging.getLogger('matplotlib')
matplotlib_logger.setLevel(logging.WARNING)

#comm = MPI.COMM_WORLD

'''
physical parameters
'''
rhog0    = 1.0      #midplane gas density, density normalization 
alpha0   = 1.0e-3   #alpha viscosity value, assumed constant
epsilon0 = 0.01      #midplane d/g ratio
st0      = 1.0e-3   #assume a constant stokes number throughout 
eta_hat0 = 0.0      #dimensionless radial pressure gradient, not used here but in eqm_horiz

'''
normalizations 
'''
Omega  = 1.0
Hgas   = 1.0
cs     = Hgas*Omega

'''
assume a constant diffusion coefficient throughout. 
'''
delta0   = alpha0*(1.0 + st0 + 4.0*st0*st0)/(1.0+st0*st0)**2
beta     =(1.0/st0 - (1.0/st0)*np.sqrt(1.0 - 4.0*st0**2))/2.0

'''
are we including the advection vz*vz' term? (atm code only works if it's neglected...)
'''
vz2_term = False 
    
'''
grid parameters
'''
zmin     = 0.0
zmax     = 5.0
nz       = 128

output_file = h5py.File('./eqm_vert_1fluid.h5','w')
output_file['rhog0']    = rhog0
output_file['alpha0']   = alpha0
output_file['epsilon0'] = epsilon0
output_file['st0']      = st0
output_file['eta_hat0'] = eta_hat0
output_file['delta0']   = delta0
output_file['beta']     = beta
output_file['zmin']     = zmin
output_file['zmax']     = zmax
output_file['nz']       = nz

'''
numerical parameters
'''
ncc_cutoff = 1e-12
tolerance  = 1e-8

'''
plotting parameters
'''
fontsize= 24
nlev    = 128
nclev   = 6
cmap    = plt.cm.inferno

'''
analytical equilibria for constant stokes number and diffusion (from two fluid equations)
'''
def epsilon_analytic(z):
    return epsilon0*np.exp(-0.5*beta*z*z/delta0)

def rhog_analytic(z):
    return rhog0*np.exp( (delta0/st0)*(epsilon_analytic(z) - epsilon0) - 0.5*z*z)

def Press_analytic(z):
    return cs*cs*rhog_analytic(z)

def rhog_analytic_dustfree(z):
    return rhog0*np.exp(-0.5*z*z)
    
def vdust_analytic(z):
    return -beta*z*Omega

def vz_analytic(z): #vz = (rhog*vgz + rhod*vdz)/rho_tot; and vgz = 0 in eqm
    eps = epsilon_analytic(z)
    fd  = eps/(1.0+eps)
    return vdust_analytic(z)*fd
    
def stokes(rhog):
    return st0*rhog/rhog #so that it returns an array for output
    
def eta_hat(z):
    #assume constant radial pressure gradient for now (division by z/z to convert to array for output)
    #not actually use in ODE, needed for outputing profile
    return eta_hat0*(z+1.0)/(z+1.0)

'''
domain setup
'''
z_basis = de.Chebyshev('z', nz, interval=(zmin,zmax), dealias=2)
domain = de.Domain([z_basis], grid_dtype=np.float64)#, comm=MPI.COMM_SELF)
  
problem = de.NLBVP(domain, variables=['epsilon', 'ln_P', 'vz'], ncc_cutoff=ncc_cutoff)

problem.parameters['cs']       = cs
problem.parameters['Hgas']     = Hgas
problem.parameters['Omega']    = Omega
problem.parameters['tau_s']    = st0/Omega

problem.parameters['delta0']     = delta0

problem.parameters['eps0']     = epsilon_analytic(zmin)
problem.parameters['ln_P0']    = np.log(Press_analytic(zmin))
problem.parameters['vz0']      = vz_analytic(zmin)

'''
equilibrium equations
'''

problem.add_equation("delta0*cs*Hgas*dz(epsilon) = (1 + epsilon)*vz")
problem.add_equation("dz(ln_P) = (1 + epsilon)*(1 + epsilon)*vz/(cs*cs*tau_s*epsilon)") 
if vz2_term == True:
    problem.add_equation("dz(vz) = -(1 + epsilon)/(tau_s*epsilon) - Omega*Omega*z/vz")
if vz2_term == False:
    problem.add_equation("vz = -epsilon*tau_s*Omega*Omega*z/(1 + epsilon)") 


'''
boundary conditions
'''     
problem.add_bc("left(epsilon) = eps0")
problem.add_bc("left(ln_P)    = ln_P0")
if vz2_term == True:
    problem.add_bc("left(vz)      = vz0")

solver = problem.build_solver()

# Setup initial guess
z            = domain.grid(0, scales=domain.dealias)

epsilon      = solver.state['epsilon']
ln_P         = solver.state['ln_P']
vz           = solver.state['vz']

epsilon.set_scales(domain.dealias)
ln_P.set_scales(domain.dealias)
vz.set_scales(domain.dealias)

epsilon['g'] = epsilon_analytic(z)
ln_P['g']    = np.log(Press_analytic(z))
vz['g']      = vz_analytic(z)


# Iterations
pert = solver.perturbations.data
pert.fill(1+tolerance)
do_plot = True

iter = 0
start_time = time.time()
try:
    while np.sum(np.abs(pert)) > tolerance and np.sum(np.abs(pert)) < 1e6:
        solver.newton_iteration()
        iter += 1
except:
    plt.show()
    raise

end_time = time.time()

if do_plot:

    '''
    plot equilibrium dust-to-gas ratio
    '''
    fig = plt.figure(figsize=(8,4.5))
    ax = fig.add_subplot()
    plt.subplots_adjust(left=0.18, right=0.95, top=0.95, bottom=0.2)

#    plt.ylim(ymin,ymax)
#    plt.xlim(xmin,xmax)

    epsilon_guess = epsilon_analytic(z)
    plt.plot(z, epsilon['g'],linewidth=2, label='numerical solution')
    plt.plot(z, epsilon_guess,linewidth=2,linestyle='dashed', label='initial guess')
            
    plt.rc('font',size=fontsize,weight='bold')

    lines1, labels1 = ax.get_legend_handles_labels()
    legend=ax.legend(lines1, labels1, loc='upper right', frameon=False, ncol=1, fontsize=fontsize/2)
    #  legend.get_frame().set_linewidth(0.0)

    #plt.title(title,weight='bold')

    plt.xticks(fontsize=fontsize,weight='bold')
    plt.xlabel('$z/H_g$',fontsize=fontsize)

    plt.yticks(fontsize=fontsize,weight='bold')
    plt.ylabel(r'$\epsilon$',fontsize=fontsize)

    fname = 'eqm_epsilon_1fluid'
    plt.savefig(fname,dpi=150)

    '''
    plot equilibrium vertical dust velocity 
    '''
    fig = plt.figure(figsize=(8,4.5))
    ax = fig.add_subplot()
    plt.subplots_adjust(left=0.18, right=0.95, top=0.95, bottom=0.2)

#    plt.ylim(ymin,ymax)
#    plt.xlim(xmin,xmax)

    vz_guess = vz_analytic(z)
    
    plt.plot(z, vz['g']*1e3/cs,linewidth=2, label='numerical solution')
    plt.plot(z, vz_guess*1e3/cs,linewidth=2,linestyle='dashed', label='initial guess')
            
    plt.rc('font',size=fontsize,weight='bold')

    lines1, labels1 = ax.get_legend_handles_labels()
    legend=ax.legend(lines1, labels1, loc='upper right', frameon=False, ncol=1, fontsize=fontsize/2)
    #  legend.get_frame().set_linewidth(0.0)

    #plt.title(title,weight='bold')

    plt.xticks(fontsize=fontsize,weight='bold')
    plt.xlabel('$z/H_g$',fontsize=fontsize)

    plt.yticks(fontsize=fontsize,weight='bold')
    plt.ylabel(r'$10^3v_z/c_s$',fontsize=fontsize)

    fname = 'eqm_vz_1fluid'
    plt.savefig(fname,dpi=150)

    '''
    plot equilibrium gas density 
    '''
    fig = plt.figure(figsize=(8,4.5))
    ax = fig.add_subplot()
    plt.subplots_adjust(left=0.18, right=0.95, top=0.95, bottom=0.2)

#    plt.ylim(ymin,ymax)
#    plt.xlim(xmin,xmax)

    rhog = np.exp(ln_P['g'])/cs/cs
    rhog_guess = Press_analytic(z)/cs/cs
    plt.plot(z, rhog,linewidth=2, label='numerical solution')
    plt.plot(z, rhog_guess,linewidth=2,linestyle='dashed', label='initial guess')
    plt.plot(z, rhog_analytic_dustfree(z),linewidth=2, label='pure gas limit')
    
    plt.rc('font',size=fontsize,weight='bold')

    lines1, labels1 = ax.get_legend_handles_labels()

    legend=ax.legend(lines1, labels1, loc='upper right', frameon=False, ncol=1, fontsize=fontsize/2)
    #  legend.get_frame().set_linewidth(0.0)

    #plt.title(title,weight='bold')

    plt.xticks(fontsize=fontsize,weight='bold')
    plt.xlabel('$z/H_g$',fontsize=fontsize)

    plt.yticks(fontsize=fontsize,weight='bold')
    plt.ylabel(r'$\rho_g/\rho_{g0}$',fontsize=fontsize)

    fname = 'eqm_rhog_1fluid'
    plt.savefig(fname,dpi=150)
    
#plt.show()

zaxis = domain.grid(0,scales=1)
epsilon.set_scales(1, keep_data=True)
ln_P.set_scales(1, keep_data=True)
vz.set_scales(1, keep_data=True)

output_file['z']       = zaxis
output_file['epsilon'] = epsilon['g']
output_file['ln_P']    = ln_P['g']
output_file['vz']      = vz['g']
rhog = np.exp(ln_P['g'])/cs/cs
output_file['stokes']  = stokes(rhog)
output_file['eta']     = eta_hat(zaxis)

output_file.close()
