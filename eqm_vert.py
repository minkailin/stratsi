"""
test dedalus ODE solver setup by solving the vertical structure with known analytic solution
assume constant stokes number and diffusion throughout domain 
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
alpha0   = 1e-3  #midplane alpha viscosity value
epsilon0 = 1.0   #midplane d/g ratio
st0      = 1e-3  #assume a constant stokes number throughout 
 
'''
assume a constant diffusion coefficient throughout. 
slight inconsistency here because gas visc depends on height, but not particle diffusion (for simplicity)
'''
delta0   = alpha0*(1.0 + st0 + 4.0*st0*st0)/(1.0+st0*st0)**2
beta     =(1.0/st0 - (1.0/st0)*np.sqrt(1.0 - 4.0*st0**2))/2.0
    
'''
grid parameters
'''
zmin     = 0.0
zmax     = 2.0
nz       = 256

'''
numerical parameters
'''
ncc_cutoff = 1e-13
tolerance  = 1e-8

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

z_basis = de.Chebyshev('z', nz, interval=(zmin,zmax), dealias=2)
domain = de.Domain([z_basis], np.float64, comm=MPI.COMM_SELF)
problem = de.NLBVP(domain, variables=['ln_epsilon', 'ln_rhog', 'chi'], ncc_cutoff=ncc_cutoff)

problem.parameters['rhog0']      = rhog0
problem.parameters['st0']        = st0
problem.parameters['delta0']     = delta0

problem.parameters['ln_epsilon0'] = np.log(epsilon_analytic(zmin))
problem.parameters['ln_rhog0']    = np.log(rhog_analytic(zmin))
problem.parameters['vdust0']      = vdust_analytic(zmin)


problem.add_equation("dz(ln_epsilon) = -sqrt(chi)/delta0")
#problem.add_equation("dz(ln_rhog) = -exp(ln_epsilon + ln_rhog)*sqrt(chi)/(st0*rhog0) - z") #for stokes ~1/rhog
#problem.add_equation("dz(chi) = -2.0*z + 2.0*exp(ln_rhog)*sqrt(chi)/(st0*rhog0)")          #for stoeks ~1/rhog
problem.add_equation("dz(ln_rhog) = -exp(ln_epsilon)*sqrt(chi)/st0 - z")
problem.add_equation("dz(chi) = -2.0*z + 2.0*sqrt(chi)/st0")

problem.add_bc("left(ln_epsilon)   = ln_epsilon0")
problem.add_bc("left(ln_rhog)      = ln_rhog0")
problem.add_bc("left(chi)          = vdust0*vdust0")

solver = problem.build_solver()

# Setup initial guess
z            = domain.grid(0, scales=domain.dealias)
ln_epsilon   = solver.state['ln_epsilon']
ln_rhog      = solver.state['ln_rhog']
chi          = solver.state['chi']

ln_epsilon.set_scales(domain.dealias)
ln_rhog.set_scales(domain.dealias)
chi.set_scales(domain.dealias)

epsilon_guess = epsilon_analytic(z)
rhog_guess    = rhog_analytic(z)   
vdust_guess   = vdust_analytic(z) 

ln_epsilon['g'] = np.log(epsilon_guess)
ln_rhog['g']    = np.log(rhog_guess)
chi['g']        = vdust_guess**2


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

    epsilon = np.exp(ln_epsilon['g'])
    plt.plot(z, epsilon,linewidth=2, label='numerical solution')
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

    fname = 'eqm_epsilon'
    plt.savefig(fname,dpi=150)

    '''
    plot equilibrium vertical dust velocity 
    '''
    fig = plt.figure(figsize=(8,4.5))
    ax = fig.add_subplot()
    plt.subplots_adjust(left=0.18, right=0.95, top=0.95, bottom=0.2)

#    plt.ylim(ymin,ymax)
#    plt.xlim(xmin,xmax)

    vdust = -np.sqrt(chi['g'])
    plt.plot(z, vdust*1e3,linewidth=2, label='numerical solution')
    plt.plot(z, vdust_guess*1e3,linewidth=2,linestyle='dashed', label='initial guess')
            
    plt.rc('font',size=fontsize,weight='bold')

    lines1, labels1 = ax.get_legend_handles_labels()
    legend=ax.legend(lines1, labels1, loc='upper right', frameon=False, ncol=1, fontsize=fontsize/2)
    #  legend.get_frame().set_linewidth(0.0)

    #plt.title(title,weight='bold')

    plt.xticks(fontsize=fontsize,weight='bold')
    plt.xlabel('$z/H_g$',fontsize=fontsize)

    plt.yticks(fontsize=fontsize,weight='bold')
    plt.ylabel(r'$10^3v_{dz}/c_s$',fontsize=fontsize)

    fname = 'eqm_vdust'
    plt.savefig(fname,dpi=150)


    '''
    plot equilibrium gas density 
    '''
    fig = plt.figure(figsize=(8,4.5))
    ax = fig.add_subplot()
    plt.subplots_adjust(left=0.18, right=0.95, top=0.95, bottom=0.2)

#    plt.ylim(ymin,ymax)
#    plt.xlim(xmin,xmax)

    rhog = np.exp(ln_rhog['g'])
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

    fname = 'eqm_rhog'
    plt.savefig(fname,dpi=150)
    
    
#plt.show()

