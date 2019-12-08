"""

stratified linear analysis of the streaming instability

ONE FLUID APPROX

"""
import sys
import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt
from dedalus import public as de
import time
from scipy.integrate import quad
from scipy.optimize import broyden1

import logging
logger = logging.getLogger(__name__)

matplotlib_logger = logging.getLogger('matplotlib')
matplotlib_logger.setLevel(logging.WARNING)

comm = MPI.COMM_WORLD

logger.info("stratified streaming instability")
from eigenproblem import Eigenproblem #added by MKL
import h5py

'''
normalizations 
'''

Omega  = 1.0
Hgas   = 1.0
cs     = Hgas*Omega

'''
parameters for eigenvalue problem
kx normalized by 1/Hgas
'''

kx     = 400.0
kx_min = 400
kx_max = 4000
nkx    = 20

'''
physics options 
can choose to include/exclude particle diffusion, 
'''
fix_metal    = True
diffusion    = True
tstop        = True

'''
problem parameters
'''
alpha0    = 1e-6
st0       = 1e-2
dg0       = 2.0
metal     = 0.02
eta_hat   = 0.05

zmin      = 0
zmax      = 0.08
nz_waves  = 256

delta0   = alpha0*(1.0 + st0 + 4.0*st0*st0)/(1.0+st0*st0)**2

'''
dimensional parameters 
'''
tau_s = st0/Omega
Diff  = delta0*cs*Hgas
visc  = alpha0*cs*Hgas

'''
output control
'''
nz_out   = 1024
out_scale= nz_out/nz_waves

'''
functions to calc eqm profiles
these analytic profiles *neglecting* quadratic terms (so not exact)
'''

def epsilon_eqm(z):
    return dg0*np.exp(-0.5*tau_s*Omega*Omega*z*z/Diff)

def dln_epsilon_eqm(z):
    return -tau_s*Omega*Omega*z/Diff

def d2ln_epsilon_eqm(z):
    return -tau_s*Omega*Omega/Diff

def depsilon_eqm(z):
    eps = epsilon_eqm(z)
    return eps*dln_epsilon_eqm(z)

def d2epsilon_eqm(z):
    eps     = epsilon_eqm(z)
    deps    = depsilon_eqm(z)
    dln_eps = dln_epsilon_eqm(z)
    return deps*dln_eps + eps*d2ln_epsilon_eqm(z)

def dln_P_eqm(z):
    eps = epsilon_eqm(z)
    return -z*(1.0 + eps)/Hgas/Hgas

def d2ln_P_eqm(z):
    eps = epsilon_eqm(z)
    return -(1.0 + eps*(1.0 - tau_s*Omega*Omega*z*z/Diff) )/Hgas/Hgas

def dln_rho_eqm(z):
    dlnP = dln_P_eqm(z)
    eps = epsilon_eqm(z)
    deps = depsilon_eqm(z)
    return dlnP + deps/(1.0+eps)

def dln_rhod_eqm(z):
    dln_eps = dln_epsilon_eqm(z)
    dln_P   = dln_P_eqm(z)
    return dln_eps + dln_P

def K_over_P_eqm(z):
    eps = epsilon_eqm(z)
    return cs*cs*tau_s*eps/(1.0+eps)/(1.0+eps)

def dln_K_eqm(z):
    eps = epsilon_eqm(z)
    dln_eps = dln_epsilon_eqm(z)
    dln_P   = dln_P_eqm(z)
    return dln_eps*(1.0-eps)/(1.0+eps) + dln_P

def vz_eqm(z):
    eps = epsilon_eqm(z)
    fd = eps/(1.0+eps)
    return -fd*tau_s*Omega*Omega*z

def dvz_eqm(z):
    eps     = epsilon_eqm(z)
    dln_eps = dln_epsilon_eqm(z) 
    fd = eps/(1.0+eps)
    return -tau_s*Omega*Omega*fd*(1.0+z*dln_eps/(1.0+eps))

def vy_eqm(z):
    eps = epsilon_eqm(z)
    fg  = 1.0/(1.0+eps)
    return -eta_hat*cs*fg

def dvy_eqm(z):
    eps = epsilon_eqm(z)
    deps = depsilon_eqm(z) 
    return eta_hat*cs*deps/(1.0+eps)/(1.0+eps)

def integrand_rhog(z, dg):
    temp = z*z/2.0 - (Diff*dg/(Omega*Omega*tau_s))*(np.exp(-0.5*Omega*Omega*z*z*tau_s/Diff) - 1.0)
    return np.exp(-Omega*Omega*temp/cs/cs)

def integrand_rhod(z, dg):
    rhog = integrand_rhog(z, dg)
    eps  = dg*np.exp(-0.5*tau_s*Omega*Omega*z*z/Diff)
    return eps*rhog

def sigma_g(dg):
    I = quad(integrand_rhog, 0.0, np.inf, args=(dg))
    return I[0]

def sigma_d(dg):
    I = quad(integrand_rhod, 0.0, np.inf, args=(dg))    
    return I[0]

def metallicity_error(dg):
    sigg = sigma_g(dg)
    sigd = sigma_d(dg)
    Z = sigd/sigg
    return Z - metal

def get_dg0_from_metal():
    Hd      = np.sqrt(delta0/(st0 + delta0))
    dgguess = metal/Hd
    sol     = broyden1(metallicity_error,[dgguess], f_tol=1e-16)
    return sol[0]

if fix_metal == True:
    dg0 = get_dg0_from_metal()

'''
setup domain and calculate derivatives of vertical profiles as needed 
'''

z_basis = de.Chebyshev('z', nz_waves, interval=(zmin*Hgas,zmax*Hgas))
domain_EVP = de.Domain([z_basis], comm=MPI.COMM_SELF)

'''
notation: W = delta_P/P = delta_ln_rhog, Q=delta_eps/eps = delta_ln_eps, U = velocities 
W_p = W_primed (dW/dz)...etc
'''

if (diffusion == True) and (tstop == True): #full problem
    waves = de.EVP(domain_EVP, ['W','W_p','Q','Q_p','Ux','Uy','Uz'], eigenvalue='sigma')
if (diffusion == False) and (tstop == True):
    waves = de.EVP(domain_EVP, ['W','W_p','Q','Ux','Uy','Uz'], eigenvalue='sigma')
if (diffusion == True) and (tstop == False):
    waves = de.EVP(domain_EVP, ['W','Q','Q_p','Ux','Uy','Uz'], eigenvalue='sigma')
if (diffusion == False) and (tstop == False): 
    waves = de.EVP(domain_EVP, ['W','Q','Ux','Uy','Uz'], eigenvalue='sigma')

'''
set up required vertical profiles as non-constant coefficients
'''

z = domain_EVP.grid(0)
    
epsilon0     = domain_EVP.new_field()
fd0          = domain_EVP.new_field()
depsilon0    = domain_EVP.new_field()
dln_epsilon0 = domain_EVP.new_field()
d2ln_epsilon0= domain_EVP.new_field()

dln_P0  = domain_EVP.new_field()
d2ln_P0 = domain_EVP.new_field()

dln_rho0 = domain_EVP.new_field()
dln_rhod0= domain_EVP.new_field()

K_over_P0= domain_EVP.new_field()
dln_K0    = domain_EVP.new_field()

vz0     = domain_EVP.new_field()
dvz0    = domain_EVP.new_field()

vy0     = domain_EVP.new_field()
dvy0    = domain_EVP.new_field()

epsilon0['g']     = epsilon_eqm(z)
fd0['g']          = epsilon_eqm(z)/(1.0+epsilon_eqm(z))
depsilon0['g']    = depsilon_eqm(z)
dln_epsilon0['g'] = dln_epsilon_eqm(z)
d2ln_epsilon0['g']= d2ln_epsilon_eqm(z)

dln_P0['g'] = dln_P_eqm(z)
d2ln_P0['g']= d2ln_P_eqm(z)

dln_rho0['g'] = dln_rho_eqm(z)
dln_rhod0['g']= dln_rhod_eqm(z)

K_over_P0['g'] = K_over_P_eqm(z)
dln_K0['g']     = dln_K_eqm(z)

vz0['g']    = vz_eqm(z)
dvz0['g']   = dvz_eqm(z)

vy0['g']     = vy_eqm(z)
dvy0['g']    = dvy_eqm(z)

'''
constant parameters
'''
waves.parameters['Diff']        = Diff
waves.parameters['visc']        = visc
waves.parameters['cs']          = cs
waves.parameters['Hgas']        = Hgas
waves.parameters['Omega']       = Omega
waves.parameters['eta_hat']     = eta_hat 
waves.parameters['tau_s']       = tau_s

waves.parameters['kx']          = kx/Hgas #convert kx to dimensional for consistency

'''
non-constant coefficients
'''
waves.parameters['epsilon0']       = epsilon0
waves.parameters['fd0']            = fd0
waves.parameters['depsilon0']      = depsilon0
waves.parameters['dln_epsilon0']   = dln_epsilon0
waves.parameters['d2ln_epsilon0']  = d2ln_epsilon0

waves.parameters['dln_P0']        = dln_P0
waves.parameters['d2ln_P0']       = d2ln_P0

waves.parameters['dln_rho0']      = dln_rho0
waves.parameters['dln_rhod0']     = dln_rhod0

waves.parameters['K_over_P0']     = K_over_P0
waves.parameters['dln_K0']         = dln_K0

waves.parameters['vy0']             = vy0
waves.parameters['dvy0']            = dvy0

waves.parameters['vz0']           = vz0
waves.parameters['dvz0']          = dvz0

'''
substitutions 
'''

waves.substitutions['ikx'] = "1j*kx"
waves.substitutions['delta_ln_rho'] = "W + epsilon0*Q/(1+epsilon0)"
waves.substitutions['delta_ln_rhod'] = "W + Q"

if tstop == True:
    waves.substitutions['dW'] = "W_p"
if tstop == False:
    waves.substitutions['dW'] = "dz(W)"

if diffusion == True:
    waves.substitutions['dQ'] = "Q_p"
if diffusion == False:
    waves.substitutions['dQ'] = "dz(Q)"

waves.substitutions['delta_ln_rho_p']  = "dW + depsilon0*Q/(1+epsilon0)/(1+epsilon0) + fd0*dQ"
waves.substitutions['delta_ln_rhod_p'] = "dW + dQ"
    
#continuity equation
if tstop == True:
    waves.substitutions['mass_LHS']="sigma*delta_ln_rho + ikx*Ux + dln_rho0*(Uz + vz0*delta_ln_rho) + vz0*delta_ln_rho_p + dvz0*delta_ln_rho + dz(Uz)"
if tstop == False:
    waves.substitutions['mass_LHS']="sigma*delta_ln_rho + ikx*Ux + dln_rho0*Uz + dz(Uz)"

if diffusion == True:
   waves.substitutions['mass_RHS']="Diff*fd0*(-kx*kx*Q + dln_rhod0*(delta_ln_rhod*dln_epsilon0 + dQ) + delta_ln_rhod_p*dln_epsilon0 + delta_ln_rhod*d2ln_epsilon0 + dz(dQ))"
if diffusion == False:
   waves.substitutions['mass_RHS']="0"
   
#pseudo-energy equation
waves.substitutions['delta_ln_K']  = "(1-epsilon0)*Q/(1+epsilon0) + W"
waves.substitutions['delta_ln_K_p']= "-2*depsilon0*Q/(1+epsilon0)/(1+epsilon0) + (1-epsilon0)*dQ/(1+epsilon0) + dW"

if tstop == True:
    waves.substitutions['energy_LHS']="sigma*W + ikx*Ux + dln_P0*(vz0*W + Uz) + vz0*dW + dvz0*W + dz(Uz)"
if tstop == False:
    waves.substitutions['energy_LHS']="sigma*(W - delta_ln_rho) + (dln_P0-dln_rho0)*Uz"

if tstop == True:
    waves.substitutions['energy_RHS']="K_over_P0*(dz(dW) + delta_ln_K_p*dln_P0 + delta_ln_K*d2ln_P0 + dln_K0*(delta_ln_K*dln_P0 + W_p) - kx*kx*W)"
if tstop == False:
    waves.substitutions['energy_RHS']="0"

#x-mom equation
waves.substitutions['P_over_rho'] = "cs*cs/(1+epsilon0)"
if tstop == True:
    waves.substitutions['xmom_LHS']="sigma*Ux + vz0*dz(Ux)"
if tstop == False:
    waves.substitutions['xmom_LHS']="sigma*Ux"
waves.substitutions['xmom_RHS']="-ikx*P_over_rho*W - 2*eta_hat*cs*Omega/(1+epsilon0)*delta_ln_rho + 2*Omega*Uy"

#y-mom equation
if tstop == True:
    waves.substitutions['ymom_LHS']="sigma*Uy + dvy0*Uz + vz0*dz(Uy)"
if tstop == False:
    waves.substitutions['ymom_LHS']="sigma*Uy"
waves.substitutions['ymom_RHS']="-0.5*Omega*Ux"

#z-mom
if tstop == True:
    waves.substitutions['zmom_LHS']="sigma*Uz + dvz0*Uz + vz0*dz(Uz)"
if tstop == False:
    waves.substitutions['zmom_LHS']="sigma*Uz"
waves.substitutions['zmom_RHS']="P_over_rho*(dln_P0*epsilon0*Q/(1+epsilon0) - dW)"

#primary equations
waves.add_equation("mass_LHS - mass_RHS = 0 ")
waves.add_equation("xmom_LHS - xmom_RHS = 0 ")
waves.add_equation("ymom_LHS - ymom_RHS = 0 ")
waves.add_equation("zmom_LHS - zmom_RHS = 0")
waves.add_equation("energy_LHS - energy_RHS = 0 ")

#equations for first derivs of perts, i.e. dz(Q) = Q_p ...etc
#in our formulation of second derivs of [epsilon, delta_vgx, delta_vgy] appear for full problem with viscosity
if tstop == True:
    waves.add_equation("dz(W) - W_p = 0")
if diffusion == True:
    waves.add_equation("dz(Q) - Q_p = 0")

'''
boundary conditions (reflection)
'''
if (diffusion == True) and (tstop == True): #full problem, 7 odes
    waves.add_bc('left(dW)=0')
    waves.add_bc('left(dQ)=0')
    waves.add_bc('left(dz(Ux))=0')
    waves.add_bc('left(dz(Uy))=0')
    waves.add_bc('left(Uz)=0')

    waves.add_bc('right(dW) = 0')
#    waves.add_bc('right(dQ) = 0')
   
#    waves.add_bc('right(Uz) = 0')
#    waves.add_bc('right(W)  = 0')
    waves.add_bc('right(Q)  = 0')

if (diffusion == False) and (tstop == True): 
    waves.add_bc('left(dW)=0')
    waves.add_bc('left(dQ)=0')
    waves.add_bc('left(dz(Ux))=0')
    waves.add_bc('left(dz(Uy))=0')
    waves.add_bc('left(Uz)=0')
    
#    waves.add_bc('right(Uz) = 0')
    waves.add_bc('right(dQ)=0')
#    waves.add_bc('right(dW) = 0')
    
if (diffusion == False) and (tstop == False): #no diffusion, perfect coupling, 2 odes (standard problem)
    waves.add_bc('left(Uz)=0')
    waves.add_bc('right(Uz) = 0')
    
if (diffusion == True) and (tstop == False):
    waves.add_bc('left(dW)=0')
#    waves.add_bc('left(dQ)=0')
    waves.add_bc('left(Uz)=0')
    waves.add_bc('right(Uz) = 0')


'''
eigenvalue problem, sweep through kx space
for each kx, filter modes and keep most unstable one
'''

#EP_list = [Eigenproblem(waves), Eigenproblem(waves, sparse=True)] 
EP = Eigenproblem(waves, sparse=True)
kx_space = np.logspace(np.log10(kx_min),np.log10(kx_max), num=nkx)

eigenfreq = []
eigenfunc = {'W':[], 'Q':[], 'Ux':[], 'Uy':[], 'Uz':[]}

for i, kx in enumerate(kx_space):

    # for n in range(0,2):
    #     EP_list[n].EVP.namespace['kx'].value = kx
    #     EP_list[n].EVP.parameters['kx'] = kx
    
    # if i == 0:
    #     EP = EP_list[0]
    #     EP.solve()
    # else:
    #     EP = EP_list[1]
    #     trial = eigenfreq[i-1]
    #     EP.solve(N=20, target = trial)

    EP.EVP.namespace['kx'].value = kx
    EP.EVP.parameters['kx'] = kx

    if i == 0:
        trial = 0.5*Omega
    else:
        trial = eigenfreq[i-1]
        
    EP.solve(N=20, target = trial)
    EP.reject_spurious()

    abs_sig = np.abs(EP.evalues_good)
    sig_acceptable = abs_sig < Omega

    sigma      = EP.evalues_good#[sig_acceptable]
    sigma_index= EP.evalues_good_index#[sig_acceptable]
    
    if sigma.size > 0:        
        growth   =  np.real(sigma)
        g1       =  np.argmax(growth)
        opt_freq = sigma[g1]
        
        g2 = sigma_index[g1]
        EP.solver.set_state(g2)
        W  = EP.solver.state['W']
        Q  = EP.solver.state['Q']
        Ux = EP.solver.state['Ux']
        Uy = EP.solver.state['Uy']
        Uz = EP.solver.state['Uz']

        W.set_scales(scales=out_scale)
        Q.set_scales(scales=out_scale)
        Ux.set_scales(scales=out_scale)
        Uy.set_scales(scales=out_scale)
        Uz.set_scales(scales=out_scale)

        Wout = W['g']
        Qout = Q['g']
        Uxout = Ux['g']
        Uyout = Uy['g']
        Uzout = Uz['g']
        
    else:
        opt_freq = np.nan
        Wout = np.zeros(nz_out)
        Qout = np.zeros(nz_out)
        Uxout= np.zeros(nz_out)
        Uyout= np.zeros(nz_out)
        Uzout= np.zeros(nz_out)

    eigenfreq.append(opt_freq) #store eigenfreq
    
    eigenfunc['W'].append([])
    eigenfunc['Q'].append([])
    eigenfunc['Ux'].append([])
    eigenfunc['Uy'].append([])
    eigenfunc['Uz'].append([])

    eigenfunc['W'][i] = np.copy(Wout)
    eigenfunc['Q'][i] = np.copy(Qout)
    eigenfunc['Ux'][i] = np.copy(Uxout)
    eigenfunc['Uy'][i] = np.copy(Uyout)
    eigenfunc['Uz'][i] = np.copy(Uzout)
    
'''
print results to screen
'''
for i, kx in enumerate(kx_space):
    print("i, kx, growth, freq = {0:3d} {1:1.2e} {2:9.6f} {3:9.6f}".format(i, kx, eigenfreq[i].real, -eigenfreq[i].imag))
    
'''
data output
'''
    
with h5py.File('stratsi_1fluid_modes.h5','w') as outfile:
    z_out = domain_EVP.grid(0, scales=out_scale)

    scale_group = outfile.create_group('scales')
    scale_group.create_dataset('kx_space',data=kx_space)
    scale_group.create_dataset('eig_freq',data=eigenfreq)
    scale_group.create_dataset('z',   data=z_out)
    scale_group.create_dataset('zmax', data=zmax)

    
    tasks_group = outfile.create_group('tasks')

    for i, kx in enumerate(kx_space):
        data_group = tasks_group.create_group('k_{:03d}'.format(i))
        data_group.create_dataset('eig_W',data=eigenfunc['W'][i])
        data_group.create_dataset('eig_Q',data=eigenfunc['Q'][i])
        data_group.create_dataset('eig_Ux',data=eigenfunc['Ux'][i])
        data_group.create_dataset('eig_Uy',data=eigenfunc['Uy'][i])
        data_group.create_dataset('eig_Uz',data=eigenfunc['Uz'][i])
    outfile.close()

'''
plot equilibrium profiles
'''

fontsize= 24
nlev    = 128
nclev   = 6
cmap    = plt.cm.inferno
    
z = domain_EVP.grid(0, scales=nz_out)
eps = epsilon_eqm(z)

fig = plt.figure(figsize=(8,4.5))
ax = fig.add_subplot()
plt.subplots_adjust(left=0.18, right=0.95, top=0.95, bottom=0.2)

ymax = np.amax(eps)

plt.xlim(zmin,zmax)
plt.ylim(0,ymax)

plt.plot(z, eps, linewidth=2)

plt.rc('font',size=fontsize,weight='bold')

#lines1, labels1 = ax.get_legend_handles_labels()
#legend=ax.legend(lines1, labels1, loc='upper right', frameon=False, ncol=1, fontsize=fontsize/2)

plt.xticks(fontsize=fontsize,weight='bold')
plt.xlabel(r'$z/H_g$',fontsize=fontsize)

plt.yticks(fontsize=fontsize,weight='bold')
plt.ylabel(r'$\rho_d/\rho_g$', fontsize=fontsize)

fname = 'stratsi_1fluid_epsilon'
plt.savefig(fname,dpi=150)
