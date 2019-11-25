"""

stratified linear analysis of the streaming instability

"""
import sys
import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt
from dedalus import public as de
import time

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

kx = 400.0

'''
physics options 
can choose to include/exclude: gas viscosity, particle diffusion, particle backreaction
'''
viscosity    = False
diffusion    = True
backreaction = True

'''
read in background vertical structure:
epsilon, rhog, vdust_z, from eqm_vert

'''

vert_eqm = h5py.File('./eqm_vert.h5', 'r')
rhog0    = vert_eqm['rhog0'][()]
alpha0   = vert_eqm['alpha0'][()]
epsilon0 = vert_eqm['epsilon0'][()]
st0      = vert_eqm['st0'][()]
eta_hat0 = vert_eqm['eta_hat0'][()]
delta0   = vert_eqm['delta0'][()] 
beta     = vert_eqm['beta'][()]
zmin     = vert_eqm['zmin'][()]
zmax     = vert_eqm['zmax'][()]
nz_vert  = vert_eqm['nz'][()]
fixedSt  = vert_eqm['fixedSt'][()]

zaxis_vert= vert_eqm['z'][:]
ln_epsilon= vert_eqm['ln_epsilon'][:]
ln_rhog   = vert_eqm['ln_rhog'][:]
vdz       = vert_eqm['vdz'][:]
stokes    = vert_eqm['stokes'][:]
eta       = vert_eqm['eta'][:]
vert_eqm.close()

horiz_eqm  = h5py.File('./eqm_horiz.h5', 'r')
zaxis_horiz= horiz_eqm['z'][:]
nz_horiz   = horiz_eqm['nz'][()]
vgx        = horiz_eqm['vgx'][:]
vgy        = horiz_eqm['vgy'][:]
vdx        = horiz_eqm['vdx'][:]
vdy        = horiz_eqm['vdy'][:]
horiz_eqm.close()

'''
setup domain and calculate derivatives of vertical profiles as needed 
'''

nz_waves = 256 #384
z_basis = de.Chebyshev('z', nz_waves, interval=(zmin,zmax))
domain_EVP = de.Domain([z_basis], comm=MPI.COMM_SELF)
'''
notation: W = delta_rhog/rhog = delta_ln_rhog, Q=delta_eps/eps = delta_ln_eps, U = velocities 
W_p = W_primed (dW/dz)...etc
'''


if (viscosity == True) and (diffusion == True):#full problem: include viscosity and particle diffusion
    waves = de.EVP(domain_EVP, ['W','Ugx','Ugx_p','Ugy','Ugy_p','Ugz','Q','Q_p','Udx','Udy','Udz'], eigenvalue='sigma')

if (viscosity == True) and (diffusion == False):#include viscosity but no particle diffusion
    waves = de.EVP(domain_EVP, ['W','Ugx','Ugx_p','Ugy','Ugy_p','Ugz','Q','Udx','Udy','Udz'], eigenvalue='sigma')
    
if (viscosity == False) and (diffusion == True):#ignore gas viscosity but include particle diffusion 
    waves = de.EVP(domain_EVP, ['W','Ugx','Ugy','Ugz','Q','Q_p','Udx','Udy','Udz'], eigenvalue='sigma')
    
if (viscosity == False) and (diffusion == False):#ignore gas viscosity and ignore diffusion  
    waves = de.EVP(domain_EVP, ['W','Ugx','Ugy','Ugz','Q','Udx','Udy','Udz'], eigenvalue='sigma')

    
'''
set up required vertical profiles in epsilon, rhog, rhod, and vdz
'''

ln_rhog0  = domain_EVP.new_field()
dln_rhog0 = domain_EVP.new_field()
d2ln_rhog0 = domain_EVP.new_field()

ln_rhod0  = domain_EVP.new_field()
dln_rhod0 = domain_EVP.new_field()

epsilon0   = domain_EVP.new_field()
depsilon0  = domain_EVP.new_field()
d2epsilon0 = domain_EVP.new_field()

ln_epsilon0 = domain_EVP.new_field()
dln_epsilon0= domain_EVP.new_field()

vdz0        = domain_EVP.new_field()
dvdz0       = domain_EVP.new_field()

inv_stokes0     = domain_EVP.new_field()

scale = nz_vert/nz_waves

ln_rhog0.set_scales(scale)
dln_rhog0.set_scales(scale)
d2ln_rhog0.set_scales(scale)

ln_rhod0.set_scales(scale)
dln_rhod0.set_scales(scale)

epsilon0.set_scales(scale)
depsilon0.set_scales(scale)
d2epsilon0.set_scales(scale)

ln_epsilon0.set_scales(scale)
dln_epsilon0.set_scales(scale)

vdz0.set_scales(scale)
dvdz0.set_scales(scale)

inv_stokes0.set_scales(scale)

ln_rhog0['g'] = ln_rhog
ln_rhog0.differentiate('z', out=dln_rhog0)
dln_rhog0.differentiate('z', out=d2ln_rhog0)

ln_rhod0['g'] = ln_epsilon + ln_rhog
ln_rhod0.differentiate('z', out=dln_rhod0)

epsilon0['g'] = np.exp(ln_epsilon)
epsilon0.differentiate('z', out=depsilon0)
depsilon0.differentiate('z', out=d2epsilon0)

ln_epsilon0['g'] = ln_epsilon
ln_epsilon0.differentiate('z', out=dln_epsilon0)

vdz0['g'] = vdz
vdz0.differentiate('z',out=dvdz0)

inv_stokes0['g'] = 1.0/stokes

ln_rhog0.set_scales(1, keep_data=True)
dln_rhog0.set_scales(1, keep_data=True)
d2ln_rhog0.set_scales(1, keep_data=True)

ln_rhod0.set_scales(1, keep_data=True)
dln_rhod0.set_scales(1, keep_data=True)

epsilon0.set_scales(1, keep_data=True)
depsilon0.set_scales(1, keep_data=True)
d2epsilon0.set_scales(1, keep_data=True)

ln_epsilon0.set_scales(1, keep_data=True)
dln_epsilon0.set_scales(1, keep_data=True)

vdz0.set_scales(1, keep_data=True)
dvdz0.set_scales(1, keep_data=True)

inv_stokes0.set_scales(1,keep_data=True)

'''
set up required vertical profiles in vgx, vgy, vdx, vdy
'''
vgx0  = domain_EVP.new_field()
dvgx0 = domain_EVP.new_field()
d2vgx0= domain_EVP.new_field()

vgy0  = domain_EVP.new_field()
dvgy0 = domain_EVP.new_field()
d2vgy0= domain_EVP.new_field()

vdx0  = domain_EVP.new_field()
dvdx0 = domain_EVP.new_field()

vdy0  = domain_EVP.new_field()
dvdy0 = domain_EVP.new_field()

scale = nz_horiz/nz_waves

vgx0.set_scales(scale)
dvgx0.set_scales(scale)
d2vgx0.set_scales(scale)

vgy0.set_scales(scale)
dvgy0.set_scales(scale)
d2vgy0.set_scales(scale)

vdx0.set_scales(scale)
dvdx0.set_scales(scale)

vdy0.set_scales(scale)
dvdy0.set_scales(scale)

vgx0['g'] = vgx
vgx0.differentiate('z', out=dvgx0)
dvgx0.differentiate('z',out=d2vgx0)

vgy0['g'] = vgy
vgy0.differentiate('z', out=dvgy0)
dvgy0.differentiate('z',out=d2vgy0)

vdx0['g'] = vdx
vdx0.differentiate('z', out=dvdx0)

vdy0['g'] = vdy
vdy0.differentiate('z', out=dvdy0)

vgx0.set_scales(1, keep_data=True)
dvgx0.set_scales(1, keep_data=True)
d2vgx0.set_scales(1, keep_data=True)

vgy0.set_scales(1, keep_data=True)
dvgy0.set_scales(1, keep_data=True)
d2vgy0.set_scales(1, keep_data=True)

vdx0.set_scales(1, keep_data=True)
dvdx0.set_scales(1, keep_data=True)

vdy0.set_scales(1, keep_data=True)
dvdy0.set_scales(1, keep_data=True)

'''
constant parameters
'''
waves.parameters['delta0']      = delta0
waves.parameters['alpha0']      = alpha0
waves.parameters['cs']          = cs
waves.parameters['Hgas']        = Hgas
waves.parameters['Omega']       = Omega

waves.parameters['kx']          = kx

'''
non-constant coefficients
'''
waves.parameters['epsilon0']      = epsilon0
waves.parameters['depsilon0']     = depsilon0
waves.parameters['dln_epsilon0']  = dln_epsilon0
waves.parameters['d2epsilon0']    = d2epsilon0

waves.parameters['dln_rhog0']      = dln_rhog0
waves.parameters['d2ln_rhog0']     = d2ln_rhog0
waves.parameters['dln_rhod0']      = dln_rhod0

waves.parameters['vdx0']             = vdx0
waves.parameters['dvdx0']            = dvdx0

waves.parameters['vdy0']             = vdy0
waves.parameters['dvdy0']            = dvdy0

waves.parameters['vdz0']           = vdz0
waves.parameters['dvdz0']          = dvdz0

waves.parameters['vgx0']              = vgx0
waves.parameters['dvgx0']             = dvgx0
waves.parameters['d2vgx0']            = d2vgx0

waves.parameters['vgy0']              = vgy0
waves.parameters['dvgy0']             = dvgy0
waves.parameters['d2vgy0']            = d2vgy0

waves.parameters['inv_stokes0']    = inv_stokes0


# W is (delta_rhog)/rhog, Q is (delta_epsilon)/epsilon  

waves.substitutions['ikx'] = "1j*kx"
waves.substitutions['delta_ln_rhod'] = "Q + W"

if diffusion == True:
    waves.substitutions['delta_ln_rhod_p'] = "Q_p + dz(W)"
    waves.substitutions['delta_eps_p_over_eps'] = "dln_epsilon0*Q + Q_p"
    waves.substitutions['delta_eps_pp_over_eps'] = "d2epsilon0*Q/epsilon0 + 2*dln_epsilon0*Q_p + dz(Q_p)"
if diffusion == False:
    waves.substitutions['delta_ln_rhod_p'] = "dz(Q) + dz(W)"
    
if fixedSt == True:
    waves.substitutions['delta_ln_taus'] = "0"
else:
    waves.substitutions['delta_ln_taus'] = "-W"

#dust continuity equation
waves.substitutions['dust_mass_LHS']="sigma*delta_ln_rhod + ikx*(Udx + vdx0*delta_ln_rhod) + dln_rhod0*(Udz + vdz0*delta_ln_rhod) + vdz0*delta_ln_rhod_p + dvdz0*delta_ln_rhod + dz(Udz)"
if diffusion == True:
    waves.substitutions['dust_mass_RHS']="delta0*cs*Hgas*(dln_epsilon0*(dln_rhog0*W + dz(W)) + d2epsilon0*W/epsilon0 - kx*kx*Q + dln_rhog0*delta_eps_p_over_eps + delta_eps_pp_over_eps)"
if diffusion == False:
    waves.substitutions['dust_mass_RHS']="0"

#dust x-mom equation
waves.substitutions['dust_xmom_LHS']="sigma*Udx + dvdx0*Udz + ikx*vdx0*Udx + vdz0*dz(Udx)"
waves.substitutions['dust_xmom_RHS']="2*Omega*Udy + inv_stokes0*delta_ln_taus*(vdx0 - vgx0) - inv_stokes0*(Udx - Ugx)"

#dust y-mom equation
waves.substitutions['dust_ymom_LHS']="sigma*Udy + dvdy0*Udz + ikx*vdx0*Udy + vdz0*dz(Udy)"
waves.substitutions['dust_ymom_RHS']="-0.5*Omega*Udx + inv_stokes0*delta_ln_taus*(vdy0 - vgy0) - inv_stokes0*(Udy - Ugy)"

#dust z-mom
waves.substitutions['dust_zmom_LHS']="sigma*Udz + dvdz0*Udz + ikx*vdx0*Udz + vdz0*dz(Udz)"
waves.substitutions['dust_zmom_RHS']="inv_stokes0*delta_ln_taus*vdz0 - inv_stokes0*(Udz - Ugz)"

#gas continuity equation
waves.substitutions['gas_mass_LHS']="sigma*W + ikx*(Ugx + vgx0*W) + dln_rhog0*Ugz + dz(Ugz)"

#linearized viscous forces on gas
#could also use eqm eqns to replace first bracket, so that we don't need to take numerical derivs of vgx..etc
if viscosity == True:
    waves.substitutions['delta_vgz_pp']="-( sigma*dz(W) + ikx*(Ugx_p + dvgx0*W + vgx0*dz(W)) + d2ln_rhog0*Ugz + dln_rhog0*dz(Ugz) )" #take deriv of gas mass eq to get d2(delta_vgz)/dz2
    waves.substitutions['delta_Fvisc_x'] = "alpha0*cs*Hgas*(ikx*dz(Ugz)/3 - 4*kx*kx*Ugx/3 + dln_rhog0*(ikx*Ugz + Ugx_p) + dz(Ugx_p)) - alpha0*cs*Hgas*(dln_rhog0*dvgx0 + d2vgx0)*W"
    waves.substitutions['delta_Fvisc_y'] = "alpha0*cs*Hgas*(dln_rhog0*Ugy_p - kx*kx*Ugy + dz(Ugy_p)) - alpha0*cs*Hgas*(dln_rhog0*dvgy0 + d2vgy0)*W"
    waves.substitutions['delta_Fvisc_z'] = "alpha0*cs*Hgas*(ikx*Ugx_p/3 - kx*kx*Ugz + dln_rhog0*(4*dz(Ugz)/3 - 2*ikx*Ugx/3) + 4*delta_vgz_pp/3)"
if viscosity == False:
    waves.substitutions['delta_Fvisc_x'] = "0"
    waves.substitutions['delta_Fvisc_y'] = "0"
    waves.substitutions['delta_Fvisc_z'] = "0"
    
#linearized back-reaction force on gas
if backreaction == True:
    waves.substitutions['delta_backreaction_x']="inv_stokes0*epsilon0*( (vgx0 - vdx0)*(delta_ln_taus - Q) - (Ugx - Udx) )"
    waves.substitutions['delta_backreaction_y']="inv_stokes0*epsilon0*( (vgy0 - vdy0)*(delta_ln_taus - Q) - (Ugy - Udy) )"
    waves.substitutions['delta_backreaction_z']="inv_stokes0*epsilon0*( (     - vdz0)*(delta_ln_taus - Q) - (Ugz - Udz) )"
if backreaction == False:
    waves.substitutions['delta_backreaction_x']="0"
    waves.substitutions['delta_backreaction_y']="0"
    waves.substitutions['delta_backreaction_z']="0"

    
#gas equations
waves.add_equation("gas_mass_LHS = 0 ")
waves.add_equation("sigma*Ugx + dvgx0*Ugz + ikx*vgx0*Ugx - 2*Omega*Ugy + ikx*cs*cs*W - delta_backreaction_x - delta_Fvisc_x = 0")
waves.add_equation("sigma*Ugy + dvgy0*Ugz + ikx*vgx0*Ugy + 0.5*Omega*Ugx - delta_backreaction_y - delta_Fvisc_y = 0")
waves.add_equation("sigma*Ugz + ikx*vgx0*Ugz + cs*cs*dz(W) - delta_backreaction_z - delta_Fvisc_z = 0")

#dust equations 
waves.add_equation("dust_mass_LHS - dust_mass_RHS = 0 ")
waves.add_equation("dust_xmom_LHS - dust_xmom_RHS = 0 ")
waves.add_equation("dust_ymom_LHS - dust_ymom_RHS = 0 ")
waves.add_equation("dust_zmom_LHS - dust_zmom_RHS = 0")

#equations for first derivs of perts, i.e. dz(Q) = Q_p ...etc
#in our formulation of second derivs of [epsilon, delta_vgx, delta_vgy] appear for full problem with viscosity
if diffusion == True:
    waves.add_equation("dz(Q) - Q_p = 0")
if viscosity == True:
    waves.add_equation("dz(Ugx) - Ugx_p = 0")
    waves.add_equation("dz(Ugy) - Ugy_p = 0")

'''
boundary conditions (reflection)
'''
waves.add_bc('left(dz(W))=0')
# waves.add_bc('left(Ugz)=0')
# waves.add_bc('left(dz(Udx))=0')
# waves.add_bc('left(dz(Udy))=0')
# waves.add_bc('left(Udz)=0')
# waves.add_bc('right(Ugz)=0')

waves.add_bc('left(dz(Ugx - epsilon0*vgx0*Q/(1+epsilon0) + epsilon0*Udx + epsilon0*vdx0*Q/(1+epsilon0)))=0')
waves.add_bc('left(dz(Ugy - epsilon0*vgy0*Q/(1+epsilon0) + epsilon0*Udy + epsilon0*vdy0*Q/(1+epsilon0)))=0')
waves.add_bc('left(Ugz + epsilon0*Udz)=0')
#waves.add_bc('right(dz(W))=0')
waves.add_bc('right(Q)=0')
waves.add_bc('right(Ugz + epsilon0*Udz + epsilon0*vdz0*Q/(1+epsilon0))=0')

if diffusion == True:
    waves.add_bc('left(Q_p)=0')

if viscosity == True:
    waves.add_bc('left(Ugx_p)=0')
    waves.add_bc('left(Ugy_p)=0')
    waves.add_bc('right(Ugx_p)=0')
    waves.add_bc('right(Ugy_p)=0')


EP = Eigenproblem(waves)
EP.EVP.namespace['kx'].value = kx
EP.EVP.parameters['kx'] = kx
EP.solve()
EP.reject_spurious()
sigma = EP.evalues_good


print(sigma)

# Solver
# solver = waves.build_solver()
# t1 = time.time()
# solver.solve_dense(solver.pencils[0])
# t2 = time.time()
# logger.info('Elapsed solve time: %f' %(t2-t1))

# Filter infinite/nan eigenmodes
#finite = np.isfinite(solver.eigenvalues)

#sigma   = solver.eigenvalues
#growth  = np.real(sigma)
#abs_sig = np.abs(sigma) 
##growth_acceptable = growth[np.logical_and(growth > 0.0, growth < 1.0)]
##growth_acceptable = np.logical_and(growth > 0.0, abs_sig < 1.0)
#growth_acceptable = abs_sig < Omega

#solver.eigenvalues = solver.eigenvalues[growth_acceptable]
#solver.eigenvectors = solver.eigenvectors[:, growth_acceptable]
#sigma  = solver.eigenvalues

growth =  np.real(sigma)
freq   = -np.imag(sigma) #define  s= s_r - i*omega
abs_sig = np.abs(sigma)

growth_acceptable = abs_sig < Omega
sigma = sigma[growth_acceptable]

growth =  np.real(sigma)
freq   = -np.imag(sigma)


N = 6 #for low freq modes, (kx*omega)^2 should be an integer (kx norm by H, omega norm by Omega)
#g1 = np.argmin(np.abs(np.power(kx*freq,2.0) - N))
#g1 = np.argmin(np.abs(freq))
#g1=np.argmin(np.abs(sigma))

g1=np.argmax(growth)

print(g1)
print(sigma[g1])

N_actual = np.power(kx*freq[g1]*Hgas/Omega,2.0)
print(N_actual)

#g1 = (EP.evalues_good_index[g1])
g1 = np.argmin(np.abs(EP.evalues-sigma[g1]))


EP.solver.set_state(g1)
W = EP.solver.state['W']
Q = EP.solver.state['Q']
Ugz = EP.solver.state['Ugz']
Udz = EP.solver.state['Udz']

# solver.set_state(g1)
# W = solver.state['W']
# Ugz = solver.state['Ugz']



'''
#plotting parameters
'''
fontsize= 24
nlev    = 128
nclev   = 6
cmap    = plt.cm.inferno

fig = plt.figure(figsize=(8,4.5))
ax = fig.add_subplot()
plt.subplots_adjust(left=0.18, right=0.95, top=0.95, bottom=0.2)

z    = domain_EVP.grid(0, scales=16)
Udz.set_scales(scales=16)
max_Udz = np.amax(np.abs(Udz['g']))
Udz_norm = np.conj(Udz['g'][0])*Udz['g']#/max_Udz/max_Udz #np.power(np.abs(Udz['g'][0]),2)

#plt.ylim(0,1)
#plt.xlim(zmin,zmax)

plt.plot(z, np.real(Udz_norm)/np.amax(np.abs(Udz_norm)), linewidth=2, label=r'real')
plt.plot(z, np.imag(Udz_norm)/np.amax(np.abs(Udz_norm)), linewidth=2, label=r'imaginary')

#plt.plot(z, np.abs(Udz['g'])/max_Udz, linewidth=2, label=r'absolute')

plt.rc('font',size=fontsize,weight='bold')

lines1, labels1 = ax.get_legend_handles_labels()
legend=ax.legend(lines1, labels1, loc='upper right', frameon=False, ncol=1, fontsize=fontsize/2)

plt.xticks(fontsize=fontsize,weight='bold')
plt.xlabel(r'$z/H_g$',fontsize=fontsize)

plt.yticks(fontsize=fontsize,weight='bold')
#plt.ylabel(r'$\delta\rho_g/\rho_g$', fontsize=fontsize)
#plt.ylabel(r'$\delta v_{gz}/c_s$', fontsize=fontsize)
plt.ylabel(r'$\delta v_{dz}/|\delta v_{dz}|_{max}$', fontsize=fontsize)
#plt.ylabel(r'$\delta v_{dz}$', fontsize=fontsize)

fname = 'stratsi_vdz'
plt.savefig(fname,dpi=150)

######################################################################################################

fig = plt.figure(figsize=(8,4.5))
ax = fig.add_subplot()
plt.subplots_adjust(left=0.18, right=0.95, top=0.95, bottom=0.2)

z    = domain_EVP.grid(0, scales=16)
Ugz.set_scales(scales=16)
max_Ugz = np.amax(np.abs(Ugz['g']))

Ugz_norm = np.conj(Ugz['g'][0])/max_Ugz/max_Ugz

plt.plot(z, np.real(Ugz['g']*Ugz_norm), linewidth=2, label=r'real')
plt.plot(z, np.imag(Ugz['g']*Ugz_norm), linewidth=2, label=r'imaginary')

plt.rc('font',size=fontsize,weight='bold')

lines1, labels1 = ax.get_legend_handles_labels()
legend=ax.legend(lines1, labels1, loc='upper right', frameon=False, ncol=1, fontsize=fontsize/2)

plt.xticks(fontsize=fontsize,weight='bold')
plt.xlabel(r'$z/H_g$',fontsize=fontsize)

plt.yticks(fontsize=fontsize,weight='bold')
plt.ylabel(r'$\delta v_{gz}/|\delta v_{gz}|_{max}$', fontsize=fontsize)

fname = 'stratsi_vgz'
plt.savefig(fname,dpi=150)

######################################################################################################

fig = plt.figure(figsize=(8,4.5))
ax = fig.add_subplot()
plt.subplots_adjust(left=0.18, right=0.95, top=0.95, bottom=0.2)

z    = domain_EVP.grid(0, scales=16)
Q.set_scales(scales=16)
Qmax = np.amax(np.abs(Q['g']))
#Wnorm = np.conj(W['g'][0])/np.power(np.abs(W['g'][0]),2)

plt.plot(z, np.real(Q['g'])/Qmax, linewidth=2, label=r'real')
plt.plot(z, np.imag(Q['g'])/Qmax, linewidth=2, label=r'imaginary')

plt.rc('font',size=fontsize,weight='bold')

lines1, labels1 = ax.get_legend_handles_labels()
legend=ax.legend(lines1, labels1, loc='upper right', frameon=False, ncol=1, fontsize=fontsize/2)

plt.xticks(fontsize=fontsize,weight='bold')
plt.xlabel(r'$z/H_g$',fontsize=fontsize)

plt.yticks(fontsize=fontsize,weight='bold')
plt.ylabel(r'$\delta \epsilon/\epsilon$', fontsize=fontsize)

fname = 'stratsi_Q'
plt.savefig(fname,dpi=150)



'''

# value at top of atmosphere in isothermal layer
brunt_max = np.max(np.sqrt(np.abs(brunt2))) # max value in atmosphere
k_Hρ = -1/2*del_ln_rho0['g'][0].real
c_s = np.sqrt(T0['g'][0].real)

logger.info("max Brunt is |N| = {} and  k_Hρ is {}".format(brunt_max, k_Hρ))
start_time = time.time()
EP = Eigenproblem(waves)
ks = np.logspace(-1,2, num=20)*k_Hρ

freqs = []
eigenfunctions = {'w':[], 'u':[], 'T':[]}
omega = {'ω_plus_min':[], 'ω_minus_max':[]}
w_weights = []
KE = domain_EVP.new_field()
rho0 = domain_EVP.new_field()
rho0['g'] = np.exp(ln_rho0['g'])
rho0_avg = (rho0.integrate('z')['g'][0]/Lz).real
logger.debug("aveage ρ0 = {:g}".format(rho0_avg))
fig, ax = plt.subplots()
for i, k in enumerate(ks):
    ω_lamb2 = k**2*c_s2
    ω_plus2 = ω_lamb2 + ω_ac2
    ω_minus2  = brunt2*ω_lamb2/(ω_lamb2 + ω_ac2)
    omega['ω_plus_min'].append(np.min(np.sqrt(ω_plus2)))
    omega['ω_minus_max'].append(np.max(np.sqrt(ω_minus2)))
    EP.EVP.namespace['k'].value = k
    EP.EVP.parameters['k'] = k
    EP.solve()
    EP.reject_spurious()
    ω = EP.evalues_good
    ax.plot([k]*len(ω), np.abs(ω.real)/brunt_max, marker='x', linestyle='none')
    freqs.append(ω)
    eigenfunctions['w'].append([])
    eigenfunctions['u'].append([])
    eigenfunctions['T'].append([])
    w_weights.append([])
    logger.info("k={:g} ; {:d} good eigenvalues among {:d} fields ({:g}%)".format(k, EP.evalues_good_index.shape[0], n_var, EP.evalues_good_index.shape[0]/(n_var*nz_waves)*100))
    for ikk, ik in enumerate(EP.evalues_good_index):
        EP.solver.set_state(ik)
        w = EP.solver.state['w']
        u = EP.solver.state['u']
        T = EP.solver.state['T1']

        i_max = np.argmax(np.abs(w['g']))
        phase_correction = w['g'][i_max]
        w['g'] /= phase_correction
        u['g'] /= phase_correction
        T['g'] /= phase_correction

        KE['g'] = 0.5*rho0['g']*(u['g']*np.conj(u['g'])+w['g']*np.conj(w['g'])).real
        KE_avg = (KE.integrate('z')['g'][0]/Lz).real
        weight = np.sqrt(KE_avg/(0.5*rho0_avg))

        eigenfunctions['w'][i].append(np.copy(w['g'])/weight)
        eigenfunctions['u'][i].append(np.copy(u['g'])/weight)
        eigenfunctions['T'][i].append(np.copy(T['g'])/weight)


ax.set_xscale('log')
end_time = time.time()
logger.info("time to solve all modes: {:g} seconds".format(end_time-start_time))


with h5py.File('wave_frequencies.h5','w') as outfile:
    scale_group = outfile.create_group('scales')

    scale_group.create_dataset('grid',data=ks)
    scale_group.create_dataset('brunt_max', data=brunt_max)
    scale_group.create_dataset('k_Hρ',  data=k_Hρ)
    scale_group.create_dataset('c_s',   data=c_s)
    scale_group.create_dataset('z',   data=z)
    scale_group.create_dataset('Lz',  data=Lz)
    scale_group.create_dataset('rho0', data=rho0['g'])

    tasks_group = outfile.create_group('tasks')

    for i, freq in enumerate(freqs):
        data_group = tasks_group.create_group('k_{:03d}'.format(i))
        data_group.create_dataset('freq',data=freq)
        data_group.create_dataset('ω_plus_min',data=omega['ω_plus_min'][i])
        data_group.create_dataset('ω_minus_max',data=omega['ω_minus_max'][i])
        data_group.create_dataset('eig_w',data=eigenfunctions['w'][i])
        data_group.create_dataset('eig_u',data=eigenfunctions['u'][i])
        data_group.create_dataset('eig_T',data=eigenfunctions['T'][i])
    outfile.close()
plt.show()
'''
