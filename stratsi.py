"""

stratified linear analysis of the streaming instability

"""
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
parameters for eigenvalue problem
kx normalized by 1/Hgas
'''

kx = 100.0


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

zaxis_vert= vert_eqm['z'][:]
ln_epsilon= vert_eqm['ln_epsilon'][:]
ln_rhog   = vert_eqm['ln_rhog'][:]
vdz       = vert_eqm['vdz'][:]
stokes    = vert_eqm['stokes'][:]
eta       = vert_eqm['eta'][:]
vert_eqm.close()

horiz_eqm  = h5py.File('./eqm_vert.h5', 'r')
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

nz_waves = 128 #384
z_basis = de.Chebyshev('z', nz_waves, interval=(0,Lz))
domain_EVP = de.Domain([z_basis], comm=MPI.COMM_SELF)
'''
notation: W = delta_rhog/rhog = delta_ln_rhog, Q=delta_rhod/rhod = delta_ln_rhod, U = velocities 
W_p = W_primed (dW/dz)...etc
'''

waves = de.EVP(domain_EVP, ['W','W_p','Ugx','Ugx_p','Ugy','Ugy_p','Ugz','Ugz_p','Q','Q_p','Udx','Udx_p','Udy','Udy_p','Udz','Udz_p'], eigenvalue='sigma')

'''
set up required vertical profiles in epsilon, rhog, rhod, and vdz
'''

ln_rhog0  = domain_EVP.new_field()
dln_rhog0 = domain_EVP.new_field()

ln_rhod0  = domain_EVP.new_field()
dln_rhod0 = domain_EVP.new_field()

epsilon0   = domain_EVP.new_field()
depsilon0  = domain_EVP.new_field()
d2epsilon0 = domain_EVP.new_field()

ln_epsilon0 = domain_EVP.new_field()
dln_epsilon0= domain_EVP.new_field()

vdz0        = domain_EVP.new_field()
dvdz0       = domain_EVP.new_field()

scale = nz_vert/nz_waves

ln_rhog0.set_scales(scale)
dln_rhog0.set_scales(scale)

ln_rhod0.set_scales(scale)
dln_rhod0.set_scales(scale)

epsilon0.set_scales(scale)
depsilon0.set_scales(scale)
d2epsilon0.set_scales(scale)

ln_epsilon0.set_scales(scale)
dln_epsilon0.set_scales(scale)

vdz0.set_scales(scale)
dvdz0.set_scales(scale)

ln_rhog0['g'] = ln_rhog
ln_rhog0.differentiate('z', out=dln_rhog0)

ln_rhod0['g'] = ln_epsilon + ln_rhog
ln_rhod0.differentiate('z', out=dln_rhod0)

epsilon0['g'] = np.exp(ln_epsilon)
epsilon0.differentiate('z', out=depsilon0)
depsilon0.differentite('z', out=d2epsilon0)

ln_epsilon0['g'] = ln_epsilon
ln_epsilon0.differentiate('z', out=dln_epsilon0)

vdz0['g'] = vdz
vdz0.differentiate('z',out=dvdz0)

ln_rhog0.set_scales(1, keep_data=True)
dln_rhog0.set_scales(1, keep_data=True)

ln_rhod0.set_scales(1, keep_data=True)
dln_rhod0.set_scales(1, keep_data=True)

epsilon0.set_scales(1, keep_data=True)
depsilon0.set_scales(1, keep_data=True)
d2epsilon0.set_scales(1, keep_data=True)

ln_epsilon0.set_scales(1, keep_data=True)
dln_epsilon0.set_scales(1, keep_data=True)

vdz0.set_scales(1, keep_data=True)
dvdz0.set_scales(1, keep_data=True)


'''
set up required vertical profiles in vgx, vgy, vdx, vdy
'''
vgx0  = domain_EVP.new_field()
dvgx0 = domain_EVP.new_field()
vgy0  = domain_EVP.new_field()
dvgy0 = domain_EVP.new_field()
vdx0  = domain_EVP.new_field()
dvdx0 = domain_EVP.new_field()
vdy0  = domain_EVP.new_field()
dvdy0 = domain_EVP.new_field()

scale = nz_vert/nz_waves

vgx0.set_scales(scale)
dvgx0.set_scales(scale)
vgy0.set_scales(scale)
dvgy0.set_scales(scale)
vdx0.set_scales(scale)
dvdx0.set_scales(scale)
vdy0.set_scales(scale)
dvdy0.set_scales(scale)

vgx0['g'] = vgx
vgx0.differentiate('z', out=dvgx0)
vgy0['g'] = vgy
vgy0.differentiate('z', out=dvgy0)
vdx0['g'] = vdx
vdx0.differentiate('z', out=dvdx0)
vdy0['g'] = vdy
vdy0.differentiate('z', out=dvdy0)

vgx0.set_scales(1, keep_data=True)
dvgx0.set_scales(1, keep_data=True)
vgy0.set_scales(1, keep_data=True)
dvgy0.set_scales(1, keep_data=True)
vdx0.set_scales(1, keep_data=True)
dvdx0.set_scales(1, keep_data=True)
vdy0.set_scales(1, keep_data=True)
dvdy0.set_scales(1, keep_data=True)



'''
constant parameters
'''
waves.parameters['delta0']      = delta0

'''
non-constant coefficients
'''
waves.parameters['epsilon0']      = epsilon0
waves.parameters['depsilon0']     = depsilon0
waves.parameters['dln_epsilon0']  = dln_epsilon0
waves.parameters['d2epsilon0']    = d2epsilon0

waves.parameters['dln_rhog0']      = dln_rhog0
waves.parameters['dln_rhod0']      = dln_rhod0

waves.parameters['vdx0']           = vdx0

waves.parameters['vdz0']           = vdz0
waves.parameters['dvdz0']          = dvdz0



waves.parameters['kx']      = kx

#W is gas, Q is dust 

waves.substitutions['delta_rhod_over_rhod'] = "Q_p + dln_rhod0*Q"
waves.substitutions['delta_rhog_over_rhog'] = "W_p + dln_rhog0*W"
waves.substitutions['delta_ln_eps'] = "Q - W"
waves.substitutions['delta_ln_eps_p'] = "Q_p - W_p"
waves.substitutions['delta_ln_eps_pp'] = "dz(Q_p) - dz(W_p)"
waves.substitutions['delta_eps_p_over_eps']="dln_epsilon0*delta_ln_eps + delta_ln_eps_p"
waves.substitutions['delta_eps_pp_over_eps'] = "d2epsilon0*delta_ln_eps/epsilon0 + 2*dln_epsilon0*delta_ln_eps_p + delta_ln_eps_pp"

#dust continuity equation
waves.substitutions['dust_mass_LHS']="sigma*Q + 1j*kx*(Udx + vdx0*Q) + dln_rhod0*Udz + Udz_p  + vdz0*delta_rhod_over_rhod + dvdz0*Q"
waves.substitutions['dust_mass_RHS']="delta0*(dln_epsilon0*delta_rhog_over_rhog + d2epsilon0*W/epsilon0 - kx*kx*delta_ln_eps + dln_rhog0*delta_eps_p_over_eps + delta_eps_pp_over_eps)"
waves.add_equation("dust_mass_LHS - dust_mass_RHS = 0 ")







waves.parameters['k'] = 1





waves.substitutions['dt(A)'] = '1j*omega*A'
waves.substitutions['dx(A)'] = '-1j*k*A'
waves.substitutions['Div_u'] = 'dx(u) + w_z'
logger.debug("Setting z-momentum equation")
waves.add_equation("dt(w) + dz(T1) + T0*dz(ln_rho1) + T1*del_ln_rho0 = 0 ")
logger.debug("Setting x-momentum equation")
waves.add_equation("dt(u) + dx(T1) + T0*dx(ln_rho1)                  = 0 ")
logger.debug("Setting continuity equation")
waves.add_equation("dt(ln_rho1) + w*del_ln_rho0 + Div_u  = 0 ")
logger.debug("Setting energy equation")
waves.add_equation("dt(T1) + w*T0_z + (gamma-1)*T0*Div_u = 0 ")
waves.add_equation("dz(w) - w_z = 0 ")
#waves.add_bc('left(dz(u)) = 0')
#waves.add_bc('right(dz(u)) = 0')
waves.add_bc('left(w) = 0')
waves.add_bc('right(w) = 0')

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
