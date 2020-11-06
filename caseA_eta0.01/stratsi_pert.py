"""
stratified linear analysis of the streaming instability
"""

from stratsi_params import *
from eigenproblem import Eigenproblem

'''
output control
'''
nz_out   = 1024
out_scale= nz_out/nz_waves

'''
read in background vertical profiles of vgx, vgy, vdx, vdy
'''

horiz_eqm  = h5py.File('./eqm_horiz.h5', 'r')
zaxis_horiz= horiz_eqm['z'][:]
vgx        = horiz_eqm['vgx'][:]
vgy        = horiz_eqm['vgy'][:]
vdx        = horiz_eqm['vdx'][:]
vdy        = horiz_eqm['vdy'][:]
horiz_eqm.close()

'''
setup domain and calculate derivatives of vertical profiles as needed 
'''

z_basis = de.Chebyshev('z', nz_waves, interval=(zmin,zmax))
domain_EVP = de.Domain([z_basis], comm=MPI.COMM_SELF)

'''
the linear problem 

W = delta_rhog/rhog = delta_ln_rhog, Q=delta_eps/eps = delta_ln_eps, U = velocities 
W_p = W_primed (dW/dz)...etc
'''

if (viscosity_pert == True) and (diffusion == True):#full problem: include viscosity and particle diffusion
    waves = de.EVP(domain_EVP, ['W','Ugx','Ugx_p','Ugy','Ugy_p','Ugz','Q','Q_p','Udx','Udy','Udz'], eigenvalue='sigma',tolerance=tol)

if (viscosity_pert == True) and (diffusion == False):#include viscosity but no particle diffusion
    waves = de.EVP(domain_EVP, ['W','Ugx','Ugx_p','Ugy','Ugy_p','Ugz','Q','Udx','Udy','Udz'], eigenvalue='sigma',tolerance=tol)
    
if (viscosity_pert == False) and (diffusion == True):#ignore gas viscosity but include particle diffusion 
    waves = de.EVP(domain_EVP, ['W','Ugx','Ugy','Ugz','Q','Q_p','Udx','Udy','Udz'], eigenvalue='sigma',tolerance=tol)
    
if (viscosity_pert == False) and (diffusion == False):#ignore gas viscosity and ignore diffusion  
    waves = de.EVP(domain_EVP, ['W','Ugx','Ugy','Ugz','Q','Udx','Udy','Udz'], eigenvalue='sigma',tolerance=tol)


'''
constant parameters
'''
waves.parameters['delta']      = delta
waves.parameters['alpha']      = alpha
waves.parameters['eta_hat']      = eta_hat
waves.parameters['inv_stokes'] = 1.0/stokes 
waves.parameters['kx']         = kx

'''
non-constant coefficients I: epsilon, rhod, rhog, vdz 
'''

z = domain_EVP.grid(0)

dln_rhog0  = domain_EVP.new_field()
d2ln_rhog0 = domain_EVP.new_field()

dln_rhod0  = domain_EVP.new_field()

epsilon0   = domain_EVP.new_field()
depsilon0  = domain_EVP.new_field()
d2epsilon0 = domain_EVP.new_field()

dln_epsilon0= domain_EVP.new_field()

vdz0        = domain_EVP.new_field()
dvdz0       = domain_EVP.new_field()

dln_rhog0['g']    = dln_rhog(z)
d2ln_rhog0['g']   = d2ln_rhog(z)

dln_rhod0['g']    = dln_rhod(z)

epsilon0['g']     = epsilon(z)
depsilon0['g']    = depsilon(z)
d2epsilon0['g']   = d2epsilon(z)

dln_epsilon0['g'] = dln_epsilon(z) 

vdz0['g']         = vdz(z)
dvdz0['g']        = dvdz(z)

waves.parameters['dln_rhog0']      = dln_rhog0
waves.parameters['d2ln_rhog0']     = d2ln_rhog0

waves.parameters['dln_rhod0']      = dln_rhod0

waves.parameters['epsilon0']      = epsilon0
waves.parameters['depsilon0']     = depsilon0
waves.parameters['d2epsilon0']    = d2epsilon0

waves.parameters['dln_epsilon0']  = dln_epsilon0

waves.parameters['vdz0']           = vdz0
waves.parameters['dvdz0']          = dvdz0

'''
non-constant coefficients II: vgx, vgy, vdx, vdy
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

scale = nz_vert/nz_waves

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

waves.parameters['vdx0']             = vdx0
waves.parameters['dvdx0']            = dvdx0

waves.parameters['vdy0']             = vdy0
waves.parameters['dvdy0']            = dvdy0

waves.parameters['vgx0']              = vgx0
waves.parameters['dvgx0']             = dvgx0
waves.parameters['d2vgx0']            = d2vgx0

waves.parameters['vgy0']              = vgy0
waves.parameters['dvgy0']             = dvgy0
waves.parameters['d2vgy0']            = d2vgy0


# W is (delta_rhog)/rhog, Q is (delta_epsilon)/epsilon  

waves.substitutions['ikx'] = "1j*kx"
waves.substitutions['delta_ln_rhod'] = "Q + W"

if diffusion == True:
    waves.substitutions['dQ'] = "Q_p"
    waves.substitutions['delta_eps_p_over_eps'] = "dln_epsilon0*Q + Q_p"
    waves.substitutions['delta_eps_pp_over_eps'] = "d2epsilon0*Q/epsilon0 + 2*dln_epsilon0*Q_p + dz(Q_p)"
if diffusion == False:
    waves.substitutions['dQ'] = "dz(Q)"
    
waves.substitutions['delta_ln_rhod_p'] = "dQ + dz(W)"
waves.substitutions['delta_ln_taus'] = "0"

#dust continuity equation
waves.substitutions['dust_mass_LHS']="sigma*delta_ln_rhod + ikx*(Udx + vdx0*delta_ln_rhod) + dln_rhod0*(Udz + vdz0*delta_ln_rhod) + vdz0*delta_ln_rhod_p + dvdz0*delta_ln_rhod + dz(Udz)"
if diffusion == True:
    waves.substitutions['dust_mass_RHS']="delta*(dln_epsilon0*(dln_rhog0*W + dz(W)) + d2epsilon0*W/epsilon0 - kx*kx*Q + dln_rhog0*delta_eps_p_over_eps + delta_eps_pp_over_eps)"
if diffusion == False:
    waves.substitutions['dust_mass_RHS']="0"

#dust x-mom equation
waves.substitutions['dust_xmom_LHS']="sigma*Udx + dvdx0*Udz + ikx*vdx0*Udx + vdz0*dz(Udx)"
waves.substitutions['dust_xmom_RHS']="2*Udy + inv_stokes*delta_ln_taus*(vdx0 - vgx0) - inv_stokes*(Udx - Ugx)"

#dust y-mom equation
waves.substitutions['dust_ymom_LHS']="sigma*Udy + dvdy0*Udz + ikx*vdx0*Udy + vdz0*dz(Udy)"
waves.substitutions['dust_ymom_RHS']="-0.5*Udx + inv_stokes*delta_ln_taus*(vdy0 - vgy0) - inv_stokes*(Udy - Ugy)"

#dust z-mom
waves.substitutions['dust_zmom_LHS']="sigma*Udz + dvdz0*Udz + ikx*vdx0*Udz + vdz0*dz(Udz)"
waves.substitutions['dust_zmom_RHS']="inv_stokes*delta_ln_taus*vdz0 - inv_stokes*(Udz - Ugz)"

#gas continuity equation
waves.substitutions['gas_mass_LHS']="sigma*W + ikx*(Ugx + vgx0*W) + dln_rhog0*Ugz + dz(Ugz)"

#linearized viscous forces on gas
#could also use eqm eqns to replace first bracket, so that we don't need to take numerical derivs of vgx..etc
if viscosity_pert == True:
    waves.substitutions['delta_vgz_pp']="-( sigma*dz(W) + ikx*(Ugx_p + dvgx0*W + vgx0*dz(W)) + d2ln_rhog0*Ugz + dln_rhog0*dz(Ugz) )" #take deriv of gas mass eq to get d2(delta_vgz)/dz2
    waves.substitutions['delta_Fvisc_x'] = "alpha*(ikx*dz(Ugz)/3 - 4*kx*kx*Ugx/3 + dln_rhog0*(ikx*Ugz + Ugx_p) + dz(Ugx_p)) - alpha*(dln_rhog0*dvgx0 + d2vgx0)*W"
    waves.substitutions['delta_Fvisc_y'] = "alpha*(dln_rhog0*Ugy_p - kx*kx*Ugy + dz(Ugy_p)) - alpha*(dln_rhog0*dvgy0 + d2vgy0)*W"
    waves.substitutions['delta_Fvisc_z'] = "alpha*(ikx*Ugx_p/3 - kx*kx*Ugz + dln_rhog0*(4*dz(Ugz)/3 - 2*ikx*Ugx/3) + 4*delta_vgz_pp/3)"
if viscosity_pert == False:
    waves.substitutions['delta_Fvisc_x'] = "0"
    waves.substitutions['delta_Fvisc_y'] = "0"
    waves.substitutions['delta_Fvisc_z'] = "0"
    
#linearized back-reaction force on gas
if backreaction == True:
    waves.substitutions['delta_backreaction_x']="inv_stokes*epsilon0*( (vgx0 - vdx0)*(delta_ln_taus - Q) - (Ugx - Udx) )"
    waves.substitutions['delta_backreaction_y']="inv_stokes*epsilon0*( (vgy0 - vdy0)*(delta_ln_taus - Q) - (Ugy - Udy) )"
    waves.substitutions['delta_backreaction_z']="inv_stokes*epsilon0*( (     - vdz0)*(delta_ln_taus - Q) - (Ugz - Udz) )"
if backreaction == False:
    waves.substitutions['delta_backreaction_x']="0"
    waves.substitutions['delta_backreaction_y']="0"
    waves.substitutions['delta_backreaction_z']="0"
    
#gas equations
waves.add_equation("gas_mass_LHS = 0 ")
waves.add_equation("sigma*Ugx + dvgx0*Ugz + ikx*vgx0*Ugx - 2*Ugy + ikx*W - delta_backreaction_x - delta_Fvisc_x = 0")
waves.add_equation("sigma*Ugy + dvgy0*Ugz + ikx*vgx0*Ugy + 0.5*Ugx - delta_backreaction_y - delta_Fvisc_y = 0")
waves.add_equation("sigma*Ugz + ikx*vgx0*Ugz + dz(W) - delta_backreaction_z - delta_Fvisc_z = 0")

#dust equations 
waves.add_equation("dust_mass_LHS - dust_mass_RHS = 0 ")
waves.add_equation("dust_xmom_LHS - dust_xmom_RHS = 0 ")
waves.add_equation("dust_ymom_LHS - dust_ymom_RHS = 0 ")
waves.add_equation("dust_zmom_LHS - dust_zmom_RHS = 0")

#equations for first derivs of perts, i.e. dz(Q) = Q_p ...etc
#in our formulation of second derivs of [epsilon, delta_vgx, delta_vgy] appear for full problem with viscosity
if diffusion == True:
    waves.add_equation("dz(Q) - Q_p = 0")
if viscosity_pert == True:
    waves.add_equation("dz(Ugx) - Ugx_p = 0")
    waves.add_equation("dz(Ugy) - Ugy_p = 0")

'''
boundary conditions (reflection)
'''
#mid-plane symmetry conditions on center-of-mass velocities, as in one-fluid case 
waves.add_bc('left(dz(W))=0')
waves.add_bc('left(dQ)=0')
waves.add_bc('left(dz(Ugx - epsilon0*vgx0*Q/(1+epsilon0) + epsilon0*Udx + epsilon0*vdx0*Q/(1+epsilon0)))=0')
waves.add_bc('left(dz(Ugy - epsilon0*vgy0*Q/(1+epsilon0) + epsilon0*Udy + epsilon0*vdy0*Q/(1+epsilon0)))=0')
waves.add_bc('left(Ugz + epsilon0*Udz)=0')

waves.add_bc('right(dz(W))=0')

if diffusion == True:
    waves.add_bc('right(Q)=0')

if viscosity_pert == True:
    waves.add_bc('left(Ugx_p)=0')
    waves.add_bc('left(Ugy_p)=0')
    waves.add_bc('right(Ugx_p)=0')
    waves.add_bc('right(Ugy_p)=0')

'''
eigenvalue problem, sweep through kx space
for each kx, filter modes and keep most unstable one
'''

EP_list = [Eigenproblem(waves), Eigenproblem(waves, sparse=True)] 
kx_space = np.logspace(np.log10(kx_min),np.log10(kx_max), num=nkx)

eigenfreq = []
eigenfunc = {'W':[], 'Q':[], 'Ugx':[], 'Ugy':[], 'Ugz':[], 'Udx':[], 'Udy':[], 'Udz':[]}

for i, kx in enumerate(kx_space):
    
    if ((i == 0) and (first_solve_dense == True)) or all_solve_dense == True:
            EP = EP_list[0]
    else:
            EP = EP_list[1]
    
    EP.EVP.namespace['kx'].value = kx
    EP.EVP.parameters['kx'] = kx

    if all_solve_dense == True:
        EP.solve()
    else:
        if i == 0:
            if first_solve_dense == True:
                EP.solve()
            else:
                trial = eigen_trial
                EP.solve(N=Neig, target = trial)
        else:
            #trial = eigenfreq[i-1]
            EP.solve(N=Neig, target = trial)

    EP.reject_spurious()

    abs_sig = np.abs(EP.evalues_good)
    sig_acceptable = (abs_sig < sig_filter) & (EP.evalues_good.real > 0.0)

    sigma      = EP.evalues_good[sig_acceptable]
    sigma_index= EP.evalues_good_index[sig_acceptable]

    eigenfunc['W'].append([])
    eigenfunc['Q'].append([])
    eigenfunc['Ugx'].append([])
    eigenfunc['Ugy'].append([])
    eigenfunc['Ugz'].append([])
    eigenfunc['Udx'].append([])
    eigenfunc['Udy'].append([])
    eigenfunc['Udz'].append([])

    if sigma.size > 0:
        eigenfreq.append(sigma)
        for n, mode in enumerate(sigma_index):
            EP.solver.set_state(mode)
        
            W  = EP.solver.state['W']
            Q  = EP.solver.state['Q']
            Ugx = EP.solver.state['Ugx']
            Ugy = EP.solver.state['Ugy']
            Ugz = EP.solver.state['Ugz']
            Udx = EP.solver.state['Udx']
            Udy = EP.solver.state['Udy']
            Udz = EP.solver.state['Udz']

            W.set_scales(scales=out_scale)
            Q.set_scales(scales=out_scale)
            Ugx.set_scales(scales=out_scale)
            Ugy.set_scales(scales=out_scale)
            Ugz.set_scales(scales=out_scale)
            Udx.set_scales(scales=out_scale)
            Udy.set_scales(scales=out_scale)
            Udz.set_scales(scales=out_scale)
            
            eigenfunc['W'][i].append(np.copy(W['g']))
            eigenfunc['Q'][i].append(np.copy(Q['g']))
            eigenfunc['Ugx'][i].append(np.copy(Ugx['g'])) 
            eigenfunc['Ugy'][i].append(np.copy(Ugy['g'])) 
            eigenfunc['Ugz'][i].append(np.copy(Ugz['g'])) 
            eigenfunc['Udx'][i].append(np.copy(Udx['g']))
            eigenfunc['Udy'][i].append(np.copy(Udy['g'])) 
            eigenfunc['Udz'][i].append(np.copy(Udz['g']))             
    else:
        eigenfreq.append(np.array([np.nan]))
        eigenfunc['W'][i].append(np.zeros(nz_out))
        eigenfunc['Q'][i].append(np.zeros(nz_out))
        eigenfunc['Ugx'][i].append(np.zeros(nz_out)) 
        eigenfunc['Ugy'][i].append(np.zeros(nz_out)) 
        eigenfunc['Ugz'][i].append(np.zeros(nz_out)) 
        eigenfunc['Udx'][i].append(np.zeros(nz_out))
        eigenfunc['Udy'][i].append(np.zeros(nz_out)) 
        eigenfunc['Udz'][i].append(np.zeros(nz_out))    


    growth = eigenfreq[i].real
    freq   = eigenfreq[i].imag
    g1     =  np.argmax(growth)
    trial  = eigenfreq[i][g1]
        
'''
print results to screen (most unstable mode)
'''
for i, kx in enumerate(kx_space):
    growth = eigenfreq[i].real
    g1     =  np.argmax(growth)
    print("i, kx, growth, freq = {0:3d} {1:1.2e} {2:9.6f} {3:9.6f}".format(i, kx, eigenfreq[i][g1].real, -eigenfreq[i][g1].imag))

'''
data output
'''

with h5py.File('stratsi_modes.h5','w') as outfile:
    z_out = domain_EVP.grid(0, scales=out_scale)

    scale_group = outfile.create_group('scales')
    scale_group.create_dataset('kx_space',data=kx_space)
    scale_group.create_dataset('z',   data=z_out)
    scale_group.create_dataset('zmax', data=zmax)

    tasks_group = outfile.create_group('tasks')

    
    for i, freq in enumerate(eigenfreq):
        data_group = tasks_group.create_group('k_{:03d}'.format(i))
        data_group.create_dataset('freq',data=freq)
        data_group.create_dataset('eig_W',data=eigenfunc['W'][i])
        data_group.create_dataset('eig_Q',data=eigenfunc['Q'][i])
        data_group.create_dataset('eig_Ugx',data=eigenfunc['Ugx'][i])
        data_group.create_dataset('eig_Ugy',data=eigenfunc['Ugy'][i])
        data_group.create_dataset('eig_Ugz',data=eigenfunc['Ugz'][i])
        data_group.create_dataset('eig_Udx',data=eigenfunc['Udx'][i])
        data_group.create_dataset('eig_Udy',data=eigenfunc['Udy'][i])
        data_group.create_dataset('eig_Udz',data=eigenfunc['Udz'][i])
    outfile.close()


