import sys
import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
import h5py
import argparse
from scipy.integrate import simps

#custom_preamble = {
#    "text.usetex": True,
#    "text.latex.preamble": [
#        r"\usepackage{amsmath}", # for the align enivironment
#        ],
#    }


from stratsi_params import alpha, delta, stokes, metal, epsilon, vdz, dvdz, rhog, dln_rhog, viscosity_pert
from stratsi_1fluid import epsilon_eqm, dvx_eqm, dvy_eqm, vz_eqm, dvz_eqm, eta_hat, P_eqm#, z_maxvshear, maxvshear

'''
process command line arguements
'''
parser = argparse.ArgumentParser()
parser.add_argument("--mode", nargs='*', help="select mode number")
parser.add_argument("--kx", nargs='*', help="select kx")
parser.add_argument("--sig", nargs='*', help="select eigenvalue")

args = parser.parse_args()
if(args.mode):
    plot_mode = np.int(args.mode[0])
else:
    plot_mode = 0

if(args.kx):
    plot_kx = np.float(args.kx[0])
else:
    plot_kx = 400.0
    
if(args.sig):
    eigenv = np.float(args.sig[0]) - np.float(args.sig[1])*1j
else:
    eigenv = 1.0

    
#print("plotting mode number {0:3d}".format(plot_mode))
    
'''
read in one-fluid data 
'''

with h5py.File('stratsi_1fluid_modes.h5','r') as infile:
  
  ks_1f    = infile['scales']['kx_space'][:]
  z_1f     = infile['scales']['z'][:]
  zmax_1f  = infile['scales']['zmax'][()]

  freqs_1f= []
  eig_W1f = []
  eig_Q1f = []
  eig_Ux= []
  eig_Uy= []
  eig_Uz= []

  for k_i in infile['tasks']:
    freqs_1f.append(infile['tasks'][k_i]['freq'][:])
    eig_W1f.append(infile['tasks'][k_i]['eig_W'][:])
    eig_Q1f.append(infile['tasks'][k_i]['eig_Q'][:])
    eig_Ux.append(infile['tasks'][k_i]['eig_Ux'][:])
    eig_Uy.append(infile['tasks'][k_i]['eig_Uy'][:])
    eig_Uz.append(infile['tasks'][k_i]['eig_Uz'][:])

if(args.mode):
    n = plot_mode

if(args.kx):
    n = np.argmin(np.abs(ks_1f-plot_kx))
    
kx_1f       = ks_1f[n]
sigma_1f    = freqs_1f[n]

growth_1f   = sigma_1f.real
freq_1f     =-sigma_1f.imag

if(args.sig):
    g1 = np.argmin(np.abs(sigma_1f-eigenv))
else:
    g1          = np.argmax(growth_1f)

sgrow_1f = growth_1f[g1]
ofreq_1f = freq_1f[g1]
print("one-fluid model: kx, growth, freq = {0:1.6e} {1:13.6e} {2:13.6e}".format(kx_1f, sgrow_1f, ofreq_1f))

W1f         = np.array(eig_W1f[n][g1])
Q1f         = np.array(eig_Q1f[n][g1])
Ux          = np.array(eig_Ux[n][g1])
Uy          = np.array(eig_Uy[n][g1])
Uz          = np.array(eig_Uz[n][g1])

#normalize eigenfunctions such that at delta rhod/rhod = W+Q is unity (and real) at its maximum
g2      = np.argmax(np.abs(W1f + Q1f))
norm_1f = W1f[g2] + Q1f[g2]

W1f /= norm_1f
Q1f /= norm_1f
Ux  /= norm_1f
Uy  /= norm_1f
Uz  /= norm_1f

del_rhod1f = W1f + Q1f 

'''
analysis of energy profiles of the most unstable mode based on 1 fluid results 
'''

eps1f = epsilon_eqm(z_1f)
dvx1f = dvx_eqm(z_1f)
dvy1f = dvy_eqm(z_1f)
vz1f  = vz_eqm(z_1f)
dvz1f = dvz_eqm(z_1f)
rho1f = P_eqm(z_1f) #divided by cs^2, but cs=1

del_rho1f = W1f + eps1f*Q1f/(1.0 + eps1f)

dW1f= np.gradient(W1f, z_1f) 
dUx = np.gradient(Ux, z_1f)
dUy = np.gradient(Uy, z_1f)
dUz = np.gradient(Uz, z_1f)

energy1f_tot = (np.abs(Ux)**2 + 4*np.abs(Uy)**2 + np.abs(Uz)**2)

energy1f_A1 = -(dvx1f*np.real(Uz*np.conj(Ux)))
energy1f_A2 = -(4.0*dvy1f*np.real(Uz*np.conj(Uy)))
energy1f_A3 = -(dvz1f*np.abs(Uz)**2)
energy1f_A = energy1f_A1 + energy1f_A2 + energy1f_A3
energy1f_B = -vz1f*np.real(dUx*np.conj(Ux) + 4.0*dUy*np.conj(Uy) + dUz*np.conj(Uz))
energy1f_C = (kx_1f*np.imag(W1f*np.conj(Ux)) - np.real(dW1f*np.conj(Uz)))/(1.0 + eps1f)
energy1f_D = -2.0*eta_hat*np.real(Q1f*np.conj(Ux))*eps1f/(1.0 + eps1f)/(1.0 + eps1f)
energy1f_E = -z_1f*eps1f*np.real(Q1f*np.conj(Uz))/(1.0 + eps1f)


energy1f_A2 /= sgrow_1f
energy1f_A  /= sgrow_1f
energy1f_B  /= sgrow_1f
energy1f_C  /= sgrow_1f
energy1f_D  /= sgrow_1f
energy1f_E  /= sgrow_1f

'''
compare integrated energetics for the most unstable mode (at each kx) based on 1 fluid result
'''

energy1f_tot_int  =[]
energy1f_A_int    =[]
energy1f_A2_int   =[]
energy1f_B_int    =[]
energy1f_C_int    =[]
energy1f_D_int    =[]
energy1f_E_int    =[]

for i, kx1f in enumerate(ks_1f):
    g3          = np.argmax(freqs_1f[i].real)
    s1f         = np.amax(freqs_1f[i].real)
        
    w1f         = np.array(eig_W1f[i][g3])
    q1f         = np.array(eig_Q1f[i][g3])
    ux          = np.array(eig_Ux[i][g3])
    uy          = np.array(eig_Uy[i][g3])
    uz          = np.array(eig_Uz[i][g3])

    g4      = np.argmax(np.abs(w1f + q1f))
    norm_1f = w1f[g4] + q1f[g4]

    w1f /= norm_1f
    q1f /= norm_1f
    ux  /= norm_1f
    uy  /= norm_1f
    uz  /= norm_1f
    
    dw1f= np.gradient(w1f, z_1f) 
    dux = np.gradient(ux, z_1f)
    duy = np.gradient(uy, z_1f)
    duz = np.gradient(uz, z_1f)

    e1f_tot = simps(rho1f*(np.abs(ux)**2 + 4*np.abs(uy)**2 + np.abs(uz)**2), z_1f)

    e1f_A1 = simps(-(dvx1f*np.real(uz*np.conj(ux)))*rho1f, z_1f)
    e1f_A2 = simps(-(4.0*dvy1f*np.real(uz*np.conj(uy)))*rho1f, z_1f)
    e1f_A3 = simps(-(dvz1f*np.abs(uz)**2)*rho1f, z_1f)
    e1f_A = e1f_A1 + e1f_A2 + e1f_A3
    e1f_B = simps(-vz1f*np.real(dux*np.conj(ux) + 4.0*duy*np.conj(uy) + duz*np.conj(uz))*rho1f, z_1f)
    e1f_C = simps((kx1f*np.imag(w1f*np.conj(ux)) - np.real(dw1f*np.conj(uz)))/(1.0 + eps1f)*rho1f, z_1f)
    e1f_D = simps(-2.0*eta_hat*np.real(q1f*np.conj(ux))*eps1f/(1.0 + eps1f)/(1.0 + eps1f)*rho1f, z_1f)
    e1f_E = simps(-z_1f*eps1f*np.real(q1f*np.conj(uz))/(1.0 + eps1f)*rho1f, z_1f)

    e1f_A /= s1f
    e1f_A2/= s1f
    e1f_B /= s1f
    e1f_C /= s1f
    e1f_D /= s1f
    e1f_E /= s1f

    energy1f_tot_int.append(e1f_tot)
    energy1f_A_int.append(e1f_A)
    energy1f_A2_int.append(e1f_A2)
    energy1f_B_int.append(e1f_B)
    energy1f_C_int.append(e1f_C)
    energy1f_D_int.append(e1f_D)
    energy1f_E_int.append(e1f_E)
    

'''
read in two-fluid data
'''

with h5py.File('stratsi_modes.h5','r') as infile:
  
  ks    = infile['scales']['kx_space'][:]
  z     = infile['scales']['z'][:]
  zmax  = infile['scales']['zmax'][()]

  freqs = []
  eig_W = []
  eig_Q = []
  eig_Ugx= []
  eig_Ugy= []
  eig_Ugz= []
  eig_Udx= []
  eig_Udy= []
  eig_Udz= []
  
  for k_i in infile['tasks']:
    freqs.append(infile['tasks'][k_i]['freq'][:])
    eig_W.append(infile['tasks'][k_i]['eig_W'][:])
    eig_Q.append(infile['tasks'][k_i]['eig_Q'][:])
    eig_Ugx.append(infile['tasks'][k_i]['eig_Ugx'][:])
    eig_Ugy.append(infile['tasks'][k_i]['eig_Ugy'][:])
    eig_Ugz.append(infile['tasks'][k_i]['eig_Ugz'][:])
    eig_Udx.append(infile['tasks'][k_i]['eig_Udx'][:])
    eig_Udy.append(infile['tasks'][k_i]['eig_Udy'][:])
    eig_Udz.append(infile['tasks'][k_i]['eig_Udz'][:])


if(args.mode):
    m = plot_mode

if(args.kx):
    m = np.argmin(np.abs(ks-plot_kx))

kx       = ks[m]
sigma    = freqs[m]

growth      = sigma.real
freq        =-sigma.imag

if(args.sig):
    g1 = np.argmin(np.abs(sigma-eigenv))
else:
    g1          = np.argmax(growth)

sgrow = growth[g1]
ofreq = freq[g1]
print("two-fluid model: kx, growth, freq = {0:1.6e} {1:13.6e} {2:13.6e}".format(kx, sgrow, ofreq))
    
W         =  np.array(eig_W[m][g1])
Q         =  np.array(eig_Q[m][g1])
Ugx          =  np.array(eig_Ugx[m][g1])
Ugy          =  np.array(eig_Ugy[m][g1])
Ugz          =  np.array(eig_Ugz[m][g1])
Udx          =  np.array(eig_Udx[m][g1])
Udy          =  np.array(eig_Udy[m][g1])
Udz          =  np.array(eig_Udz[m][g1])

g2      = np.argmax(np.abs(W+Q))
norm    = W[g2] + Q[g2]

W   /= norm
Q   /= norm
Ugx /= norm
Ugy /= norm
Ugz /= norm
Udx /= norm
Udy /= norm
Udz /= norm

del_rhod = W + Q

'''
energy analysis
'''
#read in background vertical profiles of vgx, vgy, vdx, vdy

horiz_eqm  = h5py.File('./eqm_horiz.h5', 'r')
vgx        = horiz_eqm['vgx'][:]
vgy        = horiz_eqm['vgy'][:]
vdx        = horiz_eqm['vdx'][:]
vdy        = horiz_eqm['vdy'][:]
horiz_eqm.close()

dvgx = np.gradient(vgx, z)
dvgy = np.gradient(vgy, z)
dvdx = np.gradient(vdx, z)
dvdy = np.gradient(vdy, z)

d2vgx = np.gradient(dvgx, z)
d2vgy = np.gradient(dvgy, z)
d2vdx = np.gradient(dvdx, z)
d2vdy = np.gradient(dvdy, z)

eps2f  = epsilon(z)

dW   = np.gradient(W, z)

dUgx = np.gradient(Ugx, z)
dUgy = np.gradient(Ugy, z)
dUgz = np.gradient(Ugz, z)

d2Ugx = np.gradient(dUgx, z)
d2Ugy = np.gradient(dUgy, z)
d2Ugz = np.gradient(dUgz, z)

dUdx = np.gradient(Udx, z)
dUdy = np.gradient(Udy, z)
dUdz = np.gradient(Udz, z)

energy2f_tot = eps2f*(np.abs(Udx)**2 + 4.0*np.abs(Udy)**2 + np.abs(Udz)**2)
energy2f_tot+= (np.abs(Ugx)**2 + 4.0*np.abs(Ugy)**2 + np.abs(Ugz)**2)
#energy2f_tot/= (1.0 + eps2f)

energy2f_A   =-eps2f*np.real(Udz*np.conj(dvdx*Udx + 4.0*dvdy*Udy + dvdz(z)*Udz))
energy2f_A  +=-np.real(Ugz*np.conj(dvgx*Ugx + 4.0*dvgy*Ugy))
energy2f_A2  = -eps2f*np.real(Udz*np.conj(4.0*dvdy*Udy)) - np.real(Ugz*np.conj(4.0*dvgy*Ugy))
energy2f_B   =-eps2f*vdz(z)*np.real(dUdx*np.conj(Udx) + 4.0*dUdy*np.conj(Udy) + dUdz*np.conj(Udz))
energy2f_C   = kx*np.imag(W*np.conj(Ugx)) - np.real(dW*np.conj(Ugz))
energy2f_D   = (vgx - vdx)*np.real(Q*np.conj(Ugx)) + 4.0*(vgy - vdy)*np.real(Q*np.conj(Ugy))
energy2f_D  += np.abs(Ugx - Udx)**2 + 4.0*np.abs(Ugy - Udy)**2 + np.abs(Ugz - Udz)**2
energy2f_D  *= -eps2f/stokes

energy2f_E = (eps2f/stokes)*vdz(z)*np.real(Q*np.conj(Ugz))#buoyancy in 2fluid

if viscosity_pert == True:
    dFx = d2Ugx - (4.0/3.0)*kx*kx*Ugx + (1.0/3.0)*1j*kx*dUgz + dln_rhog(z)*(dUgx + 1j*kx*Ugz)
    dFx-= W*(d2vgx + dln_rhog(z)*dvgx)
    dFx*= alpha

    dFy = d2Ugy - kx*kx*Ugy + dln_rhog(z)*dUgy 
    dFy-= W*(d2vgy + dln_rhog(z)*dvgy)
    dFy*= alpha

    dFz = (4.0/3.0)*d2Ugz - kx*kx*Ugz + (1.0/3.0)*1j*kx*dUgx + dln_rhog(z)*((4.0/3.0)*dUgz - (2.0/3.0)*1j*kx*Ugx)
    dFz*= alpha
    
    energy2f_F = np.real(dFx*np.conj(Ugx) + 4.0*dFy*np.conj(Ugy) + dFz*np.conj(Ugz))
else:
    energy2f_F = np.zeros(z.size)

energy2f_A  /= sgrow#*(1.0 + eps2f)
energy2f_A2 /= sgrow#*(1.0 + eps2f)
energy2f_B  /= sgrow#*(1.0 + eps2f)
energy2f_C  /= sgrow#*(1.0 + eps2f)
energy2f_D  /= sgrow#*(1.0 + eps2f)
energy2f_E   /= sgrow#*(1.0 + eps2f)
energy2f_F   /= sgrow#*(1.0 + eps2f)

'''
compare integrated energetics for the most unstable mode (at each kx) based on 2 fluid result
'''
energy2f_tot_int  =[]
energy2f_A_int    =[]
energy2f_A2_int   =[]
energy2f_B_int    =[]
energy2f_C_int    =[]
energy2f_D_int    =[]
energy2f_E_int    =[]
energy2f_F_int    =[]

for i, kx2f in enumerate(ks):
    g3          = np.argmax(freqs[i].real)
    s2f         = np.amax(freqs[i].real)

    w         =  np.array(eig_W[i][g3])
    q         =  np.array(eig_Q[i][g3])
    ugx       =  np.array(eig_Ugx[i][g3])
    ugy       =  np.array(eig_Ugy[i][g3])
    ugz       =  np.array(eig_Ugz[i][g3])
    udx       =  np.array(eig_Udx[i][g3])
    udy       =  np.array(eig_Udy[i][g3])
    udz       =  np.array(eig_Udz[i][g3])

    g4 = np.argmax(np.abs(w + q))
    norm    = w[g4] + q[g4]

    w   /= norm
    q   /= norm
    ugx /= norm
    ugy /= norm
    ugz /= norm
    udx /= norm
    udy /= norm
    udz /= norm
    
    dw   = np.gradient(w, z) 
    dugx = np.gradient(ugx, z)
    dugy = np.gradient(ugy, z)
    dugz = np.gradient(ugz, z)

    d2ugx= np.gradient(dugx, z)
    d2ugy= np.gradient(dugy, z)
    d2ugz= np.gradient(dugz, z)

    dudx = np.gradient(udx, z)
    dudy = np.gradient(udy, z)
    dudz = np.gradient(udz, z)

    e2f_tot = simps((eps2f*(np.abs(udx)**2 + 4.0*np.abs(udy)**2 + np.abs(udz)**2) \
                         + (np.abs(ugx)**2 + 4.0*np.abs(ugy)**2 + np.abs(ugz)**2))*rhog(z), z)
    e2f_A   = simps((-eps2f*np.real(udz*np.conj(dvdx*udx + 4.0*dvdy*udy + dvdz(z)*udz)) \
                              -np.real(ugz*np.conj(dvgx*ugx + 4.0*dvgy*ugy)))*rhog(z), z)
    e2f_A2  = simps((-eps2f*np.real(udz*np.conj(4.0*dvdy*udy)) - np.real(ugz*np.conj(4.0*dvgy*ugy)))*rhog(z), z)
    e2f_B   = simps((-eps2f*vdz(z)*np.real(dudx*np.conj(udx) + 4.0*dudy*np.conj(udy) + dudz*np.conj(udz)))*rhog(z),z)
    e2f_C   = simps((kx2f*np.imag(w*np.conj(ugx)) - np.real(dw*np.conj(ugz)))*rhog(z),z)
    e2f_D   = simps(-(eps2f/stokes)*((vgx - vdx)*np.real(q*np.conj(ugx)) + 4.0*(vgy - vdy)*np.real(q*np.conj(ugy)) \
                                            + np.abs(ugx - udx)**2 + 4.0*np.abs(ugy - udy)**2 + np.abs(ugz - udz)**2)*rhog(z), z)
    e2f_E   = simps((eps2f/stokes)*vdz(z)*np.real(q*np.conj(ugz))*rhog(z),z)
    
    if viscosity_pert == True:
        dfx = d2ugx - (4.0/3.0)*kx2f*kx2f*ugx + (1.0/3.0)*1j*kx2f*dugz + dln_rhog(z)*(dugx + 1j*kx*ugz)
        dfx-= w*(d2vgx + dln_rhog(z)*dvgx)
        dfx*= alpha

        dfy = d2ugy - kx2f*kx2f*ugy + dln_rhog(z)*dugy 
        dfy-= w*(d2vgy + dln_rhog(z)*dvgy)
        dfy*= alpha

        dfz = (4.0/3.0)*d2ugz - kx2f*kx2f*ugz + (1.0/3.0)*1j*kx*dugx + dln_rhog(z)*((4.0/3.0)*dugz - (2.0/3.0)*1j*kx*ugx)
        dfz*= alpha
    
        e2f_F=simps((np.real(dfx*np.conj(ugx) + 4.0*dfy*np.conj(ugy) + dfz*np.conj(ugz)))*rhog(z), z)
    else:
        e2f_F = 0.0

    e2f_A  /= s2f
    e2f_A2 /= s2f
    e2f_B  /= s2f
    e2f_C  /= s2f
    e2f_D  /= s2f
    e2f_E  /= s2f
    e2f_F  /= s2f

    energy2f_tot_int.append(e2f_tot)
    energy2f_A_int.append(e2f_A)
    energy2f_A2_int.append(e2f_A2)
    energy2f_B_int.append(e2f_B)
    energy2f_C_int.append(e2f_C)
    energy2f_D_int.append(e2f_D)
    energy2f_E_int.append(e2f_E)
    energy2f_F_int.append(e2f_F)
       

'''
plotting parameters
'''

fontsize= 24
nlev    = 128
nclev   = 6
cmap    = plt.cm.inferno

ymax = 1
xmin = 0.0
xmax = np.amax(np.array(zmax, zmax_1f))

'''
plot eigenvalues
'''
plt.rc('font',size=fontsize,weight='bold')

fig, axs = plt.subplots(2, sharex=True, sharey=False, gridspec_kw={'hspace': 0.1}, figsize=(8,6))
#plt.subplots_adjust(left=0.18, right=0.95, top=0.95, bottom=0.15)
plt.subplots_adjust(left=0.16, right=0.95, top=0.95, bottom=0.15)
plt.xscale('log')

for i, k in enumerate(ks_1f):
    for n, sig in enumerate(freqs_1f[i]):
        if (i == 0) & (n == 0):
            lab = r'one fluid'
        else:
            lab = ''
        axs[0].plot(k, sig.real, marker='o', linestyle='none', markersize=8, label=lab,color='black')
       
for i, k in enumerate(ks):
    for n, sig in enumerate(freqs[i]):
        if (i == 0) & (n == 0):
            lab = r'two fluid'
        else:
            lab = ''
        axs[0].plot(k, sig.real, marker='X', linestyle='none', markersize=8, label=lab,color='red')

axs[0].set_ylabel(r'$s/\Omega$')
lines1, labels1 = axs[0].get_legend_handles_labels()
legend=axs[0].legend(lines1, labels1, loc='upper left', frameon=False, ncol=1, handletextpad=-0.5,fontsize=fontsize/2)

title=r"Z={0:1.2f}, St={1:4.0e}, $\delta$={2:4.0e}".format(metal, stokes, delta)
axs[0].set_title(title,weight='bold')

for i, k in enumerate(ks_1f):
    for n, sig in enumerate(freqs_1f[i]):
        if (i == 0) & (n == 0):
            lab = r'one fluid'
        else:
            lab = ''
        axs[1].plot(k, -sig.imag, marker='o', linestyle='none', markersize=8, label=lab,color='black')

for i, k in enumerate(ks):
    for n, sig in enumerate(freqs[i]):
        if (i == 0) & (n == 0):
            lab = r'two fluid'
        else:
            lab = ''
        axs[1].plot(k, -sig.imag, marker='X', linestyle='none', markersize=8, label=lab,color='red')
       
#axs[1].plot(ks, -freqs.imag, marker='X',markersize=10,linestyle='none',  label=r'two fluid')

axs[1].set_ylabel(r'$\omega/\Omega$')
axs[1].set_xlabel(r'$k_xH_g$')
lines1, labels1 = axs[1].get_legend_handles_labels()
legend=axs[1].legend(lines1, labels1, loc='upper left', frameon=False, ncol=1, handletextpad=-0.5, fontsize=fontsize/2)

plt.xlim(np.amin(ks),np.amax(ks))

fname = 'stratsi_plot_growth'
plt.savefig(fname,dpi=150)

'''
plot max growth rates as func of kx
'''

fig, axs = plt.subplots(2, sharex=True, sharey=False, gridspec_kw={'hspace': 0.1}, figsize=(8,6))

plt.subplots_adjust(left=0.2, right=0.95, top=0.9, bottom=0.15)
plt.xscale('log')


for i, k in enumerate(ks_1f):
    g1 = np.argmax(freqs_1f[i].real)
    if i == 0:
        lab = r'one-fluid'
    else:
        lab = ''
    axs[0].plot(k, freqs_1f[i][g1].real , marker='o', linestyle='none', markersize=8, label=lab,color='black')

#axs[0].axhline(y=maxvshear, linestyle='dashed', linewidth=1, label=r'$max\left|dv_y/dz\right|/\Omega$')

for i, k in enumerate(ks):
    g1 = np.argmax(freqs[i].real)
    if i == 0:
        lab = r'two-fluid'
    else:
        lab = ''
    axs[0].plot(k, freqs[i][g1].real , marker='X', linestyle='none', markersize=8, label=lab,color='red')


axs[0].set_ylabel(r'$s_\mathrm{max}/\Omega$')
lines1, labels1 = axs[0].get_legend_handles_labels()
legend=axs[0].legend(lines1, labels1, loc='upper left', frameon=False, ncol=1, handletextpad=-0.5,fontsize=fontsize/2)

title=r"Z={0:1.2f}, St={1:4.0e}, $\delta$={2:4.0e}".format(metal, stokes, delta)
axs[0].set_title(title,weight='bold')

for i, k in enumerate(ks_1f):
    g1 = np.argmax(freqs_1f[i].real)
    if i == 0:
        lab = r'one fluid'
    else:
        lab = ''
    axs[1].plot(k, -freqs_1f[i][g1].imag, marker='o', linestyle='none', markersize=8, label=lab,color='black')
    
for i, k in enumerate(ks):
    g1 = np.argmax(freqs[i].real)
    if i == 0:
        lab = r'two fluid'
    else:
        lab = ''
    axs[1].plot(k, -freqs[i][g1].imag, marker='X', linestyle='none', markersize=8, label=lab,color='red')
  
axs[1].set_ylabel(r'$\omega/\Omega$')
axs[1].set_xlabel(r'$k_xH_g$')
#lines1, labels1 = axs[1].get_legend_handles_labels()
#legend=axs[1].legend(lines1, labels1, loc='upper left', frameon=False, ncol=1, handletextpad=-0.5, fontsize=fontsize/2)

#plt.xlim(np.amin(ks),np.amax(ks))

fname = 'stratsi_plot_growth_max'
plt.savefig(fname,dpi=150)


'''
plot eigenvalues as scatter diagram for a single kx 
'''

fig = plt.figure(figsize=(8,4.5))
ax = fig.add_subplot()
plt.subplots_adjust(left=0.2, right=0.95, top=0.9, bottom=0.2)


plt.scatter(-sigma_1f.imag, sigma_1f.real, marker='o', label=r'one fluid',color='black',s=64)
plt.scatter(-sigma.imag, sigma.real, marker='X', label=r'two fluid',color='red',s=64)

plt.rc('font',size=fontsize,weight='bold')

lines1, labels1 = ax.get_legend_handles_labels()
legend=ax.legend(lines1, labels1, loc='upper right', frameon=False, ncol=1, handletextpad=-0.5, fontsize=fontsize/2)

plt.xticks(fontsize=fontsize,weight='bold')
plt.xlabel(r'$\omega/\Omega$',fontsize=fontsize)

plt.yticks(fontsize=fontsize,weight='bold')
plt.ylabel(r'$s/\Omega$', fontsize=fontsize)

title=r"Z={0:1.2f}, St={1:4.0e}, $\delta$={2:4.0e}".format(metal, stokes, delta)
plt.title(title,weight='bold')

fname = 'stratsi_plot_eigen'
plt.savefig(fname,dpi=150)


'''
plot eigenfunctions
'''

plt.rc('font',size=fontsize/1.5,weight='bold')

fig, axs = plt.subplots(5, sharex=True, sharey=False, gridspec_kw={'hspace': 0.1}, figsize=(8,7.5))
plt.subplots_adjust(left=0.18, right=0.95, top=0.95, bottom=0.125)

axs[0].plot(z_1f, del_rhod1f.real, linewidth=2, label=r'one-fluid, real', color='black')
axs[0].plot(z_1f, del_rhod1f.imag, linewidth=2, label=r'one-fluid, imag', color='m')

axs[0].plot(z, del_rhod.real, linewidth=2, label=r'two-fluid, real', color='red', linestyle='dashed')
axs[0].plot(z, del_rhod.imag, linewidth=2, label=r'two-fluid, imag', color='c', linestyle='dashed')

axs[0].set_ylabel(r'$\delta\rho_d/\rho_d$')
lines1, labels1 = axs[0].get_legend_handles_labels()
axs[0].legend(lines1, labels1, loc=(0.6,-0.07), frameon=False, ncol=1, labelspacing=0.3, handletextpad=0.1)

title=r"Z={0:1.2f}, St={1:4.0e}, $\delta$={2:4.0e}".format(metal, stokes, delta)
axs[0].set_title(title,weight='bold')

axs[1].plot(z_1f, W1f.real, linewidth=2, label=r'one-fluid, real', color='black')
axs[1].plot(z_1f, W1f.imag, linewidth=2, label=r'one-fluid, imag', color='m')

axs[1].plot(z, W.real, linewidth=2, label=r'two-fluid, real', color='red', linestyle='dashed')
axs[1].plot(z, W.imag, linewidth=2, label=r'two-fluid, imag', color='c', linestyle='dashed')

axs[1].set_ylabel(r'$\delta\rho_g/\rho_{g}$')
axs[1].ticklabel_format(axis='y', style='sci',scilimits=(-2,2))
axs[1].yaxis.set_major_formatter(FormatStrFormatter('%3.0e'))

#lines1, labels1 = axs[1].get_legend_handles_labels()
#axs[1].legend(lines1, labels1, loc='right', frameon=False, ncol=1)

axs[2].plot(z_1f, np.abs(Ux), linewidth=2, label=r'one-fluid', color='black')
axs[2].plot(z, np.abs(Udx), linewidth=2, label=r'dust', color='red', linestyle='dashed')
axs[2].plot(z, np.abs(Ugx), linewidth=2, label=r'gas', color='lime', linestyle='dotted')
#axs[2].plot(z_1f, Ux_norm.imag, linewidth=2, color='black')
#axs[2].plot(z, Ugx_norm.imag, linewidth=2, color='red', linestyle='dashed')
#axs[2].plot(z, Udx_norm.imag, linewidth=2, color='blue', linestyle='dotted')
axs[2].set_ylabel(r'$|\delta v_{x}|$')

lines1, labels1 = axs[2].get_legend_handles_labels()
axs[2].legend(lines1, labels1, loc='right', frameon=False, ncol=1, labelspacing=0.3, handletextpad=0.1)

axs[3].plot(z_1f, np.abs(Uy), linewidth=2, label=r'one-fluid', color='black')
axs[3].plot(z, np.abs(Udy), linewidth=2, label=r'dust', color='red', linestyle='dashed')
axs[3].plot(z, np.abs(Ugy), linewidth=2, label=r'gas', color='lime', linestyle='dotted')
axs[3].set_ylabel(r'$|\delta v_{y}|$')
#lines1, labels1 = axs[3].get_legend_handles_labels()
#axs[3].legend(lines1, labels1, loc='right', frameon=False, ncol=1)

ymax = np.amax(np.abs(Uy))
#arrbeg = r'\begin{align*}'
#arrend = r'\end{align*}'
#plt.rcParams.update(custom_preamble)
axs[3].annotate(r"$k_xH_g$={0:3.0f}".format(kx)+"\n"+r"s={0:4.2f}$\Omega$".format(sgrow), xy=(0.75*xmax, 0.5*ymax))

axs[4].plot(z_1f, np.abs(Uz), linewidth=2, label=r'one-fluid', color='black')
axs[4].plot(z, np.abs(Udz), linewidth=2, label=r'dust', color='red', linestyle='dashed')
axs[4].plot(z, np.abs(Ugz), linewidth=2, label=r'gas', color='lime', linestyle='dotted')
axs[4].set_ylabel(r'$|\delta v_{z}|$')
#lines1, labels1 = axs[4].get_legend_handles_labels()
#axs[4].legend(lines1, labels1, loc='right', frameon=False, ncol=1)

axs[4].set_xlabel(r'$z/H_g$',fontweight='bold')

plt.xlim(xmin,xmax)

fname = 'stratsi_plot_eigenfunc'
plt.savefig(fname,dpi=150)


'''
2D visualization of eigenfunction (using two-fluid solution)
'''
nx = 128
nz = nx

xaxis  = (2.0*np.pi/kx)*np.linspace(-1.0, 1.0, nx)
zaxis  = np.linspace(np.amin(z), np.amax(z), nz)

X, Z = np.meshgrid(xaxis,zaxis)

rhod = np.interp(zaxis, z, del_rhod)
vdx  = np.interp(zaxis, z, Udx)
vdz  = np.interp(zaxis, z, Udz)

rhod_2D = np.repeat(rhod[...,np.newaxis], nx, axis=1)
vdx_2D   = np.repeat(vdx[...,np.newaxis], nx, axis=1)
vdz_2D   = np.repeat(vdz[...,np.newaxis], nx, axis=1)

data   = np.cos(kx*X)*rhod_2D.real - np.sin(kx*X)*rhod_2D.imag
U      = np.cos(kx*X)*vdx_2D.real - np.sin(kx*X)*vdx_2D.imag
V      = np.cos(kx*X)*vdz_2D.real - np.sin(kx*X)*vdz_2D.imag

plt.figure(figsize=(7,7))

plt.ylim(np.amin(zaxis), np.amax(zaxis))
plt.xlim(np.amin(xaxis), np.amax(xaxis))

minv = np.amin(data)
maxv = np.amax(data)

levels  = np.linspace(minv,maxv,nlev)
clevels = np.linspace(minv,maxv,nclev)

plt.rc('font',size=fontsize,weight='bold')

cp = plt.contourf(xaxis, zaxis, data, levels, cmap=cmap)

xfac = np.int(nx/64)
zfac = np.int(nz/128)

#plt.quiver(xaxis[0:nx:xfac], zaxis[0:nz:zfac], U[0:nz:zfac,0:nx:xfac],
#               V[0:nz:zfac,0:nx:xfac], color='deepskyblue',
#               width=0.005, scale=0.2
#               )

speed = np.sqrt(U**2 + V**2)
lw    = 0.7#2*speed/speed.max()

plt.streamplot(xaxis, zaxis, U, V, 
               color='deepskyblue', density=3, 
               linewidth=lw
               )

#plt.gca().set_aspect("equal")
#plt.tight_layout()
plt.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.125)

plt.colorbar(cp,ticks=clevels,format='%.2f')

title=r"$k_xH_g$={0:3.0f}".format(kx)+r", s={0:4.2f}$\Omega$".format(sgrow)
plt.title(title,weight='bold')

plt.xticks(fontsize=fontsize,weight='bold')
plt.xlabel(r'$x/H_g$',fontsize=fontsize)

plt.yticks(fontsize=fontsize,weight='bold')
plt.ylabel(r'$z/H_g$',fontsize=fontsize)

fname = 'stratsi_plot_eigenf2D'
plt.savefig(fname,dpi=150)



'''
plot kinetic energy decomposition based on 1-fluid result
'''

fig = plt.figure(figsize=(8,4.5))
ax = fig.add_subplot()
plt.subplots_adjust(left=0.2, right=0.95, top=0.9, bottom=0.2)

plt.xlim(np.amin(z_1f),np.amax(z_1f))

plt.plot(z_1f, energy1f_A, linewidth=2,label='$E_1, dv/dz$')
plt.plot(z_1f, energy1f_A2, linewidth=2,label=r'$E_{1y}$, $dv_y/dz$',color='black',marker='x',linestyle='None',markevery=8)

plt.plot(z_1f, energy1f_B, linewidth=2,label='$E_2$, vert. settling')
plt.plot(z_1f, energy1f_C, linewidth=2,label='$E_3$, pressure')
plt.plot(z_1f, energy1f_D, linewidth=2,label='$E_4$, dust-gas drift')
plt.plot(z_1f, energy1f_E, linewidth=2,label='$E_5$, buoyancy')

plt.plot(z_1f, energy1f_A + energy1f_B + energy1f_C + energy1f_D + energy1f_E, linewidth=2,label=r'$\sum E_i$',linestyle='dashed')
plt.plot(z_1f, energy1f_tot, linewidth=2,label=r'$E_{tot}$',color='black',marker='o',linestyle='None',markevery=8)

#ax.axvline(x=z_maxvshear, linestyle='dashed', linewidth=1, label=r'$max\left|dv_y/dz\right|/\Omega$')

plt.rc('font',size=fontsize,weight='bold')

lines1, labels1 = ax.get_legend_handles_labels()
legend=ax.legend(lines1, labels1, loc='upper right', frameon=False, ncol=1, fontsize=fontsize/2)

plt.xticks(fontsize=fontsize,weight='bold')
plt.xlabel(r'$z/H_g$',fontsize=fontsize)

title=r"$k_xH_g$={0:3.0f}".format(kx_1f)+r", s={0:4.2f}$\Omega$".format(sgrow_1f)
plt.title(title,weight='bold')

plt.yticks(fontsize=fontsize,weight='bold')
plt.ylabel(r'$pseudo$-$energy$', fontsize=fontsize)

fname = 'stratsi_plot_energy1f'
plt.savefig(fname,dpi=150)


'''
plot kinetic energy decomposition based on 2-fluid result
'''

fig = plt.figure(figsize=(8,4.5))
ax = fig.add_subplot()
plt.subplots_adjust(left=0.2, right=0.95, top=0.9, bottom=0.2)

plt.xlim(xmin,xmax)

plt.plot(z, energy2f_A, linewidth=2,label='$U_1$, vert. shear')
plt.plot(z, energy2f_A2, linewidth=2,label='$U_{1y}$, (vert. shear)$_y$', color='black',marker='x',linestyle='None',markevery=8)

plt.plot(z, energy2f_B, linewidth=2,label='$U_2$, dust settling')
plt.plot(z, energy2f_C, linewidth=2,label='$U_3$, gas pressure')
plt.plot(z, energy2f_D, linewidth=2,label='$U_4$, dust-gas drift')
plt.plot(z, energy2f_E, linewidth=2,label='$U_5$, buoyancy')
if viscosity_pert == True:
    plt.plot(z, energy2f_F, linewidth=2,label='$U_6$, viscosity')
    
plt.plot(z, energy2f_A + energy2f_B + energy2f_C + energy2f_D + energy2f_E + energy2f_F, linewidth=2,label=r'$\sum U_i$',linestyle='dashed')
plt.plot(z, energy2f_tot, linewidth=2,label=r'$U_{tot}$',color='black',marker='o',linestyle='None',markevery=8)

plt.rc('font',size=fontsize,weight='bold')

lines1, labels1 = ax.get_legend_handles_labels()
legend=ax.legend(lines1, labels1, loc='upper right', frameon=False, ncol=1, fontsize=fontsize/2, labelspacing=0.4)

plt.xticks(fontsize=fontsize,weight='bold')
plt.xlabel(r'$z/H_g$',fontsize=fontsize)

plt.yticks(fontsize=fontsize,weight='bold')
plt.ylabel(r'$pseudo$-$energy$', fontsize=fontsize)

title=r"$k_xH_g$={0:3.0f}".format(kx)+r", s={0:4.2f}$\Omega$".format(sgrow)
plt.title(title,weight='bold')

fname = 'stratsi_plot_energy2f'
plt.savefig(fname,dpi=150)

'''
plot energy decomposition as a function of kx (1 fluid)
'''

fig = plt.figure(figsize=(8,4.5))
ax = fig.add_subplot()
plt.subplots_adjust(left=0.185, right=0.95, top=0.9, bottom=0.2)
plt.xscale('log')
#plt.yscale('log')

plt.xlim(np.amin(ks_1f),np.amax(ks_1f))

energy1f_tot_int = np.cbrt(energy1f_tot_int)
energy1f_A_int = np.cbrt(energy1f_A_int)
energy1f_A2_int = np.cbrt(energy1f_A2_int)
energy1f_B_int = np.cbrt(energy1f_B_int)
energy1f_C_int = np.cbrt(energy1f_C_int)
energy1f_D_int = np.cbrt(energy1f_D_int)
energy1f_E_int = np.cbrt(energy1f_E_int)

#plt.plot(ks_1f, energy1f_A_int, linewidth=2,label='$dv/dz$')
plt.plot(ks_1f, energy1f_A2_int, linewidth=2,label='$dv_y/dz$')
plt.plot(ks_1f, energy1f_B_int, linewidth=2,label='vert. settling')
plt.plot(ks_1f, energy1f_C_int, linewidth=2,label='pressure')
plt.plot(ks_1f, energy1f_D_int, linewidth=2,label='dust-gas drift')
plt.plot(ks_1f, energy1f_E_int, linewidth=2,label='buoyancy')

plt.plot(ks_1f, energy1f_tot_int, linewidth=2,label=r'total',color='black',marker='o',linestyle='None',markevery=2)

plt.plot([1e2,1e4], [0,0], linewidth=1,linestyle='dashed',color='black')

lines1, labels1 = ax.get_legend_handles_labels()
legend=ax.legend(lines1, labels1, loc='upper left', frameon=False, ncol=1, fontsize=fontsize/2)

plt.rc('font',size=fontsize,weight='bold')

plt.xticks(fontsize=fontsize,weight='bold')
plt.xlabel(r'$k_xH_g$',fontsize=fontsize)

plt.yticks(fontsize=fontsize,weight='bold')
plt.ylabel(r'$\left(\int \rho E_i dz\right)^{1/3}$', fontsize=fontsize)


title=r"Z={0:1.2f}, St={1:4.0e}, $\delta$={2:4.0e}".format(metal, stokes, delta)
plt.title(title,weight='bold')

fname = 'stratsi_plot_energy1f_int'
plt.savefig(fname,dpi=150)


'''
plot energy decomposition as a function of kx (2 fluid)
'''

fig = plt.figure(figsize=(8,4.5))
ax = fig.add_subplot()
plt.subplots_adjust(left=0.185, right=0.95, top=0.9, bottom=0.2)
plt.xscale('log')

plt.xlim(np.amin(ks),np.amax(ks))

energy2f_tot_int = np.cbrt(energy2f_tot_int)
energy2f_A_int = np.cbrt(energy2f_A_int)
energy2f_A2_int = np.cbrt(energy2f_A2_int)
energy2f_B_int = np.cbrt(energy2f_B_int)
energy2f_C_int = np.cbrt(energy2f_C_int)
energy2f_D_int = np.cbrt(energy2f_D_int)
energy2f_E_int = np.cbrt(energy2f_E_int)
energy2f_F_int = np.cbrt(energy2f_F_int)

#plt.plot(ks_1f, energy1f_A_int, linewidth=2,label='$dv/dz$')
plt.plot(ks, energy2f_A2_int, linewidth=2,label='(vert. shear)$_y$')
plt.plot(ks, energy2f_B_int, linewidth=2,label='dust settling')
plt.plot(ks, energy2f_C_int, linewidth=2,label='gas pressure')
plt.plot(ks, energy2f_D_int, linewidth=2,label='dust-gas drift')
plt.plot(ks, energy2f_E_int, linewidth=2,label='buoyancy')
if viscosity_pert == True:
    plt.plot(ks, energy2f_F_int, linewidth=2,label='viscosity')

plt.plot(ks, energy2f_tot_int, linewidth=2,label=r'total',color='black',marker='o',linestyle='None',markevery=2)

plt.plot([1e2,1e4], [0,0], linewidth=1,linestyle='dashed',color='black')

lines1, labels1 = ax.get_legend_handles_labels()
legend=ax.legend(lines1, labels1, loc='upper left', frameon=False, ncol=2, fontsize=fontsize/2)

plt.rc('font',size=fontsize,weight='bold')

plt.xticks(fontsize=fontsize,weight='bold')
plt.xlabel(r'$k_xH_g$',fontsize=fontsize)

plt.yticks(fontsize=fontsize,weight='bold')
plt.ylabel(r'$\left(\int \rho_g U_i dz\right)^{1/3}$', fontsize=fontsize)

title=r"Z={0:1.2f}, St={1:4.0e}, $\delta$={2:4.0e}".format(metal, stokes, delta)
plt.title(title,weight='bold')

fname = 'stratsi_plot_energy2f_int'
plt.savefig(fname,dpi=150)
