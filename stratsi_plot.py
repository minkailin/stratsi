import sys
import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
import h5py
import argparse

#custom_preamble = {
#    "text.usetex": True,
#    "text.latex.preamble": [
#        r"\usepackage{amsmath}", # for the align enivironment
#        ],
#    }


from stratsi_params import delta, stokes, metal, epsilon, vdz, dvdz
from stratsi_1fluid import epsilon_eqm, dvx_eqm, dvy_eqm, vz_eqm, dvz_eqm, eta_hat

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

W1f         = eig_W1f[n][g1]
Q1f         = eig_Q1f[n][g1]
Ux          = eig_Ux[n][g1]
Uy          = eig_Uy[n][g1]
Uz          = eig_Uz[n][g1]

del_rhod1f    = Q1f + W1f

sgrow_1f = growth_1f[g1]
ofreq_1f = freq_1f[g1]
print("one-fluid model: kx, growth, freq = {0:1.2e} {1:13.6e} {2:13.6e}".format(kx_1f, sgrow_1f, ofreq_1f))


'''
analysis of energetics based on 1 fluid results 
'''

eps1f = epsilon_eqm(z_1f)
dvx1f = dvx_eqm(z_1f)
dvy1f = dvy_eqm(z_1f)
vz1f  = vz_eqm(z_1f)
dvz1f = dvz_eqm(z_1f)

del_rho1f = W1f + eps1f*Q1f/(1.0 + eps1f)

dW1f= np.gradient(W1f, z_1f) 
dUx = np.gradient(Ux, z_1f)
dUy = np.gradient(Uy, z_1f)
dUz = np.gradient(Uz, z_1f)

fac = 4.0

energy1f_tot = sgrow_1f*(np.abs(Ux)**2 + fac*np.abs(Uy)**2 + np.abs(Uz)**2)

energy1f_A1 = -(dvx1f*np.real(Uz*np.conj(Ux)))
energy1f_A2 = -(fac*dvy1f*np.real(Uz*np.conj(Uy)))
energy1f_A3 = -(dvz1f*np.abs(Uz)**2)
energy1f_A = energy1f_A1 + energy1f_A2 + energy1f_A3
energy1f_B = -vz1f*np.real(dUx*np.conj(Ux) + fac*dUy*np.conj(Uy) + dUz*np.conj(Uz))
energy1f_C = (kx_1f*np.imag(W1f*np.conj(Ux)) - np.real(dW1f*np.conj(Uz)))/(1.0 + eps1f)
energy1f_D = -2.0*eta_hat*np.real(del_rho1f*np.conj(Ux))/(1.0 + eps1f)
energy1f_E = -z_1f*eps1f*np.real(Q1f*np.conj(Uz))/(1.0 + eps1f)
energy1f_F = 2.0*np.real(Uy*np.conj(Ux)) - 0.5*np.real(Ux*np.conj(Uy))

energy1f_A/= energy1f_tot
energy1f_A1/= energy1f_tot
energy1f_A2/= energy1f_tot
energy1f_A3/= energy1f_tot
energy1f_B/= energy1f_tot
energy1f_C/= energy1f_tot
energy1f_D/= energy1f_tot
energy1f_E/= energy1f_tot
energy1f_F/= energy1f_tot

#exit()


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

W         = eig_W[m][g1]
Q         = eig_Q[m][g1]
Ugx          = eig_Ugx[m][g1]
Ugy          = eig_Ugy[m][g1]
Ugz          = eig_Ugz[m][g1]
Udx          = eig_Udx[m][g1]
Udy          = eig_Udy[m][g1]
Udz          = eig_Udz[m][g1]

del_rhod    = Q + W

sgrow = growth[g1]
ofreq = freq[g1]
print("two-fluid model: kx, growth, freq = {0:1.2e} {1:13.6e} {2:13.6e}".format(kx, sgrow, ofreq))

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

eps2f  = epsilon(z)

dW   = np.gradient(W, z) 
dUgx = np.gradient(Ugx, z)
dUgy = np.gradient(Ugy, z)
dUgz = np.gradient(Ugz, z)

dUdx = np.gradient(Udx, z)
dUdy = np.gradient(Udy, z)
dUdz = np.gradient(Udz, z)

energy2f_tot = eps2f*sgrow*(np.abs(Udx)**2 + 4.0*np.abs(Udy)**2 + np.abs(Udz)**2)
energy2f_tot+= sgrow*(np.abs(Ugx)**2 + 4.0*np.abs(Ugy)**2 + np.abs(Ugz)**2)
energy2f_A   =-eps2f*np.real(Udz*np.conj(dvdx*Udx + 4.0*dvdy*Udy + dvdz(z)*Udz))
energy2f_A  +=-np.real(Ugz*np.conj(dvgx*Ugx + 4.0*dvgy*Ugy))
energy2f_A2  = -eps2f*np.real(Udz*np.conj(4.0*dvdy*Udy)) - np.real(Ugz*np.conj(4.0*dvgy*Ugy))
energy2f_B   =-eps2f*vdz(z)*np.real(dUdx*np.conj(Udx) + 4.0*dUdy*np.conj(Udy) + dUdz*np.conj(Udz))
energy2f_C   = kx*np.imag(W*np.conj(Ugx)) - np.real(dW*np.conj(Ugz))
energy2f_D   = (vgx - vdx)*np.real(Q*np.conj(Ugx)) + 4.0*(vgy - vdy)*np.real(Q*np.conj(Ugy)) - vdz(z)*np.real(Q*np.conj(Ugz))
energy2f_D  += np.abs(Ugx - Udx)**2 + 4.0*np.abs(Ugy - Udy)**2 + np.abs(Ugz - Udz)**2
energy2f_D  *= -eps2f/stokes

energy2f_A /= energy2f_tot
energy2f_A2 /= energy2f_tot
energy2f_B /= energy2f_tot
energy2f_C /= energy2f_tot
energy2f_D /= energy2f_tot


'''
calculate ratio between vertical to horizontal motions, for the most unstable mode at each kx in two fluid model
'''
theta = np.zeros(ks.size)
for i, k in enumerate(ks):
    g2          = np.argmax(freqs[i].real)
    theta2_z    = np.abs(eig_Ugz[i][g2])**2.0 + np.abs(eig_Udz[i][g2])**2.0
    theta2_z   /= np.abs(eig_Ugx[i][g2])**2.0 + np.abs(eig_Ugy[i][g2])**2.0 + np.abs(eig_Udx[i][g2])**2.0 + np.abs(eig_Udy[i][g2])**2.0
    theta[i]    = np.sqrt(np.mean(theta2_z))

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
#plt.suptitle(title,y=0.99,fontsize=fontsize,fontweight='bold')
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
plt.subplots_adjust(left=0.16, right=0.95, top=0.95, bottom=0.15)
plt.xscale('log')

for i, k in enumerate(ks_1f):
    g1 = np.argmax(freqs_1f[i].real)
    if i == 0:
        lab = r'one fluid'
    else:
        lab = ''
    axs[0].plot(k, freqs_1f[i][g1].real , marker='o', linestyle='none', markersize=8, label=lab,color='black')
    
for i, k in enumerate(ks):
    g1 = np.argmax(freqs[i].real)
    if i == 0:
        lab = r'two fluid'
    else:
        lab = ''
    axs[0].plot(k, freqs[i][g1].real , marker='X', linestyle='none', markersize=8, label=lab,color='red')
    
axs[0].set_ylabel(r'$s_\mathrm{max}/\Omega$')
lines1, labels1 = axs[0].get_legend_handles_labels()
legend=axs[0].legend(lines1, labels1, loc='upper left', frameon=False, ncol=1, handletextpad=-0.5,fontsize=fontsize/2)

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

plt.xlim(np.amin(ks),np.amax(ks))

fname = 'stratsi_plot_growth_max'
plt.savefig(fname,dpi=150)


'''
plot eigenvalues as scatter diagram for a single kx 
'''

fig = plt.figure(figsize=(8,4.5))
ax = fig.add_subplot()
plt.subplots_adjust(left=0.2, right=0.95, top=0.9, bottom=0.2)

title=r"$K_x$={0:4.0f}, St={1:4.0e}, $\delta$={2:4.0e}".format(ks[m], stokes, delta)
plt.suptitle(title,y=0.99,fontsize=fontsize,fontweight='bold',x=0.55)

plt.scatter(-sigma_1f.imag, sigma_1f.real, marker='o', label=r'one fluid',color='black',s=64)
plt.scatter(-sigma.imag, sigma.real, marker='X', label=r'two fluid',color='red',s=64)

plt.rc('font',size=fontsize,weight='bold')

lines1, labels1 = ax.get_legend_handles_labels()
legend=ax.legend(lines1, labels1, loc='upper right', frameon=False, ncol=1, handletextpad=-0.5, fontsize=fontsize/2)

plt.xticks(fontsize=fontsize,weight='bold')
plt.xlabel(r'$\omega/\Omega$',fontsize=fontsize)

plt.yticks(fontsize=fontsize,weight='bold')
plt.ylabel(r'$s/\Omega$', fontsize=fontsize)

fname = 'stratsi_plot_eigen'
plt.savefig(fname,dpi=150)


'''
plot eigenfunctions
'''

plt.rc('font',size=fontsize/1.5,weight='bold')

fig, axs = plt.subplots(5, sharex=True, sharey=False, gridspec_kw={'hspace': 0.1}, figsize=(8,7.5))
plt.subplots_adjust(left=0.18, right=0.95, top=0.95, bottom=0.125)
title=r"Z={0:1.2f}, St={1:4.0e}, $\delta$={2:4.0e}".format(metal, stokes, delta)
plt.suptitle(title,y=0.99,fontsize=fontsize,fontweight='bold')

g1 = np.argmax(np.abs(del_rhod1f))
norm_1f = del_rhod1f[g1]

g1 = np.argmax(np.abs(del_rhod))
norm    = del_rhod[g1]

deleps_1f = del_rhod1f/norm_1f
deleps    = del_rhod/norm

axs[0].plot(z_1f, deleps_1f.real, linewidth=2, label=r'one-fluid, real', color='black')
axs[0].plot(z_1f, deleps_1f.imag, linewidth=2, label=r'one-fluid, imag', color='m')

axs[0].plot(z, deleps.real, linewidth=2, label=r'two-fluid, real', color='red', linestyle='dashed')
axs[0].plot(z, deleps.imag, linewidth=2, label=r'two-fluid, imag', color='c', linestyle='dashed')

axs[0].set_ylabel(r'$\delta\rho_d/\rho_d$')
#lines1, labels1 = axs[0].get_legend_handles_labels()
#axs[0].legend(lines1, labels1, loc=(0.6,-0.07), frameon=False, ncol=1, labelspacing=0.3, handletextpad=0.1)

W1f_norm = W1f/norm_1f
W_norm   = W/norm
axs[1].plot(z_1f, W1f_norm.real, linewidth=2, label=r'one-fluid, real', color='black')
axs[1].plot(z_1f, W1f_norm.imag, linewidth=2, label=r'one-fluid, imag', color='m')

axs[1].plot(z, W_norm.real, linewidth=2, label=r'two-fluid, real', color='red', linestyle='dashed')
axs[1].plot(z, W_norm.imag, linewidth=2, label=r'two-fluid, imag', color='c', linestyle='dashed')

axs[1].set_ylabel(r'$\delta\rho_g/\rho_{g}$')
axs[1].ticklabel_format(axis='y', style='sci',scilimits=(-2,2))
axs[1].yaxis.set_major_formatter(FormatStrFormatter('%3.0e'))

#lines1, labels1 = axs[1].get_legend_handles_labels()
#axs[1].legend(lines1, labels1, loc='right', frameon=False, ncol=1)

Ux_norm = Ux/norm_1f
Ugx_norm = Ugx/norm
Udx_norm = Udx/norm
axs[2].plot(z_1f, np.abs(Ux_norm), linewidth=2, label=r'one-fluid', color='black')
axs[2].plot(z, np.abs(Ugx_norm), linewidth=2, label=r'dust', color='red', linestyle='dashed')
axs[2].plot(z, np.abs(Udx_norm), linewidth=2, label=r'gas', color='lime', linestyle='dotted')
#axs[2].plot(z_1f, Ux_norm.imag, linewidth=2, color='black')
#axs[2].plot(z, Ugx_norm.imag, linewidth=2, color='red', linestyle='dashed')
#axs[2].plot(z, Udx_norm.imag, linewidth=2, color='blue', linestyle='dotted')
axs[2].set_ylabel(r'$|\delta v_{x}|$')
#lines1, labels1 = axs[2].get_legend_handles_labels()
#axs[2].legend(lines1, labels1, loc='right', frameon=False, ncol=1, labelspacing=0.3, handletextpad=0.1)

Uy_norm = Uy/norm_1f
Ugy_norm = Ugy/norm
Udy_norm = Udy/norm
axs[3].plot(z_1f, np.abs(Uy_norm), linewidth=2, label=r'one-fluid', color='black')
axs[3].plot(z, np.abs(Ugy_norm), linewidth=2, label=r'dust', color='red', linestyle='dashed')
axs[3].plot(z, np.abs(Udy_norm), linewidth=2, label=r'gas', color='lime', linestyle='dotted')
axs[3].set_ylabel(r'$|\delta v_{y}|$')
#lines1, labels1 = axs[3].get_legend_handles_labels()
#axs[3].legend(lines1, labels1, loc='right', frameon=False, ncol=1)

ymax = np.amax(np.abs(Uy_norm))
#arrbeg = r'\begin{align*}'
#arrend = r'\end{align*}'
#plt.rcParams.update(custom_preamble)
axs[3].annotate(r"$k_xH_g$={0:3.0f}".format(kx)+"\n"+r"s={0:4.2f}$\Omega$".format(sgrow), xy=(0.75*xmax, 0.5*ymax))


Uz_norm = Uz/norm_1f
Ugz_norm = Ugz/norm
Udz_norm = Udz/norm
axs[4].plot(z_1f, np.abs(Uz_norm), linewidth=2, label=r'one-fluid', color='black')
axs[4].plot(z, np.abs(Ugz_norm), linewidth=2, label=r'dust', color='red', linestyle='dashed')
axs[4].plot(z, np.abs(Udz_norm), linewidth=2, label=r'gas', color='lime', linestyle='dotted')
axs[4].set_ylabel(r'$|\delta v_{z}|$')
#lines1, labels1 = axs[4].get_legend_handles_labels()
#axs[4].legend(lines1, labels1, loc='right', frameon=False, ncol=1)

axs[4].set_xlabel(r'$z/H_g$',fontweight='bold')

#plt.xticks(f,weight='bold')
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

rhod = np.interp(zaxis, z, deleps)
vdx  = np.interp(zaxis, z, Udx_norm)
vdz  = np.interp(zaxis, z, Udz_norm)

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
plot ratio of vertical to horizontal motions

fig = plt.figure(figsize=(8,4.5))
ax = fig.add_subplot()
plt.subplots_adjust(left=0.18, right=0.95, top=0.95, bottom=0.2)
plt.xscale('log')

plt.xlim(np.amin(ks),np.amax(ks))

plt.plot(ks, theta, linewidth=2)

plt.rc('font',size=fontsize,weight='bold')

plt.xticks(fontsize=fontsize,weight='bold')
plt.xlabel(r'$k_xH_g$',fontsize=fontsize)

plt.yticks(fontsize=fontsize,weight='bold')
plt.ylabel(r'$\sqrt{\frac{|v_{gz}|^2 + |v_{dz}|^2}{|v_{gx}|^2 + |v_{gy}|^2 + |v_{gx}|^2  + |v_{gy}|^2}}$', fontsize=fontsize)

fname = 'stratsi_plot_theta'
plt.savefig(fname,dpi=150)
'''

'''
plot kinetic energy decomposition based on 1-fluid result
'''

fig = plt.figure(figsize=(8,4.5))
ax = fig.add_subplot()
plt.subplots_adjust(left=0.18, right=0.95, top=0.95, bottom=0.2)

plt.xlim(xmin,xmax)

plt.plot(z_1f, energy1f_A, linewidth=2,label='A')
plt.plot(z_1f, energy1f_A2, linewidth=2,label='A2',color='black',linestyle='dashed')

plt.plot(z_1f, energy1f_B, linewidth=2,label='B')
plt.plot(z_1f, energy1f_C, linewidth=2,label='C')
plt.plot(z_1f, energy1f_D, linewidth=2,label='D')
plt.plot(z_1f, energy1f_E, linewidth=2,label='E')
#plt.plot(z_1f, energy1f_F, linewidth=2,label='F')

plt.plot(z_1f, energy1f_A + energy1f_B + energy1f_C + energy1f_D + energy1f_E, linewidth=2,label='total')

plt.rc('font',size=fontsize,weight='bold')

lines1, labels1 = ax.get_legend_handles_labels()
legend=ax.legend(lines1, labels1, loc='upper right', frameon=False, ncol=1, fontsize=fontsize/2)

plt.xticks(fontsize=fontsize,weight='bold')
plt.xlabel(r'$z/H_g$',fontsize=fontsize)

plt.yticks(fontsize=fontsize,weight='bold')
plt.ylabel(r'$energy$ $fraction$', fontsize=fontsize)

fname = 'stratsi_plot_energy1f'
plt.savefig(fname,dpi=150)

'''
plot kinetic energy decomposition based on 2-fluid result
'''

fig = plt.figure(figsize=(8,4.5))
ax = fig.add_subplot()
plt.subplots_adjust(left=0.18, right=0.95, top=0.95, bottom=0.2)

plt.xlim(xmin,xmax)

plt.plot(z, energy2f_A, linewidth=2,label='A')
plt.plot(z_1f, energy2f_A2, linewidth=2,label='A2',color='black',linestyle='dashed')

plt.plot(z, energy2f_B, linewidth=2,label='B')
plt.plot(z, energy2f_C, linewidth=2,label='C')
plt.plot(z, energy2f_D, linewidth=2,label='D')

plt.plot(z, energy2f_A + energy2f_B + energy2f_C + energy2f_D, linewidth=2,label='total')

plt.rc('font',size=fontsize,weight='bold')

lines1, labels1 = ax.get_legend_handles_labels()
legend=ax.legend(lines1, labels1, loc='upper right', frameon=False, ncol=1, fontsize=fontsize/2)

plt.xticks(fontsize=fontsize,weight='bold')
plt.xlabel(r'$z/H_g$',fontsize=fontsize)

plt.yticks(fontsize=fontsize,weight='bold')
plt.ylabel(r'$energy$ $fraction$', fontsize=fontsize)

fname = 'stratsi_plot_energy2f'
plt.savefig(fname,dpi=150)
