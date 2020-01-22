import sys
import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt
import h5py
import argparse

from stratsi_params import delta, stokes, metal

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

print("one-fluid model: kx, growth, freq = {0:1.2e} {1:13.6e} {2:13.6e}".format(kx_1f, growth_1f[g1], freq_1f[g1]))

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

print("two-fluid model: kx, growth, freq = {0:1.2e} {1:13.6e} {2:13.6e}".format(kx, growth[g1], freq[g1]))


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
            axs[0].plot(k, sig.real, marker='o', linestyle='none', markersize=8, label=r'one fluid',color='black')
        else:
            axs[0].plot(k, sig.real, marker='o', linestyle='none', markersize=8, color='black')
            
for i, k in enumerate(ks):
    for n, sig in enumerate(freqs[i]):
        if (i == 0) & (n == 0):
            axs[0].plot(k, sig.real, marker='X', linestyle='none', markersize=8, label=r'two fluid',color='red')
        else:
            axs[0].plot(k, sig.real, marker='X', linestyle='none', markersize=8, color='red')

axs[0].set_ylabel(r'$s/\Omega$')
lines1, labels1 = axs[0].get_legend_handles_labels()
legend=axs[0].legend(lines1, labels1, loc='upper left', frameon=False, ncol=1, handletextpad=-0.5,fontsize=fontsize/2)

for i, k in enumerate(ks_1f):
    for n, sig in enumerate(freqs_1f[i]):
        if (i == 0) & (n == 0):
            axs[1].plot(k, -sig.imag, marker='o', linestyle='none', markersize=8, label=r'one fluid',color='black')
        else:
            axs[1].plot(k, -sig.imag, marker='o', linestyle='none', markersize=8,color='black')

for i, k in enumerate(ks):
    for n, sig in enumerate(freqs[i]):
        if (i == 0) & (n == 0):
            axs[1].plot(k, -sig.imag, marker='X', linestyle='none', markersize=8, label=r'two fluid',color='red')
        else:
            axs[1].plot(k, -sig.imag, marker='X', linestyle='none', markersize=8,color='red')
            
#axs[1].plot(ks, -freqs.imag, marker='X',markersize=10,linestyle='none',  label=r'two fluid')

axs[1].set_ylabel(r'$\omega/\Omega$')
axs[1].set_xlabel(r'$k_xH_g$')
lines1, labels1 = axs[1].get_legend_handles_labels()
legend=axs[1].legend(lines1, labels1, loc='upper left', frameon=False, ncol=1, handletextpad=-0.5, fontsize=fontsize/2)

plt.xlim(np.amin(ks),np.amax(ks))

fname = 'stratsi_plot_growth'
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
lines1, labels1 = axs[0].get_legend_handles_labels()
axs[0].legend(lines1, labels1, loc=(0.55,-0.08), frameon=False, ncol=1, labelspacing=0.3, handletextpad=0.1)

W1f_norm = W1f/norm_1f
W_norm   = W/norm
axs[1].plot(z_1f, W1f_norm.real, linewidth=2, label=r'one-fluid, real', color='black')
axs[1].plot(z_1f, W1f_norm.imag, linewidth=2, label=r'one-fluid, imag', color='m')

axs[1].plot(z, W_norm.real, linewidth=2, label=r'two-fluid, real', color='red', linestyle='dashed')
axs[1].plot(z, W_norm.imag, linewidth=2, label=r'two-fluid, imag', color='c', linestyle='dashed')

axs[1].set_ylabel(r'$\delta\rho_g/\rho_{g}$')
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
lines1, labels1 = axs[2].get_legend_handles_labels()
axs[2].legend(lines1, labels1, loc='right', frameon=False, ncol=1, labelspacing=0.3, handletextpad=0.1)

Uy_norm = Uy/norm_1f
Ugy_norm = Ugy/norm
Udy_norm = Udy/norm
axs[3].plot(z_1f, np.abs(Uy_norm), linewidth=2, label=r'one-fluid', color='black')
axs[3].plot(z, np.abs(Ugy_norm), linewidth=2, label=r'dust', color='red', linestyle='dashed')
axs[3].plot(z, np.abs(Udy_norm), linewidth=2, label=r'gas', color='lime', linestyle='dotted')
axs[3].set_ylabel(r'$|\delta v_{y}|$')
#lines1, labels1 = axs[3].get_legend_handles_labels()
#axs[3].legend(lines1, labels1, loc='right', frameon=False, ncol=1)

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











