import sys
import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
import h5py
import argparse
from scipy.integrate import simps

parser = argparse.ArgumentParser()
parser.add_argument("--mode", nargs='*', help="select mode number")

#case parameters
metal = 0.03
stokes= np.array([0.001, 0.03])
delta = 1e-6

stokes_string= ['0.001','0.03']
Hd           = np.sqrt(delta/(stokes + delta)) 

'''
one-fluid results
'''

filenames_1f = ['stratsi_1fluid_modes_stokes'+x+'.h5' for x in stokes_string]

kx_values_1f    = []
growth_rates_1f = []
real_freq_1f    = []
delrhod_1f      = []

for n, fname in enumerate(filenames_1f):

  with h5py.File(fname,'r') as infile:
  
    kx = infile['scales']['kx_space'][:]
    kx_values_1f.append(kx)
    z_1f     = infile['scales']['z'][:]/Hd[n] #vertical axis in units of dust scale height
    
    freqs = []
    W     = []
    Q     = []
  
    for k_i in infile['tasks']:
      freqs.append(infile['tasks'][k_i]['freq'][:])
      W.append(infile['tasks'][k_i]['eig_W'][:])
      Q.append(infile['tasks'][k_i]['eig_Q'][:])

  growth_rates_1f.append([])
  real_freq_1f.append([])
  delrhod_1f.append([])
  
  for i, kxvals in enumerate(kx):
    
    sigma      = freqs[i]

    growth     = sigma.real
    freq       =-sigma.imag

    g1          = np.argmax(growth)

    growth_rates_1f[n].append(growth[g1])
    real_freq_1f[n].append(freq[g1])

    Wmode = np.array(W[i][g1])
    Qmode = np.array(Q[i][g1])
    drhod = Wmode + Qmode
    
    g2      = np.argmax(np.abs(drhod))
    norm    = drhod[g2]
    
    delrhod_1f[n].append(drhod/drhod[g2])
    
    
'''
two-fluid results
'''

filenames_2f = ['stratsi_modes_stokes'+x+'.h5' for x in stokes_string]
    
kx_values_2f    = []
growth_rates_2f = []
real_freq_2f    = []
delrhod_2f      = []

for n, fname in enumerate(filenames_2f):

  with h5py.File(fname,'r') as infile:
  
    kx = infile['scales']['kx_space'][:]
    kx_values_2f.append(kx)
    z_2f     = infile['scales']['z'][:]/Hd[n]
    
    freqs = []
    W     = []
    Q     = []
    
    for k_i in infile['tasks']:
      freqs.append(infile['tasks'][k_i]['freq'][:])
      W.append(infile['tasks'][k_i]['eig_W'][:])
      Q.append(infile['tasks'][k_i]['eig_Q'][:])

  growth_rates_2f.append([])
  real_freq_2f.append([])
  delrhod_2f.append([])
  
  for i, kxvals in enumerate(kx):
    
    sigma    = freqs[i]

    growth   = sigma.real
    freq     =-sigma.imag

    g1          = np.argmax(growth)

    growth_rates_2f[n].append(growth[g1])
    real_freq_2f[n].append(freq[g1])

    Wmode = np.array(W[i][g1])
    Qmode = np.array(Q[i][g1])
    drhod = Wmode + Qmode
    
    g2      = np.argmax(np.abs(drhod))
    norm    = drhod[g2]
    
    delrhod_2f[n].append(drhod/drhod[g2])

'''
plotting parameters
'''

fontsize= 24
nlev    = 128
nclev   = 6
cmap    = plt.cm.inferno

'''
plot max growth rates as func of kx
'''
plt.rc('font',size=fontsize,weight='bold')

fig, axs = plt.subplots(2, sharex=True, sharey=False, gridspec_kw={'hspace': 0.1}, figsize=(8,6))

plt.subplots_adjust(left=0.2, right=0.95, top=0.9, bottom=0.15)
plt.xscale('log')

colors=['red','black','blue']
linestyles=['solid','dashed','dashed']
stokes_label = [r'$St=$'+x for x in stokes_string]

for i in range(0,2):
    if i == 0:
        lab1 = r'one-fluid'
        lab2 = r'two-fluid'
    else:
        lab1 = lab2 = ''
    axs[0].plot(kx_values_1f[i], growth_rates_1f[i], marker='o', linestyle='none', markersize=8, label=lab1,color='black')
    axs[0].plot(kx_values_2f[i], growth_rates_2f[i], marker='X', linestyle='none', markersize=8, label=lab2,color='red')
    axs[0].plot(kx_values_2f[i], growth_rates_2f[i], linestyle=linestyles[i],color='red')

    axs[1].plot(kx_values_1f[i], real_freq_1f[i], marker='o', linestyle='none', markersize=8,color='black')
    axs[1].plot(kx_values_2f[i], real_freq_2f[i], marker='X', linestyle='none', markersize=8,color='red')
    axs[1].plot(kx_values_2f[i], real_freq_2f[i],color='red',linestyle=linestyles[i],label=stokes_label[i])


axs[0].set_ylabel(r'$s_\mathrm{max}/\Omega$')

title=r"Z={0:1.2f}, $\delta$={1:4.0e}".format(metal, delta)
axs[0].set_title(title,weight='bold')

lines1, labels1 = axs[0].get_legend_handles_labels()
legend=axs[0].legend(lines1, labels1, loc='upper left', frameon=False, ncol=1, handletextpad=-0.5,fontsize=fontsize/2)

axs[1].set_ylabel(r'$\omega/\Omega$')
axs[1].set_xlabel(r'$k_xH_g$')

lines1, labels1 = axs[1].get_legend_handles_labels()
legend=axs[1].legend(lines1, labels1, loc='lower left', frameon=False, ncol=1,handletextpad=0.2, fontsize=fontsize/1.5)

fname = 'stratsi_compare_stokes'
plt.savefig(fname,dpi=150)


'''
compare dust density perturbation as function of height (normalized to Hdust)
'''

args = parser.parse_args()
if(args.mode):
    m = np.int(args.mode[0])
else:
    m = 0

plt.rc('font',size=fontsize/1.5,weight='bold')

fig, axs = plt.subplots(2, sharex=True, sharey=False, gridspec_kw={'hspace': 0.1}, figsize=(8,4))
plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.15)

for i in range(0,2):
    if i == 0:
        label = [r'one-fluid, real', r'one-fluid, imag', r'two-fluid, real', r'two-fluid, imag']
    else:
        label = ['', '', '', '']
    
    axs[i].plot(z_1f, delrhod_1f[i][m].real, linewidth=2, label=label[0], color='black')
    axs[i].plot(z_1f, delrhod_1f[i][m].imag, linewidth=2, label=label[1], color='m')

    axs[i].plot(z_2f, delrhod_2f[i][m].real, linewidth=2, label=label[2], color='red', linestyle='dashed')
    axs[i].plot(z_2f, delrhod_2f[i][m].imag, linewidth=2, label=label[3], color='c', linestyle='dashed')

    axs[i].set_ylabel(r'$\delta\rho_d/\rho_d$')

    axs[i].annotate(r"$St$={0:5.3f}".format(stokes[i]) +"\n"+r"s={0:4.2f}$\Omega$".format(growth_rates_2f[i][m]), xy=(0.1, 0.3))

    if i == 0:
        lines1, labels1 = axs[i].get_legend_handles_labels()
        axs[i].legend(lines1, labels1, loc='upper right', frameon=False, ncol=1, labelspacing=0.3, handletextpad=0.1)
        title=r"Z={0:1.2f}, $k_xH_g$={1:3.0f}, $\delta$={2:4.0e}".format(metal, kx_values_2f[i][m], delta)
        axs[i].set_title(title,weight='bold')

    if i == 1:
        axs[i].set_xlabel(r'$z/H_d$',fontweight='bold')

plt.xlim(0,5)

fname = 'stratsi_compare_stokes_eigenfunc'
plt.savefig(fname,dpi=150)
