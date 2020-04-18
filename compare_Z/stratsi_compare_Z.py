import sys
import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
import h5py
import argparse
from scipy.integrate import simps

#case parameters
#metal = 0.03
stokes= 0.01
delta = 1e-6

Z_vals     = ['0.01','0.03','0.1']

'''
one-fluid results
'''

filenames_1f = ['stratsi_1fluid_modes_Z'+x+'.h5' for x in Z_vals]

kx_values_1f    = []
growth_rates_1f = []
real_freq_1f    = []

for n, fname in enumerate(filenames_1f):

  with h5py.File(fname,'r') as infile:
  
    kx = infile['scales']['kx_space'][:]
    kx_values_1f.append(kx)

    freqs = []
    for k_i in infile['tasks']:
      freqs.append(infile['tasks'][k_i]['freq'][:])

  growth_rates_1f.append([])
  real_freq_1f.append([])
  
  for i, kxvals in enumerate(kx):
    
    sigma      = freqs[i]

    growth     = sigma.real
    freq       =-sigma.imag

    g1          = np.argmax(growth)

    growth_rates_1f[n].append(growth[g1])
    real_freq_1f[n].append(freq[g1])

'''
two-fluid results
'''

filenames_2f = ['stratsi_modes_Z'+x+'.h5' for x in Z_vals]
    
kx_values_2f    = []
growth_rates_2f = []
real_freq_2f    = []

for n, fname in enumerate(filenames_2f):

  with h5py.File(fname,'r') as infile:
  
    kx = infile['scales']['kx_space'][:]
    kx_values_2f.append(kx)

    freqs = []
    for k_i in infile['tasks']:
      freqs.append(infile['tasks'][k_i]['freq'][:])

  growth_rates_2f.append([])
  real_freq_2f.append([])
  
  for i, kxvals in enumerate(kx):
    
    sigma    = freqs[i]

    growth   = sigma.real
    freq     =-sigma.imag

    g1          = np.argmax(growth)

    growth_rates_2f[n].append(growth[g1])
    real_freq_2f[n].append(freq[g1])

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
linestyles=['solid','solid','dashed']
etas = [r'$Z=$'+x for x in Z_vals]
for i in range(0,3,2):
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
    axs[1].plot(kx_values_2f[i], real_freq_2f[i],color='red',linestyle=linestyles[i],label=etas[i])


axs[0].set_ylabel(r'$s_\mathrm{max}/\Omega$')

title=r"St={0:4.0e}, $\delta$={1:4.0e}".format(stokes, delta)
axs[0].set_title(title,weight='bold')

lines1, labels1 = axs[0].get_legend_handles_labels()
legend=axs[0].legend(lines1, labels1, loc='upper left', frameon=False, ncol=1, handletextpad=-0.5,fontsize=fontsize/2)

axs[1].set_ylabel(r'$\omega/\Omega$')
axs[1].set_xlabel(r'$k_xH_g$')

lines1, labels1 = axs[1].get_legend_handles_labels()
legend=axs[1].legend(lines1, labels1, loc='lower left', frameon=False, ncol=1,handletextpad=0.2, fontsize=fontsize/1.5)

fname = 'stratsi_compare_Z'
plt.savefig(fname,dpi=150),

