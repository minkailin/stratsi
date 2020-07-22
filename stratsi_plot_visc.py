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
    
data_file = ['stratsi_modes.h5','novisc/stratsi_modes.h5']
    
ks        = []
z         = []
freqs     = []

W         = []
Q         = []
del_rhod  = []
Ugx       = []
Ugy       = []
Ugz       = []
Udx       = []
Udy       = []
Udz       = []

'''
read in two-fluid data 
'''

for i in range(0,2): 

  with h5py.File(data_file[i],'r') as infile:
  
    ks_arr= infile['scales']['kx_space'][:]
    zaxis = infile['scales']['z'][:]
    zmax  = infile['scales']['zmax'][()]

    ks.append(np.array(ks_arr))
    z.append(np.array(zaxis))

    evals = []
    eig_W = []
    eig_Q = []
    eig_Ugx= []
    eig_Ugy= []
    eig_Ugz= []
    eig_Udx= []
    eig_Udy= []
    eig_Udz= []
  
    for k_i in infile['tasks']:
      evals.append(infile['tasks'][k_i]['freq'][:])
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
      m = np.argmin(np.abs(ks_arr-plot_kx))

    kx       = ks_arr[m]
    sigma    = evals[m]

    freqs.append(evals)
    
    growth      = sigma.real
    freq        =-sigma.imag
    
    if(args.sig):
      g1 = np.argmin(np.abs(sigma-eigenv))
    else:
      g1 = np.argmax(growth)

    sgrow = growth[g1]
    ofreq = freq[g1]
    print("two-fluid model: kx, growth, freq = {0:1.2e} {1:13.6e} {2:13.6e}".format(kx, sgrow, ofreq))

    W1  = np.array(eig_W[m][g1])
    Q1  = np.array(eig_Q[m][g1])
    Ugx1= np.array(eig_Ugx[m][g1])
    Ugy1= np.array(eig_Ugy[m][g1])
    Ugz1= np.array(eig_Ugz[m][g1])
    Udx1= np.array(eig_Udx[m][g1])
    Udy1= np.array(eig_Udy[m][g1])
    Udz1= np.array(eig_Udz[m][g1])
      
    g2      = np.argmax(np.abs(W1+Q1))
    norm    = W1[g2] + Q1[g2]

    W1   /= norm
    Q1   /= norm
    Ugx1 /= norm
    Ugy1 /= norm
    Ugz1 /= norm
    Udx1 /= norm
    Udy1 /= norm
    Udz1 /= norm

    W.append(W1)
    Q.append(Q1)
    Ugx.append(Ugx1)
    Ugy.append(Ugy1)
    Ugz.append(Ugz1)
    Udx.append(Udx1)
    Udy.append(Udy1)
    Udz.append(Udz1)
    
    del_rhod.append(W1 + Q1)

'''
plotting parameters
'''

fontsize= 24
nlev    = 128
nclev   = 6
cmap    = plt.cm.inferno

ymax = 1
xmin = 0.0
xmax = np.amax(z)

'''
plot max growth rates as func of kx
'''
plt.rc('font',size=fontsize,weight='bold')

fig, axs = plt.subplots(2, sharex=True, sharey=False, gridspec_kw={'hspace': 0.1}, figsize=(8,6))

plt.subplots_adjust(left=0.2, right=0.95, top=0.9, bottom=0.15)
plt.xscale('log')


for i, k in enumerate(ks[0]):
    g1 = np.argmax(freqs[0][i].real)
    if i == 0:
        lab = r'viscous'
    else:
        lab = ''
    axs[0].plot(k, freqs[0][i][g1].real, marker='o', linestyle='none', markersize=8, label=lab,color='black')

for i, k in enumerate(ks[1]):
    g1 = np.argmax(freqs[1][i].real)
    if i == 0:
        lab = r'inviscid'
    else:
        lab = ''
    axs[0].plot(k, freqs[1][i][g1].real, marker='X', linestyle='none', markersize=8, label=lab,color='red')

     
axs[0].set_ylabel(r'$s_\mathrm{max}/\Omega$')
lines1, labels1 = axs[0].get_legend_handles_labels()
legend=axs[0].legend(lines1, labels1, loc='upper left', frameon=False, ncol=1, handletextpad=-0.5,fontsize=fontsize/2)

title=r"Z={0:1.2f}, St={1:4.0e}, $\delta$={2:4.0e}".format(metal, stokes, delta)
axs[0].set_title(title,weight='bold')

for i, k in enumerate(ks[0]):
    g1 = np.argmax(freqs[0][i].real)
    if i == 0:
        lab = r'viscous'
    else:
        lab = ''
    axs[1].plot(k, -freqs[0][i][g1].imag, marker='o', linestyle='none', markersize=8, label=lab,color='black')
    
for i, k in enumerate(ks[1]):
    g1 = np.argmax(freqs[1][i].real)
    if i == 0:
        lab = r'inviscid'
    else:
        lab = ''
    axs[1].plot(k, -freqs[1][i][g1].imag, marker='X', linestyle='none', markersize=8, label=lab,color='red')
    
axs[1].set_ylabel(r'$\omega/\Omega$')
axs[1].set_xlabel(r'$k_xH_g$')

fname = 'stratsi_plot_visc_growth_max'
plt.savefig(fname,dpi=150)

'''
plot eigenfunctions
'''

plt.rc('font',size=fontsize/1.5,weight='bold')

fig, axs = plt.subplots(5, sharex=True, sharey=False, gridspec_kw={'hspace': 0.1}, figsize=(8,7.5))
plt.subplots_adjust(left=0.18, right=0.95, top=0.95, bottom=0.125)

axs[0].plot(z[0], del_rhod[0].real, linewidth=2, label=r'visc., real', color='black')
axs[0].plot(z[0], del_rhod[0].imag, linewidth=2, label=r'visc., imag', color='m')

axs[0].plot(z[1], del_rhod[1].real, linewidth=2, label=r'invis., real', color='red', linestyle='dashed')
axs[0].plot(z[1], del_rhod[1].imag, linewidth=2, label=r'invis., imag', color='c', linestyle='dashed')

axs[0].set_ylabel(r'$\delta\rho_d/\rho_d$')
lines1, labels1 = axs[0].get_legend_handles_labels()
axs[0].legend(lines1, labels1, loc=(0.,-0.07), frameon=False, ncol=1, labelspacing=0.3, handletextpad=0.1)

title=r"Z={0:1.2f}, St={1:4.0e}, $\delta$={2:4.0e}".format(metal, stokes, delta)
axs[0].set_title(title,weight='bold')

axs[1].plot(z[0], W[0].real, linewidth=2, label=r'viscous, real', color='black')
axs[1].plot(z[0], W[0].imag, linewidth=2, label=r'inviscid, imag', color='m')

axs[1].plot(z[1], W[1].real, linewidth=2, label=r'viscous, real', color='red', linestyle='dashed')
axs[1].plot(z[1], W[1].imag, linewidth=2, label=r'inviscid, imag', color='c', linestyle='dashed')

axs[1].set_ylabel(r'$\delta\rho_g/\rho_{g}$')
axs[1].ticklabel_format(axis='y', style='sci',scilimits=(-2,2))
axs[1].yaxis.set_major_formatter(FormatStrFormatter('%3.0e'))

axs[2].plot(z[0], np.abs(Ugx[0]), linewidth=2, label=r'viscous, gas', color='black')
axs[2].plot(z[0], np.abs(Udx[0]), linewidth=2, label=r'viscous, dust', color='m')
axs[2].plot(z[1], np.abs(Ugx[1]), linewidth=2, label=r'inviscid, gas', color='red', linestyle='dashed')
axs[2].plot(z[1], np.abs(Udx[1]), linewidth=2, label=r'inviscid, gas', color='c', linestyle='dashed')
axs[2].set_ylabel(r'$|\delta v_{x}|$')

lines1, labels1 = axs[2].get_legend_handles_labels()
axs[2].legend(lines1, labels1, loc='right', frameon=False, ncol=1, labelspacing=0.2, handletextpad=0.1)

axs[3].plot(z[0], np.abs(Ugy[0]), linewidth=2, label=r'viscous, gas', color='black')
axs[3].plot(z[0], np.abs(Udy[0]), linewidth=2, label=r'viscous, dust', color='m')
axs[3].plot(z[1], np.abs(Ugy[1]), linewidth=2, label=r'inviscid, gas', color='red', linestyle='dashed')
axs[3].plot(z[1], np.abs(Udy[1]), linewidth=2, label=r'inviscid, gas', color='c', linestyle='dashed')
axs[3].set_ylabel(r'$|\delta v_{y}|$')

ymax = np.amax(np.abs(Udy))
axs[3].annotate(r"$k_xH_g$={0:3.0f}".format(ks[0][m])+"\n"+r"s={0:4.2f}$\Omega$ (viscous)".format(np.amax(freqs[0][m].real)), xy=(0.6*xmax, 0.5*ymax))
axs[3].annotate(r"s={0:4.2f}$\Omega$ (inviscid)".format(np.amax(freqs[1][m].real)), xy=(0.6*xmax, 0.3*ymax))

axs[4].plot(z[0], np.abs(Ugz[0]), linewidth=2, label=r'viscous, gas', color='black')
axs[4].plot(z[0], np.abs(Udz[0]), linewidth=2, label=r'viscous, dust', color='m')
axs[4].plot(z[1], np.abs(Ugz[1]), linewidth=2, label=r'inviscid, gas', color='red', linestyle='dashed')
axs[4].plot(z[1], np.abs(Udz[1]), linewidth=2, label=r'inviscid, gas', color='c', linestyle='dashed')
axs[4].set_ylabel(r'$|\delta v_{z}|$')

axs[4].set_xlabel(r'$z/H_g$',fontweight='bold')

plt.xlim(xmin,xmax)

fname = 'stratsi_plot_visc_eigenfunc'
plt.savefig(fname,dpi=150)


