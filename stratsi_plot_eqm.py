import sys
import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
import h5py
import argparse
from scipy.integrate import simps

from stratsi_params import alpha, delta, stokes, metal, epsilon, vdz, dvdz, rhog, dln_rhog, viscosity_pert, zmax, eta_hat

'''
energy analysis
'''
#read in background vertical profiles of vgx, vgy, vdx, vdy

horiz_eqm  = h5py.File('./eqm_horiz.h5', 'r')
z          = horiz_eqm['z'][:]
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

vdx_lhs = vdz(z)*np.gradient(vdx,z)
vdx_rhs = 2.0*vdy - (vdx - vgx)/stokes

vdy_lhs = vdz(z)*np.gradient(vdy,z)
vdy_rhs =-0.5*vdx - (vdy - vgy)/stokes

if viscosity_pert == False:
    alpha = 0.0

vgx_lhs = -(alpha/rhog(z))*np.gradient(rhog(z)*dvgx,z)
vgx_rhs = 2.0*vgy + 2.0*eta_hat - (eps2f/stokes)*(vgx - vdx)

vgy_lhs = -(alpha/rhog(z))*np.gradient(rhog(z)*dvgy,z)
vgy_rhs = -0.5*vgx - (eps2f/stokes)*(vgy - vdy)


'''
plotting parameters
'''

fontsize= 24
nlev    = 128
nclev   = 6
cmap    = plt.cm.inferno

'''
plot eigenfunctions
'''

plt.rc('font',size=fontsize/1.5,weight='bold')

fig = plt.figure(figsize=(8,4.5))
ax = fig.add_subplot()
plt.subplots_adjust(left=0.2, right=0.95, top=0.9, bottom=0.2)

plt.xlim(0,zmax)

#plt.plot(z, vdx_lhs, linewidth=2,label='vdx, lhs')
#plt.plot(z, vdx_rhs, linewidth=2,label='vdx, rhs',linestyle='dashed')

#plt.plot(z, vdy_lhs, linewidth=2,label='vdy, lhs')
#plt.plot(z, vdy_rhs, linewidth=2,label='vdy, rhs',linestyle='dashed')

#plt.plot(z, vgx_lhs, linewidth=2,label='vgx, lhs')
#plt.plot(z, vgx_rhs, linewidth=2,label='vgx, rhs',linestyle='dashed')

plt.plot(z, vgy_lhs, linewidth=2,label='vgx, lhs')
plt.plot(z, vgy_rhs, linewidth=2,label='vgx, rhs',linestyle='dashed')

plt.rc('font',size=fontsize,weight='bold')

lines1, labels1 = ax.get_legend_handles_labels()
legend=ax.legend(lines1, labels1, loc='upper right', frameon=False, ncol=1, fontsize=fontsize/2, labelspacing=0.4)

plt.xticks(fontsize=fontsize,weight='bold')
plt.xlabel(r'$z/H_g$',fontsize=fontsize)

fname = 'stratsi_plot_eqm'
plt.savefig(fname,dpi=150)
