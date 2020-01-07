import sys
import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt
import h5py
import argparse

'''
process command line arguements
'''
parser = argparse.ArgumentParser()
parser.add_argument("--mode", nargs='*', help="select mode number")
parser.add_argument("--kx", nargs='*', help="select kx")
args = parser.parse_args()
if(args.mode):
    plot_mode = np.int(args.mode[0])
else:
    plot_mode = 0

if(args.kx):
    plot_kx = np.float(args.kx[0])
else:
    plot_kx = 400.0
    
#print("plotting mode number {0:3d}".format(plot_mode))
    
'''
read in one-fluid data 
'''

with h5py.File('stratsi_1fluid_modes.h5','r') as infile:
  
  ks_1f    = infile['scales']['kx_space'][:]
  freqs_1f = infile['scales']['eig_freq'][:]
  z_1f     = infile['scales']['z'][:]
  zmax_1f  = infile['scales']['zmax'][()]

  eig_W1f = []
  eig_Q1f = []
  eig_Ux= []
  eig_Uy= []
  eig_Uz= []

  for k_i in infile['tasks']:
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
W1f         = eig_W1f[n]
Q1f         = eig_Q1f[n]
Ux          = eig_Ux[n]
Uy          = eig_Uy[n]
Uz          = eig_Uz[n]

print("one-fluid model: kx, growth, freq = {0:1.2e} {1:13.6e} {2:13.6e}".format(kx_1f, sigma_1f.real, -sigma_1f.imag))


'''
read in two-fluid data
'''

with h5py.File('stratsi_modes.h5','r') as infile:
  
  ks    = infile['scales']['kx_space'][:]
  freqs = infile['scales']['eig_freq'][:]
  z     = infile['scales']['z'][:]
  zmax  = infile['scales']['zmax'][()]
  
  eig_W = []
  eig_Q = []
  eig_Ugx= []
  eig_Ugy= []
  eig_Ugz= []
  eig_Udx= []
  eig_Udy= []
  eig_Udz= []
  
  for k_i in infile['tasks']:
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
W         = eig_W[m]
Q         = eig_Q[m]
Ugx          = eig_Ugx[m]
Ugy          = eig_Ugy[m]
Ugz          = eig_Ugz[m]
Udx          = eig_Udx[m]
Udy          = eig_Udy[m]
Udz          = eig_Udz[m]

print("two-fluid model: kx, growth, freq = {0:1.2e} {1:13.6e} {2:13.6e}".format(kx, sigma.real, -sigma.imag))


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
plot eigenfunctions
'''

fig = plt.figure(figsize=(8,4.5))
ax = fig.add_subplot()
plt.subplots_adjust(left=0.18, right=0.95, top=0.95, bottom=0.2)

plt.xlim(xmin,xmax)
plt.ylim(-ymax,ymax)

Uz_norm = np.conj(Uz[0])*Uz
Uz_norm/= np.amax(np.abs(Uz_norm))

plt.plot(z_1f, np.real(Uz_norm), linewidth=2, label=r're, one-fluid')#, color='b')
plt.plot(z_1f, np.imag(Uz_norm), linewidth=2, label=r'im, one-fluid')#, linestyle='dashed',color='b')

# plt.gca().set_prop_cycle(None) #resets color cycle

# Udz_norm = np.conj(Udz[0])*Udz
# Udz_norm/= np.amax(np.abs(Udz_norm))

# plt.plot(z, np.real(Udz_norm), linewidth=2, label=r're, two-fluid',linestyle='dashed')
# plt.plot(z, np.imag(Udz_norm), linewidth=2, label=r'im, two-fluid',linestyle='dashed')

plt.rc('font',size=fontsize,weight='bold')

lines1, labels1 = ax.get_legend_handles_labels()
legend=ax.legend(lines1, labels1, loc='upper right', frameon=False, ncol=1, fontsize=fontsize/2)

#title = r"$k_xH_g=${0:4.0f}, s={1:5.3f}$\Omega$".format(kx_1f, sigma_1f.real)
#plt.title(title,weight='bold')

plt.xticks(fontsize=fontsize,weight='bold')
plt.xlabel(r'$z/H_g$',fontsize=fontsize)

plt.yticks(fontsize=fontsize,weight='bold')
plt.ylabel(r'$\delta v_{z}$', fontsize=fontsize)

fname = 'stratsi_plot_vz'
plt.savefig(fname,dpi=150)

#########################################################################################################


fig = plt.figure(figsize=(8,4.5))
ax = fig.add_subplot()
plt.subplots_adjust(left=0.18, right=0.95, top=0.95, bottom=0.2)

plt.xlim(xmin,xmax)
plt.ylim(-ymax,ymax)

Udz_norm = np.conj(Udz[0])*Udz
Udz_norm/= np.amax(np.abs(Udz_norm))

plt.plot(z, np.real(Udz_norm), linewidth=2, label=r're, two-fluid')
plt.plot(z, np.imag(Udz_norm), linewidth=2, label=r'im, two-fluid')

plt.rc('font',size=fontsize,weight='bold')

lines1, labels1 = ax.get_legend_handles_labels()
legend=ax.legend(lines1, labels1, loc='upper right', frameon=False, ncol=1, fontsize=fontsize/2)

plt.xticks(fontsize=fontsize,weight='bold')
plt.xlabel(r'$z/H_g$',fontsize=fontsize)

plt.yticks(fontsize=fontsize,weight='bold')
plt.ylabel(r'$\delta v_{dz}$', fontsize=fontsize)

fname = 'stratsi_plot_vdz'
plt.savefig(fname,dpi=150)

#########################################################################################################

fig = plt.figure(figsize=(8,4.5))
ax = fig.add_subplot()
plt.subplots_adjust(left=0.18, right=0.95, top=0.95, bottom=0.2)

plt.xlim(xmin,xmax)
plt.ylim(0,ymax)

Q1f_abs = np.abs(Q1f)
Q1f_abs/= np.amax(Q1f_abs)
plt.plot(z_1f, Q1f_abs, linewidth=2, label=r'one-fluid')

Q_abs = np.abs(Q)
Q_abs/= np.amax(Q_abs)
plt.plot(z, Q_abs, linewidth=2, label=r'two-fluid',linestyle='dashed')

plt.rc('font',size=fontsize,weight='bold')

lines1, labels1 = ax.get_legend_handles_labels()
legend=ax.legend(lines1, labels1, loc='upper right', frameon=False, ncol=1, fontsize=fontsize/2)

plt.xticks(fontsize=fontsize,weight='bold')
plt.xlabel(r'$z/H_g$',fontsize=fontsize)

plt.yticks(fontsize=fontsize,weight='bold')
plt.ylabel(r'$|\delta\epsilon/\epsilon|$', fontsize=fontsize)

fname = 'stratsi_plot_Q'
plt.savefig(fname,dpi=150)

#########################################################################################################
'''
compare vz to horizontal velocities
'''

fig = plt.figure(figsize=(8,4.5))
ax = fig.add_subplot()
plt.subplots_adjust(left=0.18, right=0.95, top=0.95, bottom=0.2)

plt.xlim(xmin,xmax)
#plt.ylim(0,ymax)

theta_1f_vert = np.abs(Uz)/np.amax(abs(Uz))
theta_1f_horz = np.sqrt(np.abs(Ux)**2 + np.abs(Uy)**2)/np.amax(abs(Uz))
plt.plot(z_1f, theta_1f_vert, linewidth=2, label=r'one-fluid, $v_z$')
plt.plot(z_1f, theta_1f_horz, linewidth=2, label=r'one-fluid, $v_{x,y}$')

plt.gca().set_prop_cycle(None)

theta_vert = np.abs(Udz)/np.amax(abs(Udz))
theta_horz = np.sqrt(np.abs(Udx)**2 + np.abs(Udy)**2)/np.amax(abs(Udz))
plt.plot(z, theta_vert, linewidth=2, label=r'two-fluid, $v_{dz}$', linestyle='dashed')
plt.plot(z, theta_horz, linewidth=2, label=r'two-fluid, $v_{dx,dy}$', linestyle='dashed')

plt.rc('font',size=fontsize,weight='bold')

lines1, labels1 = ax.get_legend_handles_labels()
legend=ax.legend(lines1, labels1, loc='upper right', frameon=False, ncol=1, fontsize=fontsize/2)

plt.xticks(fontsize=fontsize,weight='bold')
plt.xlabel(r'$z/H_g$',fontsize=fontsize)

plt.yticks(fontsize=fontsize,weight='bold')
#plt.ylabel(r'$\theta$', fontsize=fontsize)

fname = 'stratsi_plot_theta'
plt.savefig(fname,dpi=150)
