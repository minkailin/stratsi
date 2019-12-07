import sys
import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt
import h5py

with h5py.File('stratsi_1fluid_modes.h5','r') as infile:
  
  ks    = infile['scales']['kx_space'][:]
  sigma = infile['scales']['eig_freq'][:]
  zaxis = infile['scales']['z'][:]

  # print(sigma[0])
  # print(sigma[1])
  # print(zaxis)
  # print(ks)
  
  eig_W = []
  eig_Q = []
  eig_Ux= []
  eig_Uy= []
  eig_Uz= []
# 
  for k_i in infile['tasks']:
    eig_W.append(infile['tasks'][k_i]['eig_W'][:])
    eig_Q.append(infile['tasks'][k_i]['eig_Q'][:])
    eig_Ux.append(infile['tasks'][k_i]['eig_Ux'][:])
    eig_Uy.append(infile['tasks'][k_i]['eig_Uy'][:])
    eig_Uz.append(infile['tasks'][k_i]['eig_Uz'][:])


    
W = np.shape(eig_W)
    
print(W)
    
'''
#plotting parameters
'''


'''
fontsize= 24
nlev    = 128
nclev   = 6
cmap    = plt.cm.inferno

fig = plt.figure(figsize=(8,4.5))
ax = fig.add_subplot()
plt.subplots_adjust(left=0.18, right=0.95, top=0.95, bottom=0.2)

z    = domain_EVP.grid(0, scales=16)
Uz.set_scales(scales=16)

max_Uz = np.amax(np.abs(Uz['g']))

Uz_norm = np.conj(Uz['g'][0])*Uz['g']
plt.plot(z, np.real(Uz_norm)/np.amax(np.abs(Uz_norm)), linewidth=2, label=r'real')
plt.plot(z, np.imag(Uz_norm)/np.amax(np.abs(Uz_norm)), linewidth=2, label=r'imaginary')

#plt.plot(z, np.abs(Uz['g'])/max_Uz, linewidth=2, label=r'real')

plt.rc('font',size=fontsize,weight='bold')

lines1, labels1 = ax.get_legend_handles_labels()
legend=ax.legend(lines1, labels1, loc='upper right', frameon=False, ncol=1, fontsize=fontsize/2)

plt.xticks(fontsize=fontsize,weight='bold')
plt.xlabel(r'$z/H_g$',fontsize=fontsize)

plt.yticks(fontsize=fontsize,weight='bold')
plt.ylabel(r'$\delta v_{z}/|\delta v_{z}|_{max}$', fontsize=fontsize)
#plt.ylabel(r'$\delta v_{z}$', fontsize=fontsize)

fname = 'stratsi_vz_1fluid'
plt.savefig(fname,dpi=150)

'''
'''

######################################################################################################
fig = plt.figure(figsize=(8,4.5))
ax = fig.add_subplot()
plt.subplots_adjust(left=0.18, right=0.95, top=0.95, bottom=0.2)

z    = domain_EVP.grid(0, scales=16)
W.set_scales(scales=16)
Wnorm = np.conj(W['g'][0])/np.power(np.abs(W['g'][0]),2)

plt.plot(z, np.real(W['g']*Wnorm), linewidth=2, label=r'real')
plt.plot(z, np.imag(W['g']*Wnorm), linewidth=2, label=r'imaginary')

plt.rc('font',size=fontsize,weight='bold')

lines1, labels1 = ax.get_legend_handles_labels()
legend=ax.legend(lines1, labels1, loc='upper right', frameon=False, ncol=1, fontsize=fontsize/2)

plt.xticks(fontsize=fontsize,weight='bold')
plt.xlabel(r'$z/H_g$',fontsize=fontsize)

plt.yticks(fontsize=fontsize,weight='bold')
plt.ylabel(r'$\delta \rho_{g}/\rho_{g}$', fontsize=fontsize)

fname = 'stratsi_W_1fluid'
plt.savefig(fname,dpi=150)

######################################################################################################
fig = plt.figure(figsize=(8,4.5))
ax = fig.add_subplot()
plt.subplots_adjust(left=0.18, right=0.95, top=0.95, bottom=0.2)

z    = domain_EVP.grid(0, scales=16)
Q.set_scales(scales=16)
Qmax = np.amax(np.abs(Q['g']))
#Qnorm = np.conj(Q['g'][0])/np.power(np.abs(Q['g'][0]),2)

plt.plot(z, np.real(Q['g'])/Qmax, linewidth=2, label=r'real')
plt.plot(z, np.imag(Q['g'])/Qmax, linewidth=2, label=r'imaginary')

plt.rc('font',size=fontsize,weight='bold')

lines1, labels1 = ax.get_legend_handles_labels()
legend=ax.legend(lines1, labels1, loc='upper right', frameon=False, ncol=1, fontsize=fontsize/2)

plt.xticks(fontsize=fontsize,weight='bold')
plt.xlabel(r'$z/H_g$',fontsize=fontsize)

plt.yticks(fontsize=fontsize,weight='bold')
plt.ylabel(r'$\delta \epsilon/\epsilon$', fontsize=fontsize)

fname = 'stratsi_Q_1fluid'
plt.savefig(fname,dpi=150)

######################################################################################################



fig = plt.figure(figsize=(8,4.5))
ax = fig.add_subplot()
plt.subplots_adjust(left=0.18, right=0.95, top=0.95, bottom=0.2)

z    = domain_EVP.grid(0, scales=16)
epsilon0.set_scales(scales=16)
plt.plot(z, np.real(epsilon0['g']),linewidth=2, label='numerical solution')

plt.rc('font',size=fontsize,weight='bold')

lines1, labels1 = ax.get_legend_handles_labels()
legend=ax.legend(lines1, labels1, loc='upper right', frameon=False, ncol=1, fontsize=fontsize/2)

plt.xticks(fontsize=fontsize,weight='bold')
plt.xlabel('$z/H_g$',fontsize=fontsize)

plt.yticks(fontsize=fontsize,weight='bold')
plt.ylabel(r'$\epsilon$',fontsize=fontsize)

fname = 'stratsi_epsilon_1fluid'
plt.savefig(fname,dpi=150)

fig = plt.figure(figsize=(8,4.5))
ax = fig.add_subplot()
plt.subplots_adjust(left=0.18, right=0.95, top=0.95, bottom=0.2)

z    = domain_EVP.grid(0, scales=16)
vz0.set_scales(scales=16)
vy0.set_scales(scales=16)

plt.plot(z, np.real(vz0['g']),linewidth=2, label=r'$v_z/c_s$')
plt.plot(z, np.real(vy0['g']),linewidth=2, label=r'$v_y/c_s$')

plt.rc('font',size=fontsize,weight='bold')

lines1, labels1 = ax.get_legend_handles_labels()
legend=ax.legend(lines1, labels1, loc='upper right', frameon=False, ncol=1, fontsize=fontsize/2)

plt.xticks(fontsize=fontsize,weight='bold')
plt.xlabel('$z/H_g$',fontsize=fontsize)

plt.yticks(fontsize=fontsize,weight='bold')
plt.ylabel(r'velocities',fontsize=fontsize)

fname = 'stratsi_vzy_1fluid'
plt.savefig(fname,dpi=150)

'''
