"""
stratified streaming instability 

equilibrium vertical profiles of horizontal velocities 
"""

'''
import parameters and functions
'''
from stratsi_params import *

'''
process command line arguements
'''
parser = argparse.ArgumentParser()
parser.add_argument("--xlim", "-xl", nargs='*', help="set horizontal axis range")
parser.add_argument("--ylim", "-yl", nargs='*', help="set vertical axis range")
args = parser.parse_args()
if(args.xlim):
#    print(args.xlim)
    #xlim_float = [float(x) for x in args.xlim]
    #xbounds = np.array(xlim_float) 
    xbounds = np.array(args.xlim).astype(np.float) 

if(args.ylim):
    ybounds = np.array(args.ylim).astype(np.float) 
        
'''
numerical parameters
'''
ncc_cutoff = 1e-12
tolerance  = 1e-12

'''
setup grid and problem 
'''
z_basis = de.Chebyshev('z', nz_vert, interval=(zmin,zmax), dealias=2)
domain  = de.Domain([z_basis], np.float64, comm=MPI.COMM_SELF)

if viscosity_eqm == True:
    problem = de.LBVP(domain, variables=['vgx', 'vgx_prime', 'vgy', 'vgy_prime', 'vdx', 'vdy'], ncc_cutoff=ncc_cutoff)
if viscosity_eqm == False:
    problem = de.LBVP(domain, variables=['vgx', 'vgy', 'vdx', 'vdy'], ncc_cutoff=ncc_cutoff)

'''
constant coefficients 
'''
problem.parameters['stokes']   = stokes
problem.parameters['alpha']    = alpha
problem.parameters['eta_hat']  = eta_hat
    
'''
non-constant coefficients:
'''

z = domain.grid(0)
    
eps_profile      = domain.new_field()
dln_rhog_profile = domain.new_field()
vdz_profile      = domain.new_field()

eps_profile['g']      = epsilon(z)
dln_rhog_profile['g'] = dln_rhog(z)
vdz_profile['g']      = vdz(z)

problem.parameters['eps_profile']      = eps_profile
problem.parameters['dln_rhog_profile'] = dln_rhog_profile
problem.parameters['vdz_profile']      = vdz_profile

'''
full gas equations
'''
if viscosity_eqm ==  True:
    problem.add_equation("alpha*dz(vgx_prime) + alpha*dln_rhog_profile*vgx_prime + 2*vgy - eps_profile*(vgx - vdx)/stokes = -2*eta_hat")
    problem.add_equation("dz(vgx) - vgx_prime = 0")
    problem.add_equation("alpha*dz(vgy_prime) + alpha*dln_rhog_profile*vgy_prime - 0.5*vgx - eps_profile*(vgy - vdy)/stokes = 0")
    problem.add_equation("dz(vgy) - vgy_prime = 0")
    
'''
gas equations in the inviscid limit
'''
if viscosity_eqm == False:
    problem.add_equation("2*vgy - eps_profile*(vgx - vdx)/stokes = -2*eta_hat")
    problem.add_equation("-0.5*vgx - eps_profile*(vgy - vdy)/stokes = 0")

'''
dust equations
the quadratic terms in vdust are small, they don't make a practical difference.
''' 
problem.add_equation("vdz_profile*dz(vdx) - 2.0*vdy + (vdx - vgx)/stokes = 0")
problem.add_equation("vdz_profile*dz(vdy) + 0.5*vdx + (vdy - vgy)/stokes = 0")
    
'''
boundary conditions for full problem
use analytic solution at z=infinity or where epsilon -> 0 (pure gas disk) to set vgx, vgy, vdx, vdy 
set vgx and vgy to be symmetric about midplane
'''

if viscosity_eqm == True: #full 2nd order ODE in gas velocities 
    problem.add_bc("left(vgx_prime)      = 0")
    problem.add_bc("left(vgy_prime)      = 0")
    problem.add_bc("right(vgx)           = 0")
    problem.add_bc("right(vgy)           =-eta_hat")
    
problem.add_bc("right(vdx)            = -2.0*stokes*eta_hat/(1.0 + stokes**2.0)")
problem.add_bc("right(vdy)            = -eta_hat/(1.0 + stokes**2.0)")

'''
build problem and solver
'''
solver = problem.build_solver()

'''
solve equations (linear problem)
'''
solver.solve()

'''
extract solutions (to fine grid)
'''

z    = domain.grid(0, scales=domain.dealias)
vgx  = solver.state['vgx']
vgy  = solver.state['vgy']
vdx  = solver.state['vdx']
vdy  = solver.state['vdy']

vgx.set_scales(domain.dealias)
vgy.set_scales(domain.dealias)
vdx.set_scales(domain.dealias)
vdy.set_scales(domain.dealias)

vgx_norm = vgx['g']
vgy_norm = vgy['g']
vdx_norm = vdx['g']
vdy_norm = vdy['g']

'''
plot equilibrium velocities 
'''

fontsize= 24
nlev    = 128
nclev   = 6
cmap    = plt.cm.inferno

fig = plt.figure(figsize=(8,4.5))
ax = fig.add_subplot()
plt.subplots_adjust(left=0.18, right=0.95, top=0.95, bottom=0.2)

if(args.xlim):
    xmin  = xbounds[0]
    xmax  = xbounds[1]
else:
    xmin  = zmin
    xmax  = zmax

if(args.ylim):
    ymin = ybounds[0]
    ymax = ybounds[1]
else:
    x1 = np.argmin(np.absolute(z - xmin))
    x2 = np.argmin(np.absolute(z - xmax))    
    arr = np.append(vgx_norm[x1:x2],[vgy_norm[x1:x2],vdx_norm[x1:x2],vdy_norm[x1:x2]])
    ymin = np.amin(arr)
    ymax = np.amax(arr)


plt.ylim(ymin,ymax)
plt.xlim(xmin,xmax)

plt.plot(z, vgx_norm, linewidth=2, label=r'$v_{gx}/c_s$')
plt.plot(z, vgy_norm, linewidth=2, label=r'$v_{gy}/c_s$')
plt.plot(z, vdx_norm, linewidth=2, label=r'$v_{dx}/c_s$',linestyle='dashed')
plt.plot(z, vdy_norm, linewidth=2, label=r'$v_{dy}/c_s$',linestyle='dashed')

plt.rc('font',size=fontsize,weight='bold')

lines1, labels1 = ax.get_legend_handles_labels()

legend=ax.legend(lines1, labels1, loc='lower left', frameon=False, ncol=1, fontsize=fontsize/2)

#legend.get_frame().set_linewidth(0.0)
#plt.title(title,weight='bold')

plt.xticks(fontsize=fontsize,weight='bold')
plt.xlabel(r'$z/H_g$',fontsize=fontsize)

plt.yticks(fontsize=fontsize,weight='bold')
plt.ylabel(r'$velocities$', fontsize=fontsize)

fname = 'eqm_velocity'
plt.savefig(fname,dpi=150)

'''
output vertical profiles of horizontal velocites
'''
output_file = h5py.File('./eqm_horiz.h5','w')
zaxis = domain.grid(0,scales=1)
vgx.set_scales(1, keep_data=True)
vgy.set_scales(1, keep_data=True)
vdx.set_scales(1, keep_data=True)
vdy.set_scales(1, keep_data=True)

output_file['z']   = zaxis
output_file['vgx'] = vgx['g']
output_file['vgy'] = vgy['g']
output_file['vdx'] = vdx['g']
output_file['vdy'] = vdy['g']
output_file.close()