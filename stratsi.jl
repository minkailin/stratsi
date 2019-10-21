#modules
using DifferentialEquations
using ODEInterfaceDiffEq
using ODEInterface
using PyPlot
#using Plots
#gr()

plt = PyPlot
matplotlib.use("Agg")

#=
problem parameters
st      = midplane stokes number 
delta   = midplane diffusion coefficient 
epsilon = midplane dust-to-gas ratio  
=#
const st0      = 1.0e-3
const eta_hat0 = 0.05
const alpha0   = 1.0e-3
const epsilon0 = 1.0
const rhog0    = 1.0 

const delta0   = alpha0*(1.0 + st0 + 4.0*st0*st0)/(1.0+st0*st0)^2
const Delta2   = st0*st0 + (1.0 + epsilon0)^2 
const beta     =(1.0/st0 - (1.0/st0)*sqrt(1.0 - 4.0*st0^2))/2.0

const zmin     = 1.0e-4
const zmax     = 2.0
const nz       = 128

const vdx0 = -2.0*eta_hat0*st0/Delta2
const vdy0 = -(1.0+epsilon0)*eta_hat0/Delta2
const vgx0 = 2.0*eta_hat0*epsilon0*st0/Delta2
const vgy0 = -(1.0 + epsilon0 + st0*st0)*eta_hat0/Delta2

#=
prescriptions for stokes number (epstein) and diffusion coefficients
=#

function stokes_number(rhog)
    return st0*rhog0/rhog
#    return st0
end

function diffusion_coeff(z, rhog)
    return delta0
end
 
function analytic_solutions(z)
    vdustz = -beta*z
    eps    = epsilon0*exp(-0.5*beta*z*z/delta0)
    rhogas = rhog0*exp( (delta0/st0)*(eps - epsilon0) - 0.5*z*z )
    return (epsilon = eps, rhog = rhogas, vdz = vdustz)
end

function visc(z)
    #return 1.0e-6
    eps, rhog, vdz = analytic_solutions(z)
    return alpha0*rhog0/rhog_analytic(z) #this is needed because we assume the dynamical viscosity = nu*rhog = constant throughout 
end

function eta_hat(z)
    return eta_hat0
end

function deriv_vert(dfunc, func, params, z)
#    ln_epsilon, ln_rhog, vdz = func
    ln_epsilon, ln_rhog, chi = func
    
    vdz     = -sqrt(chi)
    rhog    = exp(ln_rhog)
    epsilon = exp(ln_epsilon)
    diff    = diffusion_coeff(z, rhog)
    stokes  = stokes_number(rhog)

    dfunc[1] =  vdz/diff
    dfunc[2] =  epsilon*vdz/stokes - z 
#    dfunc[3] = -z/vdz - 1.0/stokes
    dfunc[3] = -2.0*z - 2.0*vdz/stokes
end



eps_init, rhog_init, vdz_init = analytic_solutions(zmin)

init = [log(eps_init), log(rhog_init), vdz_init^2]

zaxis = (zmin, zmax)


prob  = ODEProblem(deriv_vert,init,zaxis)
sol   = solve(prob,Rodas5(),reltol=1e-8,abstol=1e-8)

zplot   = collect(range(zmin, length=nz, stop=zmax))
num     = sol.(zplot)
dgratio = exp.([x[1] for x in num])

#yaxis = analytic_solutions.(zaxis)
eps_analytic  = [x.epsilon for x in analytic_solutions.(zplot)]
#plot(zplot,dgratio)
#plot!(zaxis,eps_analytic)

#=
plot(solarr[1],linewidth=3,title=string("Equilibrium density"),
     xaxis=xaxis,yaxis=yaxis,label=string("\\alpha_{0}=",α0array[1]),dpi=300,
     tickfontsize=20,legendfontsize=20,
     framestyle=:box, #legend=:topleft
     titlefontsize=20,guidefontsize=20,size=[800,450],
     fontfamily="Arial", bottom_margin=4.0mm, top_margin=2.0mm,
     ylims=(0.0,1.0),grid=:none,xminorticks=1,yminorticks=2
     )
for n = 2:length(α0array)
    plot!(solarr[n],lw=3,label=string("\\alpha_{0}=",α0array[n]),xaxis=xaxis)
end
savefig("density")
=#

fontsize=24

xaxis = "\$z/H_g\$"
yaxis = "dust-to-gas ratio"

fig = plt.figure(figsize=(8,4.5))
ax  = fig.add_subplot()
plt.subplots_adjust(left=0.18, right=0.95, top=0.95, bottom=0.2)

plt.plot(zplot, dgratio, linewidth=2, label="\$\\epsilon\$ (numerical)")
plt.plot(zplot, eps_analytic, linewidth=2, label="\$\\epsilon\$ (analytic)", linestyle="dashed")
legend(fontsize=12,frameon=0)

rc("font",size=fontsize,weight="bold")

plt.xticks(fontsize=fontsize,weight="bold")
plt.xlabel(xaxis,fontsize=fontsize,weight="bold")

plt.yticks(fontsize=fontsize,weight="bold")
plt.ylabel(yaxis, fontsize=fontsize, weight="bold")

#lines1, labels1 = get_legend_handles_labels()

#legend=ax.legend(lines1, labels1, loc='upper right', frameon=False, ncol=1, fontsize=fontsize/2)

savefig("eqm_epsilon_jl",dpi=150)
#close(fig)
