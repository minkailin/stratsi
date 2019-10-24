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
const fixedSt  = true
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
plotting parameters
=#
const fontsize=24

#=
prescriptions for stokes number (epstein) and diffusion coefficients
=#

function stokes_number(rhog)
    if fixedSt == false
        return st0*rhog0/rhog
    else
        return st0
    end
end

function visc(rhog)
    return alpha0*rhog0/rhog #this is needed because we assume the dynamical viscosity = nu*rhog = constant throughout 
end

function eta_hat(z)
    return eta_hat0
end

function diffusion_coeff(z)
    return delta0
end

function deriv_vert!(dfunc, func, params, z)
    ln_epsilon, ln_rhog, vdz = func
#    ln_epsilon, ln_rhog, chi = func
    
#    vdz     = -sqrt(chi)
    rhog    = exp(ln_rhog)
    epsilon = exp(ln_epsilon)
    diff    = diffusion_coeff(z, rhog)
    stokes  = stokes_number(rhog)

    dfunc[1] =  vdz/diff
    dfunc[2] =  epsilon*vdz/stokes - z 
    dfunc[3] = -z/vdz - 1.0/stokes
#    dfunc[3] = -2.0*z - 2.0*vdz/stokes
end

function vertical_profiles_analytic(z)
    #in the limit of constant stokes number
    eps    = epsilon0*exp(-0.5*beta*z*z/delta0)
    rhogas = rhog0*exp( (delta0/st0)*(eps - epsilon0) - 0.5*z*z)
    vdustz = -beta*z
    return (epsilon = eps, rhog = rhogas, vdz = vdustz)
end

function vertical_profiles(z; sol=0)
    if fixedSt == true
        eps, rhogas, vdustz = vertical_profiles_analytic(z)
    else
        eps   = exp(sol(z)[1])
        rhogas= exp(sol(z)[2])
        vdustz= sol(z)[3]
    end
    return (epsilon = eps, rhog = rhogas, vdz = vdustz)
end

function deriv_horiz!(dfunc, func, params, z)
    vgx, dvgx, vgy, dvgy, vdx, vdy = func 
    eps, rhog, vdz = vertical_profiles(z, sol=params)
    nu     = visc(rhog)
    stokes = stokes_number(rhog)
    eta    = eta_hat(z)

    dfunc[1] = dvgx

    dfunc[2] = -2.0*vgy - 2.0*eta + eps*(vgx - vdx)/stokes
    dfunc[2]/= nu

    dfunc[3] = dvgy
    
    dfunc[4] = 0.5*vgx + eps*(vgy - vdy)/stokes
    dfunc[4]/= nu

    dfunc[5] = 2.0*vdy - (vdx - vgx)/stokes
    dfunc[5]/= vdz

    dfunc[6] = -0.5*vdx - (vdy - vgy)/stokes
    dfunc[6]/= vdz
end

function deriv_horz_bc!(residual, func, params, z)
    
    # at z=0 we demand dust velocities match to unstratified solution and gas velocities symmetric

    dvgx_beg= func[1][2]
    dvgy_beg= func[1][4]
    vdx_beg = func[1][5]
    vdy_beg = func[1][6]
    
    residual[1] = dvgx_beg
    residual[2] = dvgy_beg
    residual[3] = vdx_beg - vdx0
    residual[4] = vdy_beg - vdy0
    
    # at z=max we demand gas velocities be that of pure gas system 
    
    vgx_end = func[end][1]
    vgy_end = func[end][3]

    residual[5] = vgx_end
    residual[6] = vgy_end + eta_hat(zmax)
end

#=
solve the basic state dgratio, rhog, and vdz
=#
zaxis = (zmin, zmax)
if fixedSt == false
    eps_init, rhog_init, vdz_init = vertical_profiles_analytic(zmin)
    init = [log(eps_init), log(rhog_init), vdz_init]
    prob  = ODEProblem(deriv_vert!,init,zaxis)
    sol   = solve(prob,radau5(),reltol=1e-8,abstol=1e-8)
else
    sol   = 0
end

zplot   = collect(range(zmin, length=nz, stop=zmax))
num     = vertical_profiles.(zplot,sol=sol)
dgratio = [x.epsilon for x in num]
eps_analytic  = [x.epsilon for x in vertical_profiles_analytic.(zplot)]

xaxis = "\$z/H_g\$"
yaxis = "dust-to-gas ratio"

fig = plt.figure(figsize=(8,4.5))
ax  = fig.add_subplot()
plt.subplots_adjust(left=0.18, right=0.95, top=0.95, bottom=0.2)

plt.plot(zplot, dgratio, linewidth=2, label="\$\\epsilon\$ (numerical)")
plt.plot(zplot, eps_analytic, linewidth=2, label="\$\\epsilon\$ (const. \$St_0\$)", linestyle="dashed")
legend(fontsize=12,frameon=0)

rc("font",size=fontsize,weight="bold")

plt.xticks(fontsize=fontsize,weight="bold")
plt.xlabel(xaxis,fontsize=fontsize,weight="bold")

plt.yticks(fontsize=fontsize,weight="bold")
plt.ylabel(yaxis, fontsize=fontsize, weight="bold")

#lines1, labels1 = get_legend_handles_labels()
#legend=ax.legend(lines1, labels1, loc='upper right', frameon=False, ncol=1, fontsize=fontsize/2)

savefig("eqm_epsilon_jl",dpi=150)

#initial condition for vgx, dvgx, vgy, dvgy, vdx, vdy
init_horiz       = [vgx0, 0.0, vgy0, 0.0, vdx0, vdy0]
prob_horiz       = ODEProblem(deriv_horiz!, init_horiz, zaxis, sol)
sol_horiz   = solve(prob_horiz,radau5(),reltol=1e-8,abstol=1e-8)

#prob_horiz = TwoPointBVProblem(deriv_horiz!, deriv_horz_bc!, init_horiz, zaxis)
#sol_horiz  = solve(prob_horiz, MIRK4(), dt=zmax/nz)

#prob_horiz = BVProblem(deriv_horiz!, deriv_horz_bc!, init_horiz, zaxis)
#sol_horiz  = solve(prob_horiz, Shooting(Vern7()))

vel_horiz = sol_horiz.(zplot)
vgx = [x[1] for x in vel_horiz]
vgy = [x[3] for x in vel_horiz]
vdx = [x[5] for x in vel_horiz]
vdy = [x[6] for x in vel_horiz]

ymin = minimum(minimum(vel_horiz))
ymax = maximum(maximum(vel_horiz))

yaxis = "horiz. velocities"

fig = plt.figure(figsize=(8,4.5))
ax  = fig.add_subplot()
plt.subplots_adjust(left=0.18, right=0.95, top=0.95, bottom=0.2)

plt.ylim(-0.1,0.1)

plt.plot(zplot, vgx, linewidth=2, label="\$v_{gx}\$")
plt.plot(zplot, vgy, linewidth=2, label="\$v_{gy}\$")
#plt.plot(zplot, vdx, linewidth=2, label="\$v_{dx}\$")
#plt.plot(zplot, vdy, linewidth=2, label="\$v_{dy}\$")

legend(fontsize=12,frameon=0)

rc("font",size=fontsize,weight="bold")


#plt.xlim(xmin,xmax)

plt.xticks(fontsize=fontsize,weight="bold")
plt.xlabel(xaxis,fontsize=fontsize,weight="bold")

plt.yticks(fontsize=fontsize,weight="bold")
plt.ylabel(yaxis, fontsize=fontsize, weight="bold")

savefig("eqm_velocity_jl",dpi=150)





#close(fig)

#yaxis = analytic_solutions.(zaxis)
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
