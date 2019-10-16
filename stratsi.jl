#modules
using DifferentialEquations
using ODEInterfaceDiffEq
using ODEInterface
using Plots
gr()
#=
problem parameters
st      = midplane stokes number 
delta   = midplane diffusion coefficient 
epsilon = midplane dust-to-gas ratio  
=#
const st0      = 1.0e-3
const delta0   = 1.0e-5
const epsilon0 = 0.5
const rhog0    = 1.0 

const zmax     = 2.0

#=
prescriptions for stokes number (epstein) and diffusion coefficients
=#

function stokes_number(rhog)
    return st0*rhog0/rhog
#    return st0
#function stokes_number(z)
#return st0*exp(0.5*z*z)
end

function diffusion_coeff(z, rhog)
    return delta0
end

function epsilon_analytic(z)
    beta = 1.0/st0 - sqrt(1.0/st0^2 - 4.0)
    beta/= 2.0
    return epsilon0*exp(-0.5*beta*z*z/delta0)
end

function analytic_solutions(z)
    beta = 1.0/st0 - sqrt(1.0/st0^2 - 4.0)
    beta/= 2.0

    vdustz = -beta*z
 
    eps    = epsilon0*exp(-0.5*beta*z*z/delta0)
    rhogas = rhog0*exp( (delta0/st0)*(eps - epsilon0) - 0.5*z*z )
    rhodust= eps*rhogas

    return (rhod = rhodust, rhog = rhogas, vdz = vdustz)
end
    


function deriv(dfunc, func, params, z)

#    rhod, rhog, chi = func
    
    epsilon, rhog, vdz = func
    
    diff    = diffusion_coeff(z, rhog)
    stokes  = stokes_number(rhog)

    #=
    dfunc[1] = -sqrt(chi)*(1.0/diff + epsilon/stokes) - z
    dfunc[1]*= rhod

    dfunc[2] = -sqrt(chi)*epsilon/stokes - z
    dfunc[2]*= rhog

    dfunc[3] = -2.0*z + 2.0*sqrt(chi)/stokes
    =#

    #=
    dfunc[1] = vdz*(1.0/diff + epsilon/stokes) - z
    dfunc[1]*= rhod

    dfunc[2] = vdz*epsilon/stokes - z
    dfunc[2]*= rhog

    dfunc[3] = -z/vdz  - 1.0/stokes
    =#

    #dfunc[1] = -2.0*z + 2.0*sqrt(chi)/stokes

    #vdz = -st0*z

   # vdz = -sqrt(chi)
    
    dfunc[1] = vdz/diff
    dfunc[1]*= epsilon 
    dfunc[2] = epsilon*vdz/stokes - z
    dfunc[2]*= rhog

    #dfunc[3] = -2.0*z - 2.0*vdz/stokes
    dfunc[3] = -z/vdz  - 1.0/stokes

end

#=
function dg(t, rhod, rhog)
    return (t, rhod/rhog)
end

function vz_from_chi(t,chi)
    return (t, -sqrt(chi))
end
=#

zmin = 1.0e-6

rhod_init, rhog_init, vdz_init = analytic_solutions(zmin)

init = [rhod_init/rhog_init, rhog_init, vdz_init]

zaxis = (zmin, zmax)
prob  = ODEProblem(deriv,init,zaxis)
sol   = solve(prob,radau(),reltol=1e-12,abstol=1e-12)

zaxis = sol.t
dgratio = [x[1] for x in sol.u]
#gas  = [x[2] for x in sol.u]
#chi  = [x[3] for x in sol.u]
#vdz  = -sqrt.(chi)
#vdz  = [x[3] for x in sol.u].*1e3

#plot(zaxis,dgratio)
#plot!(time,gas)
#plot!(time,gas)
#plot!(time,vdz)



#zaxis = collect(range(zmin, length=length(sol.u), stop=zmax))
yaxis = analytic_solutions.(zaxis)
rhod  = [x.rhod for x in yaxis]
rhog  = [x.rhog for x in yaxis]
plot(zaxis,(rhod./rhog.-dgratio)./dgratio)

