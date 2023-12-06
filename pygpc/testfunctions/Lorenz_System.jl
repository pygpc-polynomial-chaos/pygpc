# sometimes required to set up julia when starting for the first time:

# import Pkg
# Pkg.instantiate()

using DifferentialEquations

function lorenz!(du,u,p,t)
    σ, β, ρ = p
    du[1] = σ*(u[2]-u[1])
    du[2] = u[1]*(ρ-u[3]) - u[2]
    du[3] = u[1]*u[2] - β*u[3]
   end
   
function Julia_Lorenz(p, u0, t_vals)
   tspan = (first(t_vals), last(t_vals))
   prob = ODEProblem(lorenz!,u0,tspan, p)
   sol = solve(prob)     
   return sol(t_vals)
end