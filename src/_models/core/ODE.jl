################################################################################
#Solver if Beta depends on a states Vector
function seir_states_ode!(du,u,p,t)
    (S,E,I,R,C) = u
    (states,β,γ,ϵ) = p

    N = S+E+I+R
    βₜ = obtain_β(β, states, t)
    infection = βₜ*(I*S)/N
    exposed = ϵ*E
    recovery = γ*I
    @inbounds begin
        du[1] = -infection
        du[2] = infection - exposed
        du[3] = exposed - recovery
        du[4] = recovery
        du[5] = infection
    end
    nothing
end

################################################################################
# ODE Solver
# Solver if Beta is known already
function seir_ode!(du,u,p,t)
    (S,E,I,R,C) = u
    (β,γ,ϵ) = p

    N = S+E+I+R
    infection = β*(I*S)/N
    exposed = ϵ*E
    recovery = γ*I
    @inbounds begin
        du[1] = -infection              # S
        du[2] = infection - exposed     # E
        du[3] = exposed - recovery      # I
        du[4] = recovery                # R
        du[5] = infection               # Cumulative Cases
    end
    nothing
end

################################################################################
################################################################################
################################################################################
function get_incremental_ODE(θ)
    @unpack state₀, latent, β, γ, ϵ, N = θ
    # Initial conditions
    T = eltype(β)
    u0 = initial_state_to_population(T, state₀, N)
    # Initiate buffer for cases and deaths
    sol_C = zeros(T, length(latent))
    # Assign length of death probabilities and hyperparameter
    tspan = get_tspan(1)
    # Solve ODE
    βₜ = get_β(β, obtain_s(latent, 1))
    p = [βₜ, γ, ϵ]
    problem = ODEProblem(seir_ode!,
          u0,
          (0.0, 1.0), #tspan,
          p)
    sol = solve(
        problem,
        #Euler(), dt = 0.01,
        Tsit5(),
        saveat = tspan[end])
    # Compute Model implied cases and return model distribution
    sol_C[1] = cases_from_sol(sol)
    for iter in 2:length(latent)
        u0 = obtain_state₀(latent, iter-1)
        tspan = get_tspan(iter)
        βₜ = get_β(β, obtain_s(latent, iter))
        p[1] = βₜ
        # Solve ODE
        problem = remake(problem;
            u0 = u0,
            tspan =tspan,
            p = p
        )
        sol = solve(problem,
            Tsit5(),
            saveat = float(iter)
        )
        # Compute Model implied cases and return model distribution
        sol_C[iter] = cases_from_sol(sol)
    end
    return sol_C, sol
end

#Marginal ODE
function get_ODE(θ, tspan, solver, dt)
    @unpack state₀, latent, β, γ, ϵ, N = θ
    # Initial conditions
    u0 = initial_state_to_population(eltype(β), state₀, N)
    p=(latent,β,γ,ϵ)
    # Solve ODE
    prob = ODEProblem(seir_states_ode!,
          u0,
          tspan,
          p)
    sol = solve(prob,
              solver, #Euler(), #Tsit5(),
              dt = dt, #0.01,
              saveat = 1.0)
# Compute deaths from cases
    #sol_C = [sol.u[iter][end] - sol.u[iter-1][end] for iter in 2:length(sol.u)]
    sol_C = all_cases_from_sol(sol)
    return sol_C, sol
end

################################################################################
# Standard SEIR ODE without latent state trajectory
function get_batch_ODE(θ, tspan, solver, dt)
    @unpack state₀, β, γ, ϵ, N = θ
    # Initial conditions
    u0 = initial_state_to_population(typeof(β), state₀, N)
    p=(β,γ,ϵ)
    # Solve ODE
    prob = ODEProblem(seir_ode!,
          u0,
          tspan,
          p)
    sol = solve(prob,
              solver, #Euler(), #Tsit5(),
              dt = dt, #0.01,
              saveat = 1.0)
# Compute deaths from cases
    sol_C = all_cases_from_sol(sol)
    return sol_C, sol
end

################################################################################
# Compute deaths from cases
function deaths_from_cases(sol_C, hyperparam::C) where {C<:CovidDeathHyperParameter}
    @unpack ifr, index_change_ifr, reverse_death_probability, death_lookback = hyperparam
    hyperparam = CovidDeathHyperParameter(ifr, index_change_ifr, reverse_death_probability, length(reverse_death_probability))
    sol_D = zeros(eltype(sol_C), size(sol_C))
    sol_D[1] = 0.0
    for iter in 2:length(sol_C)
        #NOTE: iter-1 because deaths at t determined from cases up to t-1
        sol_D[iter] = accumulate_deaths(sol_C, hyperparam, iter-1)
    end
    return sol_D
end
