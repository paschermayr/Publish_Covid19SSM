################################################################################
#Solver if Beta depends on a states Vector
function seeiir_states_ode!(du,u,p,t)
    (S,E₁,E₂,I₁,I₂,R,C) = u
    (states,vaccinations, β,γ,ϵ,ρ, gamma_change) = p
    N = S+E₁+E₂+I₁+I₂+R

    βₜ = obtain_β(β, states, t)
    vaccinationsₜ = vaccinations[minimum_t(t)]
    γₜ = gamma_interval(gamma_change, t, γ)

    infection = βₜ*S*(I₁+I₂)/N
    exposed₁ = ϵ*E₁
    exposed₂ = ϵ*E₂
    recovery₁ = γₜ*I₁
    recovery₂ = γₜ*I₂
    vaccine = ρ*vaccinationsₜ

    @inbounds begin
        du[1] = - infection - vaccine       # S
        du[2] = infection - exposed₁        # E₁
        du[3] = exposed₁ - exposed₂         # E₂
        du[4] = exposed₂ - recovery₁        # I₁
        du[5] = recovery₁ - recovery₂       # I₂
        du[6] = recovery₂ + vaccine         # R
        du[7] = infection                   # Cumulative Cases
    end
    nothing
end

################################################################################
# ODE Solver
# Solver if Beta is known already
function seeiir_ode!(du,u,p,t)
    (S,E₁,E₂,I₁,I₂,R,C) = u
    (β,vaccinations,γ,ϵ,ρ) = p
    N = S+E₁+E₂+I₁+I₂+R

    infection = β*S*(I₁+I₂)/N
    exposed₁ = ϵ*E₁
    exposed₂ = ϵ*E₂
    recovery₁ = γ*I₁
    recovery₂ = γ*I₂
    vaccine = ρ*vaccinations

    @inbounds begin
        du[1] = - infection - vaccine       # S
        du[2] = infection - exposed₁        # E₁
        du[3] = exposed₁ - exposed₂         # E₂
        du[4] = exposed₂ - recovery₁        # I₁
        du[5] = recovery₁ - recovery₂       # I₂
        du[6] = recovery₂ + vaccine         # R
        du[7] = infection                   # Cumulative Cases
    end
    nothing
end

################################################################################
################################################################################
################################################################################
function get_incremental_ODE_SEEIIR(θ)
    @unpack state₀, latent, vaccinations, β, γ, ϵ, ρ, gamma_change, N = θ
    # Initial conditions
    T = eltype(β)
    u0 = initial_state_to_population_seeiir(T, state₀, 300., N)
    # Initiate buffer for cases and deaths
    sol_C = zeros(T, length(latent))
    # Assign length of death probabilities and hyperparameter
    tspan = get_tspan(1)
    # Solve ODE
    βₜ = get_β(β, obtain_s(latent, 1))
    vaccinationsₜ = vaccinations[1]
    γₜ = gamma_interval(gamma_change, 1, γ)
    p = [βₜ, vaccinationsₜ, γₜ, ϵ, ρ]

    problem = ODEProblem(seeiir_ode!,
          u0,
          tspan,
          p)
    sol = solve(problem, Tsit5(), saveat = tspan[end])
    # Compute Model implied cases and return model distribution
    sol_C[1] = cases_from_sol(sol)
    for iter in 2:length(latent)
        u0 = obtain_state₀(latent, iter-1)
        tspan = get_tspan(iter)
        βₜ = get_β(β, obtain_s(latent, iter))
        vaccinationsₜ = vaccinations[iter]
        γₜ = gamma_interval(gamma_change, iter, γ)

        p[1] = βₜ
        p[2] = vaccinationsₜ
        p[3] = γₜ
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
function get_ODE_SEEIIR(θ, tspan, solver, dt)
    @unpack state₀, latent, vaccinations, β, γ, ϵ, ρ, gamma_change, N = θ
    # Initial conditions
    u0 = initial_state_to_population_seeiir(eltype(β), state₀, 300., N)
    p=(latent,vaccinations,β,γ,ϵ,ρ,gamma_change)
    # Solve ODE
    prob = ODEProblem(seeiir_states_ode!,
          u0,
          tspan,
          p
         )
    sol = solve(prob,
              solver, #Euler(), #Tsit5(),
              dt = dt, #0.01,
              saveat = 1.0
        )
# Compute deaths from cases
    sol_C = all_cases_from_sol(sol)
    return sol_C, sol
end
