############################################################################################
# Likelihood

"Compute the most recent cases from an ODE solution. Cumulative cases are the last index in the solution."
function cases_from_sol(sol)
    cases = sol.u[end][end] - sol.u[end - 1][end]
    return cases > 0.0 ? cases : 1.0e-10
end

function all_cases_from_sol(sol)
    cases = zeros(eltype(sol.u[begin]), length(sol.u)-1)
    for iter in 2:length(sol.u)
        casesₜ = sol.u[iter][end] - sol.u[iter-1][end]
        cases[iter-1] = casesₜ > 0.0 ? casesₜ : 1.0e-10
    end
    return cases
end

############################################################################################
# For Deaths
function cumulative_ll_poisson_deaths(sol_X, data, counter::Integer)
    ll = 0.0
    #!NOTE: start from 2, as deaths lag cases and first death case is noisy.
    for iter in counter:(length(sol_X)) #29:(length(data))
        #NOTE: parametrization has sol_X as mean and ϕ overdispersion
        ll += logpdf(Poisson(sol_X[iter]), data[iter])
    end
    return ll
end

function cumulative_ll_nb_deaths(sol_X, ϕ, data, counter::Integer)
    ll = 0.0
    #!NOTE: start from 2, as deaths lag cases and first death case is noisy.
    for iter in counter:(length(sol_X))
        #NOTE: parametrization has sol_X as mean and ϕ overdispersion
        ll += logpdf(NegativeBinomial(ϕ, ϕ / (ϕ + sol_X[iter])), data[iter])
    end
    return ll
end

#!NOTES: Adds a scalar at ecah index to adjust real data for underreporting
function cumulative_ll_poisson_cases(sol_X, data, underreporting::Vector{T}, counter::Integer) where {T}
    ll = 0.0
    #!NOTE: start from 2, as deaths lag cases and first death case is noisy.
    for iter in counter:(length(sol_X)) #29:(length(data))
        #NOTE: parametrization has sol_X as mean and ϕ overdispersion
        ll += logpdf(Poisson(underreporting[iter] * sol_X[iter]), data[iter])
    end
    return ll
end

function cumulative_ll_nb_cases(sol_X, ϕ, data, underreporting::Vector{T}, counter::Integer) where {T}
    ll = 0.0
    #!NOTE: start from 2, as deaths lag cases and first death case is noisy.
    for iter in counter:(length(sol_X))
        #NOTE: parametrization has sol_X as mean and ϕ overdispersion
        ll += logpdf(NegativeBinomial(ϕ, ϕ / (ϕ + (underreporting[iter] * sol_X[iter]))), data[iter])
    end
    return ll
end

struct Sol_DeathsCases{A,B,C<:Real,D<:Real}
    sol_D::A
    sol_C::B
    ϕ_D::C
    ϕ_C::D
end
function cumulative_ll_nb_deathscases(sol::Sol_DeathsCases, data::D, underreporting::Vector{T}, counter::Integer) where {D<:AbstractArray{<:NamedTuple}, T}
    @unpack sol_D, sol_C, ϕ_D, ϕ_C = sol
    @argcheck length(sol_D) == length(sol_C)
    ll = 0.0
    #!NOTE: start from 2, as deaths lag cases and first death case is noisy.
    for iter in counter:(length(sol_D))
        #NOTE: parametrization has sol_X as mean and ϕ overdispersion
        ll += logpdf(NegativeBinomial(ϕ_D, ϕ_D / (ϕ_D + (sol_D[iter]))), data[iter].deaths)
        ll += logpdf(NegativeBinomial(ϕ_C, ϕ_C / (ϕ_C + (underreporting[iter] * sol_C[iter]))), data[iter].cases)
    end
    return ll
end

############################################################################################
# For latent states
function add_ll_states(dynamicsˢ, dynamicsᵈ, latent, counter::Integer)
    ll_states = 0.0
    for iter in counter:length(latent)
        if obtain_s(latent, iter-1) != obtain_s(latent, iter) #s[iter-1] != s[iter]
            ll_states += logpdf(dynamicsˢ[obtain_s(latent, iter-1)], obtain_s(latent, iter))
            ll_states += logpdf(dynamicsᵈ[obtain_s(latent, iter)], latent[iter][1][2])
        end
    end
    return ll_states
end

################################################################################
#Include
include("hsmm.jl")
include("indexing.jl")
include("deaths.jl")
include("ODE.jl")
include("ODE-seeiir.jl")
include("dynamics-seir.jl")
include("dynamics-seeiir.jl")
include("hsmm.jl")
include("hsmm.jl")

################################################################################
#export
