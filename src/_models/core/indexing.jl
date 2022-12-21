################################################################################
#2 A bunch of utility functions to index the latent state trajectories to put indexing in a function (and make the compiler work it out)


"Return gamma after or before change"
function gamma_interval(T, t, γ)
    return t < T ? γ[begin] : γ[end]
end

"Get lookback value for t for death distributions. If t < maxlookback, return 1."
function lookback_t(maxlookback, t)
    return t < maxlookback ? 1 : t-(maxlookback-1)
end

"Function to search intervals of vec for x. If x smaller than first vec value, 1 is returned. If larger, length(vec) returned. Used to return correct ifr value between dates/indices used for Covid Waves."
function find_interval(vec, x)
    i = searchsortedlast(vec, x)
    return i + 1
 end

"Return relevant ifr value given current iteration 'iter' and index changes 'index_change_ifr'. See 'find_interval'."
function current_rate(ifr, index_change_ifr, iter)
    idx = find_interval(index_change_ifr, iter)
    return ifr[idx]
end

"Obtain latent state from particle trajectory for cases, which includes cases and deaths. Returns only state, not duration."
function obtain_s(states, t)
    sₜ = states[t][1][1]
    return sₜ
end
"Obtain current ODE state from particle trajectory for cases, which includes cases and deaths."
function obtain_state₀(states, t)
    return states[t][2]
end

"Utility function to return a single scalar from vector 'βᵥ' on index 'state'."
function get_β(βᵥ, state)
    return βᵥ[state]
end

"Compute initial number of SEIR state, given population 'N' and probabilities 'state₀'. 'val' is used to infer the type for the return objective to be AD compatible."
function initial_state_to_population(T::Type{R}, state₀, N) where {R}
    # Initial conditions
    u0= zeros(T, 5)
    u0[1] = state₀[1]*N     #S
    u0[2] = state₀[2]*N     #E
    u0[3] = state₀[3]*N     #I
    u0[4] = state₀[4]*N     #R
    u0[5] = state₀[3]*N + state₀[4]*N  #Cumulative cases
    return u0
end

function initial_state_to_population_seeiir(T::Type{R}, state₀, cases₀, N) where {R}
    # Initial conditions
    u0= zeros(T, 7)
    for iter in firstindex(state₀):lastindex(state₀)-1
        u0[iter] = state₀[iter]*N
    end
    u0[end] = cases₀ #u0[4] + u0[5] + u0[6] # I + R
    return u0
end

"Assign a tspan that is compatible with DiffEquations.jl. Needs to be a float type."
function get_tspan(iter)
    return (iter - 1.0, iter + 0.0)
end

################################################################################
# Assign ODE if full latent vector already present
"Function to obtain last available discrete time index if ODE solution is interpolated between time steps."
function minimum_t(t)
    if t < 1
        tₘᵢₙ = 1
    else
    #!NOTE: This should be ceil(t), as we predict state first, then choose beta
       tₘᵢₙ = Int(ceil(t))
    end
    return tₘᵢₙ
end

"Obtain Beta based on states vector, where state element is HSMM, ODEstate, and current cases."
function obtain_β(β, states, t)
    βₜ = β[states[minimum_t(t)][1][1]]
    return βₜ
end
