################################################################################
#1 Get death probability f_j after being infected for j days - Integral over Gamma distribution
    # Follows a Gamma(shape=6.29,rate =0.26) distribution -> Julia works with shape, so 1/rate

death_dist = Distributions.Gamma(6.29, 1/0.26)
#Compute Death probabilities -> integrals from t=1 to t=28
death_probability = zeros(t_last)
death_probability[1] = Distributions.cdf(death_dist, 1.5) - Distributions.cdf(death_dist, 0.5)
for iter in 2:t_last
    death_probability[iter] = Distributions.cdf(death_dist, iter + 0.5) - Distributions.cdf(death_dist, iter - 0.5)
end
sum(death_probability) #0.71
plot(death_probability)
reverse_death_probability = reverse(death_probability)

################################################################################
"Hyperparameter to compute model implied deaths from cases in particle filter."
struct CovidDeathHyperParameter{A}
    ifr::A
    index_change_ifr::Vector{Int64}
    reverse_death_probability::Vector{Float64}
    death_lookback::Int64
    function CovidDeathHyperParameter(
        ifr::A,
        index_change_ifr::Vector{Int64},
        reverse_death_probability::Vector{Float64},
        death_lookback::Int64
        ) where {A}
        return new{A}(ifr, index_change_ifr, reverse_death_probability, death_lookback)
    end
end

################################################################################
"Compute model implied deaths, based on Flaxman et al. (2020). See 'current_rate' for ifr values."
function accumulate_deaths(cases, hyperparam::C, iter::Integer) where {C<:CovidDeathHyperParameter}
    @unpack ifr, index_change_ifr, reverse_death_probability, death_lookback = hyperparam
    lookback = lookback_t(death_lookback, iter)
    deaths = zero(eltype(ifr))
    for (counter, i) in enumerate(lookback:iter)
        ifr_current = current_rate(ifr, index_change_ifr, i)
        deaths += ifr_current * cases[i] * reverse_death_probability[counter]
    end
    return deaths
end

"Same as accumulate_deaths, but use a particle trajectory for cases, instead of only cases vector, which includes cases and deaths, so needs a further index."
function accumulate_deaths_pf(cases, hyperparam::C, iter::Integer) where {C<:CovidDeathHyperParameter}
    @unpack ifr, index_change_ifr, reverse_death_probability, death_lookback = hyperparam
    lookback = lookback_t(death_lookback, iter)
    deaths = zero(eltype(ifr))
    for (counter, i) in enumerate(lookback:iter)
        ifr_current = current_rate(ifr, index_change_ifr, i)
        deaths += ifr_current * cases[i][3] * reverse_death_probability[counter]
    end
    return deaths
end
