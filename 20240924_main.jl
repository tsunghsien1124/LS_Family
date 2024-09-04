using Parameters
using ProgressMeter
using QuantEcon: MarkovChain, stationary_distributions
using QuadGK: quadgk
using Distributions
using Roots: find_zero
using Interpolations: linear_interpolation

function adda_cooper(N::Integer, ρ::Real, σ::Real; μ::Real=0.0)
    """
    Approximation of an autoregression process with a Markov chain proposed by Adda and Cooper (2003)
    """

    σ_ϵ = σ / sqrt(1.0 - ρ^2.0)
    ϵ = σ_ϵ .* quantile.(Normal(), [i / N for i = 0:N]) .+ μ
    z = zeros(N)
    for i = 1:N
        if i != (N + 1) / 2
            z[i] = N * σ_ϵ * (pdf(Normal(), (ϵ[i] - μ) / σ_ϵ) - pdf(Normal(), (ϵ[i+1] - μ) / σ_ϵ)) + μ
        end
    end
    Π = zeros(N, N)
    if ρ == 0.0
        Π .= 1.0 / N
    else
        for i = 1:N, j = 1:N
            f(u) = exp(-(u - μ)^2.0 / (2.0 * σ_ϵ^2.0)) * (cdf(Normal(), (ϵ[j+1] - μ * (1.0 - ρ) - ρ * u) / σ) - cdf(Normal(), (ϵ[j] - μ * (1.0 - ρ) - ρ * u) / σ))
            integral, err = quadgk(u -> f(u), ϵ[i], ϵ[i+1])
            Π[i, j] = (N / sqrt(2.0 * π * σ_ϵ^2.0)) * integral
        end
    end
    return z, Π
end

function parameters_function(;
    life_span::Int64=16,                # model life span
    period::Int64=3,                    # model period
    β::Float64=0.9730,                  # discount factor
    r_f::Float64=0.0344,                # risk-free rate 
    τ::Float64=0.0093,                  # transaction cost
    γ::Float64=2.00,                    # risk aversion coefficient
    ω::Float64=0.56,                    # consumption weight
    T::Float64=1.50,                    # time endowment
    η::Float64=0.40,                    # couple equivalence scale
    ϕ::Float64=0.395,                   # wage garnishment rate
    ψ::Float64=0.011,                   # divorce probability
    e_f_ρ::Float64=0.9630,              # AR(1) of female persistent income
    e_f_σ::Float64=sqrt(0.014),         # s.d. of female persistent income 
    e_f_size::Int64=5,                  # number of female persistent income 
    e_m_ρ::Float64=0.9730,              # AR(1) of male persistent income
    e_m_σ::Float64=sqrt(0.016),         # s.d. of male persistent income 
    e_m_size::Int64=5,                  # number of male persistent income 
    a_min::Real=-5.0,                   # min of asset holding
    a_max::Real=800.0,                  # max of asset holding
    a_size_neg::Integer=501,            # number of grid of negative asset holding for VFI
    a_size_pos::Integer=101,            # number of grid of positive asset holding for VFI
    a_degree::Integer=3,                # curvature of the positive asset gridpoints
)
    """
    contruct an immutable object containg all paramters
    """

    # lifecycle profile (Gourinchas and Parker, 2002)
    h_grid = [0.774482122, 0.819574547, 0.873895492, 0.9318168, 0.986069673, 1.036889326, 1.082870993, 1.121249981, 1.148476948, 1.161069822, 1.156650443, 1.134940682, 1.09844343, 1.05261516, 1.005569967, 0.9519]
    h_size = length(h_grid)

    # female persistent income
    e_f_grid, e_f_Γ = adda_cooper(e_f_size, e_f_ρ, e_f_σ)
    e_f_G = stationary_distributions(MarkovChain(e_f_Γ, e_f_grid))[1]

    # male persistent income
    e_m_grid, e_m_Γ = adda_cooper(e_m_size, e_m_ρ, e_m_σ)
    e_m_G = stationary_distributions(MarkovChain(e_m_Γ, e_m_grid))[1]

    # labor supply
    n_grid = [0.0, 0.5, 1.0]
    n_size = length(n_grid)

    # expenditure schock
    κ_grid = zeros(life_span)
    κ_size = length(κ_grid)

    # asset holding grid for VFI
    a_grid_neg = collect(range(a_min, 0.0, length=a_size_neg))
    a_grid_pos = ((range(0.0, stop=a_size_pos - 1, length=a_size_pos) / (a_size_pos - 1)) .^ a_degree) * a_max
    a_grid = cat(a_grid_neg[1:(end-1)], a_grid_pos, dims=1)
    a_size = length(a_grid)
    a_ind_zero = a_size_neg

    # return values
    return (
        life_span=life_span,
        period=period,
        β=β,
        r_f=r_f,
        τ=τ,
        γ=γ,
        ω=ω,
        T=T,
        η=η,
        ϕ=ϕ,
        ψ=ψ,
        h_grid=h_grid,
        h_size=h_size,
        e_f_ρ=e_f_ρ,
        e_f_σ=e_f_σ,
        e_f_size=e_f_size,
        e_f_grid=e_f_grid,
        e_f_Γ=e_f_Γ,
        e_f_G=e_f_G,
        e_m_ρ=e_m_ρ,
        e_m_σ=e_m_σ,
        e_m_size=e_m_size,
        e_m_grid=e_m_grid,
        e_m_Γ=e_m_Γ,
        e_m_G=e_m_G,
        n_grid=n_grid,
        n_size=n_size,
        κ_grid=κ_grid,
        κ_size=κ_size,
        a_grid=a_grid,
        a_grid_neg=a_grid_neg,
        a_grid_pos=a_grid_pos,
        a_size=a_size,
        a_size_neg=a_size_neg,
        a_size_pos=a_size_pos,
        a_ind_zero=a_ind_zero,
        a_degree=a_degree,
    )
end

mutable struct MutableVariables
    """
    construct a type for mutable variables
    """
    # single female
    V_r_s_f::Array{Float64,4}
    V_d_s_f::Array{Float64,2}
    policy_a_s_f::Array{Float64,4}
    policy_n_s_f::Array{Float64,4}
    policy_d_s_f::Array{Float64,4}
    R_s_f::Array{Float64,3}
    q_s_f::Array{Float64,3}
    # single male
    V_r_s_m::Array{Float64,4}
    V_d_s_m::Array{Float64,2}
    policy_a_s_m::Array{Float64,4}
    policy_n_s_m::Array{Float64,4}
    policy_d_s_m::Array{Float64,4}
    R_s_m::Array{Float64,3}
    q_s_m::Array{Float64,3}
    # couple 
    V_r_c::Array{Float64,6}
    V_d_c::Array{Float64,3}
    policy_a_c::Array{Float64,6}
    policy_n_c::Array{Float64,6}
    policy_d_c::Array{Float64,6}
    R_c::Array{Float64,4}
    q_c::Array{Float64,4}
    # divorced
    V_r_div::Array{Float64,4}
    V_d_div::Array{Float64,2}
    policy_a_div::Array{Int64,4}
    policy_n_div::Array{Int64,4}
    policy_d_div::Array{Int64,4}
    R_div::Array{Float64,3}
    q_div::Array{Float64,3}
end

function utility_function(c::Float64, l::Float64, γ::Float64, ω::Float64)
    """
    compute utility of CRRA utility function with coefficient γ
    """
    if (c > 0.0) && (l > 0.0)
        return γ == 1.0 ? log(c^ω * l^(1.0 - ω)) : 1.0 / ((1.0 - γ) * (c^ω * l^(1.0 - ω))^(γ - 1.0))
    else
        return -1E-16
    end
end

log_function(threshold_e::Float64) = threshold_e > 0.0 ? log(threshold_e) : -Inf

function bounds_function(V_d_j::Float64, V_nd_j::Vector{Float64}, a_grid::Vector{Float64})
    """
    compute bounds for finding roots
    """
    
    if V_d_j > maximum(V_nd_j)
        error("V_d > V_nd for all a")
    elseif V_d_j < minimum(V_nd_j)
        error("V_d < V_nd for all a")
    else
        a_ind = findfirst(V_nd_j .> V_d_j)
        @inbounds ub = a_grid[a_ind]
        @inbounds lb = a_grid[a_ind-1]
        return lb, ub
    end
end

function threshold_function(V_d_j::Array{Float64,4}, V_nd_j::Array{Float64,5}, parameters::NamedTuple)
    """
    update default thresholds
    """

    # unpack parameters
    @unpack a_size_neg, a_grid, e_1_size, e_1_grid, e_2_size, e_2_grid, e_3_size, e_3_grid, ν_size, ν_grid = parameters

    # construct containers
    threshold_a = zeros(e_1_size, e_2_size, e_3_size, ν_size)
    threshold_e_2 = zeros(a_size_neg, e_1_size, e_3_size, ν_size)

    # loop over states
    for ν_i = 1:ν_size, e_3_i = 1:e_3_size, e_1_i = 1:e_1_size

        # println("v_i = $ν_i, e_3_i = $e_3_i, e_1_i = $e_1_i")

        # defaulting thresholds in wealth (a)
        for e_2_i = 1:e_2_size
            @inbounds @views V_nd_Non_Inf = findall(V_nd[:, e_1_i, e_2_i, e_3_i, ν_i] .!= -Inf)
            @inbounds @views a_grid_itp = a_grid[V_nd_Non_Inf]
            @inbounds @views V_nd_grid_itp = V_nd[V_nd_Non_Inf, e_1_i, e_2_i, e_3_i, ν_i]
            V_nd_itp = Akima(a_grid_itp, V_nd_grid_itp)
            @inbounds V_diff_itp(a) = V_nd_itp(a) - V_d[e_1_i, e_2_i, e_3_i, ν_i]

            if minimum(V_nd_grid_itp) > V_d[e_1_i, e_2_i, e_3_i, ν_i]
                @inbounds threshold_a[e_1_i, e_2_i, e_3_i, ν_i] = -Inf
            else
                @inbounds V_diff_lb, V_diff_ub = zero_bounds_function(V_d[e_1_i, e_2_i, e_3_i, ν_i], V_nd[:, e_1_i, e_2_i, e_3_i, ν_i], a_grid)
                @inbounds threshold_a[e_1_i, e_2_i, e_3_i, ν_i] = find_zero(a -> V_diff_itp(a), (V_diff_lb, V_diff_ub), Bisection())
            end
        end

        # defaulting thresholds in endowment (e)
        @inbounds @views thres_a_Non_Inf = findall(threshold_a[e_1_i, :, e_3_i, ν_i] .!= -Inf)
        @inbounds @views thres_a_grid_itp = -threshold_a[e_1_i, thres_a_Non_Inf, e_3_i, ν_i]
        earning_grid_itp = w * exp.(e_1_grid[e_1_i] .+ e_2_grid[thres_a_Non_Inf] .+ e_3_grid[e_3_i]) .- ν_grid[ν_i]
        threshold_earning_itp = Spline1D(thres_a_grid_itp, earning_grid_itp; k=1, bc="extrapolate")
        # threshold_earning_itp = Akima(thres_a_grid_itp, earning_grid_itp)

        Threads.@threads for a_i = 1:a_size_neg
            @inbounds earning_thres = threshold_earning_itp(-a_grid[a_i])
            e_2_thres = log_function(earning_thres / w) - e_1_grid[e_1_i] - e_3_grid[e_3_i]
            @inbounds threshold_e_2[a_i, e_1_i, e_3_i, ν_i] = e_2_thres
        end
    end

    return threshold_a, threshold_e_2
end

#==============================#
# Solve stationary equilibrium #
#==============================#
parameters = parameters_function();