using Parameters
using ProgressMeter
using QuantEcon: MarkovChain, stationary_distributions
using QuadGK: quadgk
using Distributions
using Roots: find_zero, Bisection
using Interpolations
using LinearAlgebra
using FLOWMath: Akima
using Optim

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
            integral = quadgk(u -> f(u), ϵ[i], ϵ[i+1])[1]
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
    ϕ::Float64=0.395,                   # wage garnishment rate
    ψ::Float64=0.011,                   # divorce probability
    e_m_ρ::Float64=0.9730,              # AR(1) of male persistent income
    e_m_σ::Float64=sqrt(0.016),         # s.d. of male persistent income 
    e_m_size::Int64=5,                  # number of male persistent income 
    a_min::Float64=-20.0,               # min of asset holding
    a_max::Float64=800.0,               # max of asset holding
    a_size_neg::Int64=501,              # number of grid of negative asset holding for VFI
    a_size_pos::Int64=101,              # number of grid of positive asset holding for VFI
    a_degree::Int64=3,                  # curvature of the positive asset gridpoints
)
    """
    contruct an immutable object containg all paramters
    """
    # model-period-year parameters
    β = β^period
    r_f = (1.0 + r_f)^period - 1.0
    τ = (1.0 + τ)^period - 1.0
    e_m_σ = sqrt((e_m_ρ^4 + e_m_ρ^2 + 1.0) * e_m_σ^2)
    e_m_ρ = e_m_ρ^period

    # lifecycle profile (Gourinchas and Parker, 2002)
    h_grid = [
        0.774482122, 0.819574547, 0.873895492, 0.9318168, 0.986069673,
        1.036889326, 1.082870993, 1.121249981, 1.148476948, 1.161069822,
        1.156650443, 1.134940682, 1.09844343, 1.05261516, 1.005569967,
        0.9519
    ]
    h_size = length(h_grid)

    # male persistent income
    e_m_grid, e_m_Γ = adda_cooper(e_m_size, e_m_ρ, e_m_σ)
    e_m_grid = exp.(e_m_grid)
    e_m_G = stationary_distributions(MarkovChain(e_m_Γ, e_m_grid))[1]

    # expenditure schock
    κ_grid = zeros(3, life_span)
    κ_size = size(κ_grid)[1]
    κ_Γ = [1.0, 0.0, 0.0]

    # asset holding grid for VFI
    a_grid_neg = collect(range(a_min, 0.0, length=a_size_neg))
    a_grid_pos = ((range(0.0, stop=a_size_pos - 1, length=a_size_pos) / (a_size_pos - 1)) .^ a_degree) * a_max
    a_grid = cat(a_grid_neg[1:(end-1)], a_grid_pos, dims=1)
    a_size = length(a_grid)
    a_ind_zero = a_size_neg

    # labor supply
    n_grid = collect(0.0:0.5:1.0)
    n_size = length(n_grid)

    # normalization factor
    χ = (1.0 - β^life_span) / (1.0 - β)

    # iterators
    loop_VFI = collect(Iterators.product(1:κ_size, 1:e_m_size, 1:a_size))
    loop_thres_a = collect(Iterators.product(1:κ_size, 1:e_m_size))
    loop_thres_e = collect(Iterators.product(1:κ_size, 1:a_size_neg))

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
        ϕ=ϕ,
        ψ=ψ,
        h_grid=h_grid,
        h_size=h_size,
        e_m_ρ=e_m_ρ,
        e_m_σ=e_m_σ,
        e_m_size=e_m_size,
        e_m_grid=e_m_grid,
        e_m_Γ=e_m_Γ,
        e_m_G=e_m_G,
        κ_grid=κ_grid,
        κ_size=κ_size,
        κ_Γ=κ_Γ,
        a_grid=a_grid,
        a_grid_neg=a_grid_neg,
        a_grid_pos=a_grid_pos,
        a_size=a_size,
        a_size_neg=a_size_neg,
        a_size_pos=a_size_pos,
        a_ind_zero=a_ind_zero,
        a_degree=a_degree,
        n_grid=n_grid,
        n_size=n_size,
        χ=χ,
        loop_VFI=loop_VFI,
        loop_thres_a=loop_thres_a,
        loop_thres_e=loop_thres_e,
    )
end

mutable struct Mutable_Variables
    """
    construct a type for mutable variables
    """
    # single male
    V_s_m::Array{Float64,4} # (a, κ, e, h)
    E_V_s_m::Array{Float64,3} # (a', e, h-1)
    V_s_m_r::Array{Float64,4} # (a, κ, e, h)
    V_s_m_d::Array{Float64,2} # (e, h)
    policy_s_m_r_a::Array{Float64,4}
    policy_s_m_r_n::Array{Float64,4}
    policy_s_m_d::Array{Float64,4}
    policy_s_m_d_n::Array{Float64,2}
    thres_s_m_a::Array{Float64,3} # (κ, e, h)
    thres_s_m_e::Array{Float64,3} # (a, κ, h)
    R_s_m::Array{Float64,3} # (a', e, h ≈ j)
    q_s_m::Array{Float64,3} # (a', e, h ≈ j)
    rbl_s_m::Array{Float64,3} # (e, h ≈ j, 2)
end

function utility_function(c::Float64, l::Float64, γ::Float64, ω::Float64, χ::Float64)
    """
    compute utility of CRRA utility function with coefficient γ
    """
    if (c > 0.0) && (l > 0.0)
        return γ == 1.0 ? log(c^ω * l^(1.0 - ω)) / χ : 1.0 / (χ * (1.0 - γ) * (c^ω * l^(1.0 - ω))^(γ - 1.0))
    else
        return -Inf
    end
end

function variables_function(parameters::NamedTuple)
    """
    construct a mutable object containing endogenous variables
    """

    # unpack parameters
    @unpack a_size, a_grid, e_m_size, e_m_grid, κ_size, κ_grid, h_size, h_grid, n_size, n_grid = parameters
    @unpack r_f, ϕ, γ, ω, T, χ = parameters

    # define value and policy functions
    V_s_m = zeros(a_size, κ_size, e_m_size, h_size)
    E_V_s_m = zeros(a_size, e_m_size, h_size - 1)
    V_s_m_r = zeros(a_size, κ_size, e_m_size, h_size)
    V_s_m_d = zeros(e_m_size, h_size)
    policy_s_m_r_a = zeros(a_size, κ_size, e_m_size, h_size)
    policy_s_m_r_n = zeros(a_size, κ_size, e_m_size, h_size)
    policy_s_m_d = zeros(a_size, κ_size, e_m_size, h_size)
    policy_s_m_d_n = zeros(e_m_size, h_size)

    # solve the last period
    h = h_grid[h_size]
    for e_m_i in 1:e_m_size
        e_m = e_m_grid[e_m_i]

        # default
        n = 0.0
        l = T - n
        c = h * e_m * n * (1.0 - ϕ)
        u = utility_function(c, l, γ, ω, χ)
        V_s_m_d[e_m_i, h_size] = u
        policy_s_m_d_n[e_m_i, h_size] = n
        for n_i in 2:n_size
            n = n_grid[n_i]
            l = T - n
            c = h * e_m * n * (1.0 - ϕ)
            u = utility_function(c, l, γ, ω, χ)
            if u > V_s_m_d[e_m_i, h_size]
                V_s_m_d[e_m_i, h_size] = u
                policy_s_m_d_n[e_m_i, h_size] = n
            end
        end

        # repayment
        for κ_i in 1:κ_size, a_i in 1:a_size
            a = a_grid[a_i]
            κ = κ_grid[κ_i, h_size]
            n = 0.0
            l = T - n
            c = h * e_m * n + a - κ
            u = utility_function(c, l, γ, ω, χ)
            V_s_m_r[a_i, κ_i, e_m_i, h_size] = u
            policy_s_m_r_n[a_i, κ_i, e_m_i, h_size] = n
            for n_i in 2:n_size
                n = n_grid[n_i]
                l = T - n
                c = h * e_m * n + a - κ
                u = utility_function(c, l, γ, ω, χ)
                if u > V_s_m_r[a_i, κ_i, e_m_i, h_size]
                    V_s_m_r[a_i, κ_i, e_m_i, h_size] = u
                    policy_s_m_r_n[a_i, κ_i, e_m_i, h_size] = n
                end
            end
            # to default or not
            if V_s_m_r[a_i, κ_i, e_m_i, h_size] <= V_s_m_d[e_m_i, h_size]
                V_s_m[a_i, κ_i, e_m_i, h_size] = V_s_m_d[e_m_i, h_size]
                policy_s_m_d[a_i, κ_i, e_m_i, h_size] = 1.0
            else
                V_s_m[a_i, κ_i, e_m_i, h_size] = V_s_m_r[a_i, κ_i, e_m_i, h_size]
            end
        end
    end

    # define default thresholds
    thres_s_m_a = zeros(κ_size, e_m_size, h_size)
    thres_s_m_e = zeros(a_size, κ_size, h_size)

    # define repayment probability, pricing function, and risky borrowing limit
    R_s_m = zeros(a_size, e_m_size, h_size - 1)
    q_s_m = ones(a_size, e_m_size, h_size - 1) ./ (1.0 + r_f)
    rbl_s_m = zeros(e_m_size, h_size - 1, 2)

    # return outputs
    variables = Mutable_Variables(V_s_m, E_V_s_m, V_s_m_r, V_s_m_d, policy_s_m_r_a, policy_s_m_r_n, policy_s_m_d, policy_s_m_d_n, thres_s_m_a, thres_s_m_e, R_s_m, q_s_m, rbl_s_m)
    return variables
end

log_function(threshold_e::Float64) = threshold_e > 0.0 ? log(threshold_e) : -Inf

function threshold_function!(
    h_i::Int64,
    variables::Mutable_Variables,
    parameters::NamedTuple
)
    """
    update default thresholds
    """
    # unpack parameters
    @unpack a_size, a_grid, e_m_size, e_m_grid, κ_size, κ_grid = parameters

    # defaulting thresholds in wealth
    for e_m_i in 1:e_m_size, κ_i in 1:κ_size
        @inbounds @views V_r_h = variables.V_s_m_r[:, κ_i, e_m_i, h_i]
        @inbounds @views V_d_h = variables.V_s_m_d[e_m_i, h_i]
        if V_d_h > maximum(V_r_h)
            error("V_d > V_nd for all a")
            variables.thres_s_m_a[κ_i, e_m_i, h_i] = Inf
        elseif V_d_h < minimum(V_r_h)
            error("V_d < V_nd for all a")
            variables.thres_s_m_a[κ_i, e_m_i, h_i] = -Inf
        else
            @inbounds @views V_r_h_no_Inf = findall(V_r_h .!= -Inf)
            @inbounds @views a_grid_itp = a_grid[V_r_h_no_Inf]
            @inbounds @views V_r_h_grid_itp = V_r_h[V_r_h_no_Inf]
            V_r_h_itp = linear_interpolation(a_grid_itp, V_r_h_grid_itp, extrapolation_bc=Interpolations.Line())
            @inbounds V_h_diff_itp(a) = V_r_h_itp(a) - V_d_h
            a_ind = findfirst(V_r_h .> V_d_h)
            @inbounds V_h_diff_lb = a_grid[a_ind-1]
            @inbounds V_h_diff_ub = a_grid[a_ind]
            @inbounds variables.thres_s_m_a[κ_i, e_m_i, h_i] = find_zero(a -> V_h_diff_itp(a), (V_h_diff_lb, V_h_diff_ub), Bisection())
        end
    end

    # defaulting thresholds in earnings
    for κ_i in 1:κ_size, a_i in 1:a_size
        a = a_grid[a_i]
        @inbounds @views thres_a_no_Inf = findall(variables.thres_s_m_a[κ_i, :, h_i] .!= -Inf)
        @inbounds @views thres_a_grid_itp = -variables.thres_s_m_a[κ_i, thres_a_no_Inf, h_i]
        @inbounds @views e_m_grid_itp = e_m_grid[thres_a_no_Inf]
        e_m_itp = linear_interpolation(thres_a_grid_itp, e_m_grid_itp, extrapolation_bc=Interpolations.Line())
        @inbounds variables.thres_s_m_e[a_i, κ_i, h_i] = e_m_itp(-a)
    end
    variables.thres_s_m_e[:, :, h_i] .= log_function.(variables.thres_s_m_e[:, :, h_i])

    # return results
    return nothing
end

function pricing_and_rbl_function!(
    h_i::Int64,
    variables::Mutable_Variables,
    parameters::NamedTuple)
    """
    update pricing function and borrowing risky limit
    """

    # unpack parameters
    @unpack e_m_size, e_m_ρ, e_m_σ, e_m_grid, a_grid, a_size_neg, a_grid_neg, a_ind_zero, h_grid, κ_size, κ_Γ, r_f, τ, ϕ = parameters

    # loop over states
    for e_m_i in 1:e_m_size

        # risk-based borrowing price
        @inbounds e_m_μ = e_m_ρ * e_m_grid[e_m_i]
        for a_p_i in 1:(a_size_neg-1)
            @inbounds variables.R_s_m[a_p_i, e_m_i, h_i] = 0.0
            @inbounds a_p = a_grid[a_p_i]
            for κ_p_i in 1:κ_size
                @inbounds e_m_p_thres = variables.thres_s_m_e[a_p_i, κ_p_i, h_i+1]
                repayment_prob = 1.0 - cdf(Normal(e_m_μ, e_m_σ), e_m_p_thres)
                if repayment_prob ≈ 1.0
                    @inbounds variables.R_s_m[a_p_i, e_m_i, h_i] += κ_Γ[κ_p_i] * (-a_p)
                else
                    @inbounds @views garnishment_itp = linear_interpolation(e_m_grid, ϕ * h_grid[h_i+1] .* e_m_grid .* variables.policy_s_m_d_n[:, h_i+1], extrapolation_bc=Interpolations.Line())
                    expected_garnishment(x) = garnishment_itp(exp(x)) * pdf(Normal(e_m_μ, e_m_σ), x)
                    @inbounds variables.R_s_m[a_p_i, e_m_i, h_i] += κ_Γ[κ_p_i] * quadgk(x -> expected_garnishment(x), -Inf, e_m_p_thres)[1]
                    @inbounds variables.R_s_m[a_p_i, e_m_i, h_i] += κ_Γ[κ_p_i] * repayment_prob * (-a_p)
                end
            end
            @inbounds variables.q_s_m[a_p_i, e_m_i, h_i] = variables.R_s_m[a_p_i, e_m_i, h_i] / ((-a_p) * (1.0 + r_f + τ))
        end

        # risky borrowing limit and maximum discounted borrwoing amount
        @inbounds @views qa = variables.q_s_m[1:a_ind_zero, e_m_i, h_i] .* a_grid_neg
        qa_min_ind = argmin(qa)
        if qa_min_ind == 1
            qa_lb = 1
            qa_ub = 2
        elseif qa_min_ind == a_ind_zero
            qa_lb = a_ind_zero - 1
            qa_ub = a_ind_zero
        else
            qa_lb = qa_min_ind - 1
            qa_ub = qa_min_ind + 1
        end
        qa_itp = Akima(a_grid_neg, qa)
        res_rbl = optimize(qa_itp, a_grid[qa_lb], a_grid[qa_ub])
        @inbounds variables.rbl_s_m[e_m_i, h_i, 1] = Optim.minimizer(res_rbl)
        @inbounds variables.rbl_s_m[e_m_i, h_i, 2] = Optim.minimum(res_rbl)
    end

    # return results
    return nothing
end

function E_V_function!(
    h_i::Int64,
    variables::Mutable_Variables,
    parameters::NamedTuple
)
    """
    construct expected value functions
    """

    # unpack parameters
    @unpack e_m_size, e_m_Γ, κ_size, κ_Γ, a_size, β = parameters

    # update expected value 
    for a_p_i in 1:a_size, e_m_i in 1:e_m_size
        @inbounds variables.E_V_s_m[a_p_i, e_m_i, h_i] = 0.0
        for κ_p_i in 1:κ_size, e_m_p_i in 1:e_m_size
            @inbounds variables.E_V_s_m[a_p_i, e_m_i, h_i] += β * e_m_Γ[e_m_i, e_m_p_i] * κ_Γ[κ_p_i] * variables.V_s_m[a_p_i, κ_p_i, e_m_p_i, h_i+1]
        end
    end

    # replace NaN with -Inf
    replace!(variables.E_V_s_m, NaN => -Inf)

    # return results
    return nothing
end

function value_and_policy_function!(
    h_i::Int64,
    variables::Mutable_Variables,
    parameters::NamedTuple
)
    """
    one-step update of value and policy functions at age h_i
    """

    # unpack parameters
    @unpack h_grid, a_size, a_grid, e_m_size, e_m_grid, κ_size, κ_grid, n_size, n_grid, T, γ, ω, ϕ, χ = parameters

    # loop over all states
    h = h_grid[h_i]
    for e_m_i in 1:e_m_size
        e_m = e_m_grid[e_m_i]

        # extract risky borrowing limit and maximum discounted borrowing amount
        @inbounds @views rbl_a = variables.rbl_s_m[e_m_i, h_i, 1]

        # construct interpolated functions
        @inbounds @views qa = variables.q_s_m[:, e_m_i, h_i] .* a_grid
        qa_itp = Akima(a_grid, qa)
        @inbounds @views EV = variables.E_V_s_m[:, e_m_i, h_i]
        EV_itp = Akima(a_grid, EV)

        # construct objective functions
        obj_r(a_p, CoH, l) = -(utility_function(CoH - qa_itp(a_p), l, γ, ω, χ) + EV_itp(a_p))

        # default
        n = 0.0
        l = T - n
        c = h * e_m * n * (1.0 - ϕ)
        u = utility_function(c, l, γ, ω, χ) + EV_itp(0.0)
        variables.V_s_m_d[e_m_i, h_i] = u
        variables.policy_s_m_d_n[e_m_i, h_i] = n
        for n_i in 2:n_size
            n = n_grid[n_i]
            l = T - n
            c = h * e_m * n * (1.0 - ϕ)
            u = utility_function(c, l, γ, ω, χ) + EV_itp(0.0)
            if u > variables.V_s_m_d[e_m_i, h_i]
                variables.V_s_m_d[e_m_i, h_i] = u
                variables.policy_s_m_d_n[e_m_i, h_i] = n
            end
        end

        # repayment
        for κ_i in 1:κ_size, a_i in 1:a_size
            a = a_grid[a_i]
            κ = κ_grid[κ_i, h_i]
            n = 0.0
            l = T - n
            CoH = h * e_m * n + a - κ
            if CoH < rbl_a
                variables.V_s_m_r[a_i, κ_i, e_m_i, h_i] = -Inf
                variables.policy_s_m_r_a[a_i, κ_i, e_m_i, h_i] = rbl_a
                variables.policy_s_m_r_n[a_i, κ_i, e_m_i, h_i] = n
            else
                res_r = optimize(a_p -> obj_r(a_p, CoH, l), rbl_a, CoH)
                variables.V_s_m_r[a_i, κ_i, e_m_i, h_i] = -Optim.minimum(res_r)
                variables.policy_s_m_r_a[a_i, κ_i, e_m_i, h_i] = Optim.minimizer(res_r)
                variables.policy_s_m_r_n[a_i, κ_i, e_m_i, h_i] = n
            end
            for n_i in 2:n_size
                n = n_grid[n_i]
                l = T - n
                CoH = h * e_m * n + a - κ
                if CoH < rbl_a
                    variables.V_s_m_r[a_i, κ_i, e_m_i, h_i] = -Inf
                    variables.policy_s_m_r_a[a_i, κ_i, e_m_i, h_i] = rbl_a
                    variables.policy_s_m_r_n[a_i, κ_i, e_m_i, h_i] = n
                else
                    res_r = optimize(a_p -> obj_r(a_p, CoH, l), rbl_a, CoH)
                    if -Optim.minimum(res_r) > variables.V_s_m_r[a_i, κ_i, e_m_i, h_i]
                        variables.V_s_m_r[a_i, κ_i, e_m_i, h_i] = -Optim.minimum(res_r)
                        variables.policy_s_m_r_a[a_i, κ_i, e_m_i, h_i] = Optim.minimizer(res_r)
                        variables.policy_s_m_r_n[a_i, κ_i, e_m_i, h_i] = n
                    end
                end
            end

            # to default or not
            if variables.V_s_m_r[a_i, κ_i, e_m_i, h_i] <= variables.V_s_m_d[e_m_i, h_i]
                variables.V_s_m[a_i, κ_i, e_m_i, h_i] = variables.V_s_m_d[e_m_i, h_i]
                variables.policy_s_m_d[a_i, κ_i, e_m_i, h_i] = 1.0
            else
                variables.V_s_m[a_i, κ_i, e_m_i, h_i] = variables.V_s_m_r[a_i, κ_i, e_m_i, h_i]
            end
        end
    end

    # return results
    return nothing
end

#==============================#
# Solve stationary equilibrium #
#==============================#
parameters = parameters_function();
variables = variables_function(parameters);

# testing functions
threshold_function!(parameters.h_size, variables, parameters)
pricing_and_rbl_function!(parameters.h_size - 1, variables, parameters)
E_V_function!(parameters.h_size - 1, variables, parameters)
value_and_policy_function!(parameters.h_size - 1, variables, parameters)

threshold_function!(parameters.h_size - 1, variables, parameters)
pricing_and_rbl_function!(parameters.h_size - 2, variables, parameters)
E_V_function!(parameters.h_size - 2, variables, parameters)
value_and_policy_function!(parameters.h_size - 2, variables, parameters)

threshold_function!(parameters.h_size - 2, variables, parameters)
pricing_and_rbl_function!(parameters.h_size - 3, variables, parameters)
E_V_function!(parameters.h_size - 3, variables, parameters)
value_and_policy_function!(parameters.h_size - 3, variables, parameters)

threshold_function!(parameters.h_size - 3, variables, parameters)
pricing_and_rbl_function!(parameters.h_size - 4, variables, parameters)
E_V_function!(parameters.h_size - 4, variables, parameters)
value_and_policy_function!(parameters.h_size - 4, variables, parameters)

threshold_function!(parameters.h_size - 4, variables, parameters)
pricing_and_rbl_function!(parameters.h_size - 5, variables, parameters)
E_V_function!(parameters.h_size - 5, variables, parameters)
value_and_policy_function!(parameters.h_size - 5, variables, parameters)

threshold_function!(parameters.h_size - 5, variables, parameters)
pricing_and_rbl_function!(parameters.h_size - 6, variables, parameters)
E_V_function!(parameters.h_size - 6, variables, parameters)
value_and_policy_function!(parameters.h_size - 6, variables, parameters)

# cheching figures
using Plots
plot(parameters.a_grid_neg, variables.q_s_m[1:parameters.a_ind_zero, :, end])
plot(parameters.a_grid_neg, variables.q_s_m[1:parameters.a_ind_zero, :, end] .* parameters.a_grid_neg)
plot!(variables.rbl_s_m[:, end, 1], variables.rbl_s_m[:, end, 2], seriestype=:scatter)
# plot(parameters.a_grid, variables.q_s_m[:, :, end] .* parameters.a_grid)

plot(parameters.a_grid_neg, variables.q_s_m[1:parameters.a_ind_zero, 2, end-5:end])

plot(parameters.a_grid_neg, variables.q_s_m[1:parameters.a_ind_zero, :, end-2])
