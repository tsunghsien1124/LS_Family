using Parameters
using ProgressMeter
using QuantEcon: MarkovChain, stationary_distributions
using QuadGK
using Distributions
using LinearAlgebra

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
    a_min::Float64=-2.5,                # min of asset holding
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
    κ_grid = ones(3, life_span) .* 0.01
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
    # loop_VFI = collect(Iterators.product(1:κ_size, 1:e_m_size, 1:a_size))
    # loop_thres_a = collect(Iterators.product(1:κ_size, 1:e_m_size))
    # loop_thres_e = collect(Iterators.product(1:κ_size, 1:a_size_neg))

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
        # loop_VFI=loop_VFI,
        # loop_thres_a=loop_thres_a,
        # loop_thres_e=loop_thres_e,
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
    q_s_m::Array{Float64,3} # (a', e, h ≈ j)
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
        l = T .- n_grid
        c = h .* e_m .* n_grid .* (1.0 - ϕ)
        u = utility_function.(c, l, Ref(γ), Ref(ω), Ref(χ))
        u_max_i = argmax(u)
        @inbounds V_s_m_d[e_m_i, h_size] = u[u_max_i]
        @inbounds policy_s_m_d_n[e_m_i, h_size] = n_grid[u_max_i]

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

    # define repayment probability, pricing function, and risky borrowing limit
    q_s_m = ones(a_size, e_m_size, h_size - 1) ./ (1.0 + r_f)

    # return outputs
    variables = Mutable_Variables(V_s_m, E_V_s_m, V_s_m_r, V_s_m_d, policy_s_m_r_a, policy_s_m_r_n, policy_s_m_d, policy_s_m_d_n, q_s_m)
    return variables
end

function pricing_and_rbl_function!(
    h_i::Int64,
    variables::Mutable_Variables,
    parameters::NamedTuple)
    """
    update pricing function and borrowing risky limit
    """

    # unpack parameters
    @unpack e_m_size, e_m_grid, e_m_Γ, a_grid, a_size_neg, a_grid_neg, a_ind_zero, h_grid, κ_size, κ_grid, κ_Γ, r_f, τ, ϕ = parameters

    # risk-based borrowing price
    h_p = h_grid[h_i+1]
    for e_m_i in 1:e_m_size, a_p_i in 1:(a_size_neg-1)
        @inbounds a_p = a_grid[a_p_i]
        @inbounds variables.q_s_m[a_p_i, e_m_i, h_i] = 0.0
        for e_m_p_i in 1:e_m_size, κ_p_i in 1:κ_size
            e_m_p = e_m_grid[e_m_p_i]
            κ_p = κ_grid[κ_p_i]
            if variables.policy_s_m_d[a_p_i, κ_p_i, e_m_p_i, h_i+1] == 1.0
                @inbounds variables.q_s_m[a_p_i, e_m_i, h_i] += κ_Γ[κ_p_i] * e_m_Γ[e_m_i, e_m_p_i] * ((h_p * e_m_p * variables.policy_s_m_d_n[e_m_p_i, h_i+1]) * ϕ / (κ_p-a_p))
            else
                @inbounds variables.q_s_m[a_p_i, e_m_i, h_i] += κ_Γ[κ_p_i] * e_m_Γ[e_m_i, e_m_p_i]
            end
        end
        @inbounds variables.q_s_m[a_p_i, e_m_i, h_i] = clamp(variables.q_s_m[a_p_i, e_m_i, h_i], 0.0, 1.0) / (1.0 + r_f + τ)
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
    @unpack h_grid, a_size, a_grid, a_ind_zero, e_m_size, e_m_grid, κ_size, κ_grid, n_size, n_grid, T, γ, ω, ϕ, χ = parameters

    # loop over all states
    h = h_grid[h_i]
    for e_m_i in 1:e_m_size
        e_m = e_m_grid[e_m_i]

        # construct useful vectors
        @inbounds @views qa = variables.q_s_m[:, e_m_i, h_i] .* a_grid
        @inbounds @views EV = variables.E_V_s_m[:, e_m_i, h_i]

        # construct objective functions
        obj_r(a_p, CoH, l) = -(utility_function(CoH - qa_itp(a_p), l, γ, ω, χ) + EV_itp(a_p))

        # default
        l = T .- n_grid
        c = h .* e_m .* n_grid .* (1.0 - ϕ)
        u = utility_function.(c, l, Ref(γ), Ref(ω), Ref(χ)) .+ EV[a_ind_zero]
        u_max_i = argmax(u)
        @inbounds variables.V_s_m_d[e_m_i, h_i] = u[u_max_i]
        @inbounds variables.policy_s_m_d_n[e_m_i, h_i] = n_grid[u_max_i]

        # repayment
        for κ_i in 1:κ_size, a_i in 1:a_size
            a = a_grid[a_i]
            κ = κ_grid[κ_i, h_i]
            n = 0.0
            l = T - n
            c = h .* e_m .* n .+ a .- κ .- qa
            u = utility_function.(c, Ref(l), Ref(γ), Ref(ω), Ref(χ)) .+ EV
            u_max_i = argmax(u)
            variables.V_s_m_r[a_i, κ_i, e_m_i, h_i] = u[u_max_i]
            variables.policy_s_m_r_a[a_i, κ_i, e_m_i, h_i] = a_grid[u_max_i]
            variables.policy_s_m_r_n[a_i, κ_i, e_m_i, h_i] = n
            for n_i in 2:n_size
                n = n_grid[n_i]
                l = T - n
                c = h .* e_m .* n .+ a .- κ .- qa
                u = utility_function.(c, Ref(l), Ref(γ), Ref(ω), Ref(χ)) .+ EV
                u_max_i = argmax(u)
                if u[u_max_i] > variables.V_s_m_r[a_i, κ_i, e_m_i, h_i]
                    variables.V_s_m_r[a_i, κ_i, e_m_i, h_i] = u[u_max_i]
                    variables.policy_s_m_r_a[a_i, κ_i, e_m_i, h_i] = a_grid[u_max_i]
                    variables.policy_s_m_r_n[a_i, κ_i, e_m_i, h_i] = n
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
pricing_and_rbl_function!(parameters.h_size - 1, variables, parameters)
E_V_function!(parameters.h_size - 1, variables, parameters)
value_and_policy_function!(parameters.h_size - 1, variables, parameters)

pricing_and_rbl_function!(parameters.h_size - 2, variables, parameters)
E_V_function!(parameters.h_size - 2, variables, parameters)
value_and_policy_function!(parameters.h_size - 2, variables, parameters)

pricing_and_rbl_function!(parameters.h_size - 3, variables, parameters)
E_V_function!(parameters.h_size - 3, variables, parameters)
value_and_policy_function!(parameters.h_size - 3, variables, parameters)

pricing_and_rbl_function!(parameters.h_size - 4, variables, parameters)
E_V_function!(parameters.h_size - 4, variables, parameters)
value_and_policy_function!(parameters.h_size - 4, variables, parameters)

pricing_and_rbl_function!(parameters.h_size - 5, variables, parameters)
E_V_function!(parameters.h_size - 5, variables, parameters)
value_and_policy_function!(parameters.h_size - 5, variables, parameters)

pricing_and_rbl_function!(parameters.h_size - 6, variables, parameters)
E_V_function!(parameters.h_size - 6, variables, parameters)
value_and_policy_function!(parameters.h_size - 6, variables, parameters)

# cheching figures
using Plots
plot(parameters.a_grid_neg, variables.q_s_m[1:parameters.a_ind_zero, :, end-3])
plot(parameters.a_grid_neg, variables.q_s_m[1:parameters.a_ind_zero, :, end-3] .* parameters.a_grid_neg)
# plot(parameters.a_grid, variables.q_s_m[:, :, end] .* parameters.a_grid)

plot(parameters.a_grid_neg, variables.q_s_m[1:parameters.a_ind_zero, 3, end-4:end], legend=:topleft)

# plot(parameters.a_grid_neg, variables.q_s_m[1:parameters.a_ind_zero, :, end-3])
