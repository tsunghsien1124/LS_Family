# Prototype model, version where the multi-threading was adjusted for cluster speed

using Parameters
using LinearAlgebra

using ProgressMeter

# using NLopt
# using Dierckx

using Optim
using Optim: converged, maximum, maximizer, minimizer, iterations
using Interpolations
# using LineSearches

using Distributions

using Cubature

# using Plots
# using StatsFuns

println("Julia is running with $(Threads.nthreads()) threads...")

include("GQ_algorithm.jl")

function parameters_function(;
    β::Real = 0.95,                              # Discount factor
    # β::Real = 0.9,
    λ::Real = 0.5,                              # Utility weight for couples
    # r::Real = 0.04,
    r::Real = 0.014,
    ρ::Real = 0.975,                            # survival probability
    # a_size_neg::Integer = 30,                   # number of  assets
    # a_size_neg::Integer = 71,
    a_size_neg::Integer = 15,
    # a_size_pos::Integer = 30,
    a_size_pos::Integer = 30,
    # a_size_pos::Integer = 230,
    # a_size_pos::Integer = 5,
    # a_min::Real = -1.00,                       # minimum of assets
    a_min::Real = -0.75,
    # a_max::Real = 4.00,                        # maximum of assets
    a_max::Real = 10.00,
    # a_degree::Real = 1.0,                      # governs how grid is spaced
    # d_size::Integer = 2                        # Number of default/repayment options
    T::Real = 1.1,
    α::Real = 0.5,                             # Weight of leisure in utility function # From Alon et al. (2020)
    # α::Real = 1.27,
    γ_c::Real = 2.0,                             # CRRA parameter on consumption
    γ_l::Real = 3.0,                             # CRRA parameter on leisure
    # ϕ::Real = 0.355,                             # Wage garnishment
    ϕ::Real = 0.3,
    χ::Real = 0.1,                                 # Default Utility Stigma cost
    # θ::Real = 0.55,                              # Returns to labor # From Alon et al. (2020)
    θ::Real = 1.0,
    ν::Real = 0.4,                               # Share of singles in economy
    x_integration::Real = 0.2                    # Integration border
    # x_integration::Real = 0.15
    )
    """
    contruct an immutable object containing all parameters
    """

    # persistent productivity shock
    # z_grid = [0.37996271996003, 0.63109515876990, 0.86127208980130, 1.17540057527388, 1.95226945619490]
    # # z_grid = [0.35, 0.6, 0.85, 1.2, 1.9]
    # z_size = length(z_grid)
    # Γ_z = [0.86379548987220  0.13510724642768  0.00109682214183  0.00000033823300  0.00000010332529;
    # 0.13510724429090  0.67772725268309  0.18381378713248  0.00335137854872  0.00000033734481;
    # 0.00109682516680  0.18381378410751  0.63017878145138  0.18381378410751  0.00109682516680;
    # 0.00000033734481  0.00335137854872 0.18381378713248  0.67772725268309  0.13510724429090;
    # 0.00000010332529  0.00000033823300  0.00109682214183  0.13510724642768  0.86379548987220]
    # # Γ_z = [0.8  0.1  0.06  0.03  0.01;
    # #        0.1  0.65  0.15  0.07  0.03;
    # #        0.06  0.15  0.59  0.15  0.05;
    # #        0.03  0.07 0.15  0.63  0.12;
    # #        0.01  0.03  0.05  0.12  0.79]
    # # G_z = [G_e_L G_e_M 1-G_e_L-G_e_M]
    #
    # # Transitory productivity shock
    # η_grid = [0.61505879, 0.978521538, 1.556768907]
    # # η_grid = [0.25, 0.5, 1.0, 1.5, 1.75]
    # # η_grid = [0.15, 0.25, 0.5, 1.0, 1.5, 1.75]
    # η_size = length(η_grid)
    # Γ_η = [0.1, 0.8, 0.1]
    # # Γ_η = [0.05, 0.1, 0.7, 0.1, 0.05]
    # # Γ_η = [0.02, 0.05, 0.1, 0.7, 0.1, 0.03]

    z_grid = [0.574824402185664, 1.0, 1.7396617057273214]
    z_size = length(z_grid)
    Γ_z = [ 0.818    0.178  0.004;
            0.178    0.644  0.178;
            0.004  0.178  0.818]

    # Transitory productivity shock
    η_grid = [0.777792, 1.0, 1.28569]
    η_size = length(η_grid)
    Γ_η = [0.3333333333333333, 0.3333333333333333, 0.3333333333333333]

    # Expense shocks
    # κ_grid = [0.0 0.264 0.8218]
    # κ_grid = [0.0 0.1 0.4]
    # κ_grid = [0.0 0.1 0.2]
    # κ_grid = [0.0 0.05 0.1]
    # κ_grid = [0.0 0.0 0.0]
    # κ_size = length(κ_grid)
    # Γ_κ = [(1.0-0.07104-0.0046), 0.07104, 0.0046]
    # Γ_κ = [(1.0-0.15-0.05), 0.15, 0.05]
    # Γ_κ = [1.0]

    # Asset grid
    a_grid_neg = collect(range(a_min, 0.0, length=a_size_neg))
    a_grid_pos = collect(range(0.0, a_max, length=a_size_pos))
    a_grid = cat(a_grid_neg[1:(end-1)], a_grid_pos, dims = 1)

    # a_grid = collect(range(a_min, a_max, step=0.2))

    # a_grid_neg = reverse((exp.(collect(range(log(1.0), stop=log(-a_min+1), length = a_size_neg))).-1)*(-1))
    # a_grid_pos = exp.(collect(range(log(1.0), stop=log(a_max+1), length = a_size_pos))).-1
    # a_grid = cat(a_grid_neg[1:(end-1)], a_grid_pos, dims = 1)

    a_ind_zero = findall(iszero, a_grid)[]
    # a_size_neg = length(a_grid_neg)
    a_size_neg = length(a_grid[1:a_ind_zero])
    a_size = length(a_grid)

    # Asset grid for distribution
    # dist_factor = 2
    dist_factor = 1
    a_grid_neg_dist = collect(range(a_min, 0.0, length=dist_factor*a_size_neg))
    a_grid_pos_dist = collect(range(0.0, a_max, length=dist_factor*a_size_pos))
    a_grid_dist = cat(a_grid_neg_dist[1:(end-1)], a_grid_pos_dist, dims = 1)

    a_ind_zero_dist = findall(iszero, a_grid_dist)[]
    a_size_dist = length(a_grid_dist)

    # Labor
    # n_grid = collect(range(0.0, 1.0, length=15))
    # n_grid = collect(range(0.0, 1.0, length=10))
    n_grid = collect(range(0.0, 1.0, length=5))
    # n_grid = collect(range(0.0, 1.0, length=3))
    n_size = length(n_grid)

    BM_indices = BM_function(a_size)

    # return the outcome
    return (β=β, λ=λ, r=r, ρ=ρ, T=T, α=α, γ_c=γ_c, γ_l=γ_l, ϕ=ϕ, χ=χ, θ=θ, ν=ν,
    a_grid=a_grid, a_size=a_size, a_ind_zero=a_ind_zero, a_min=a_min, a_max=a_max,
    a_grid_dist=a_grid_dist, a_size_dist=a_size_dist, a_ind_zero_dist=a_ind_zero_dist,
    z_grid=z_grid, z_size=z_size, Γ_z=Γ_z, η_grid=η_grid, η_size=η_size, Γ_η=Γ_η, BM_indices = BM_indices, n_grid=n_grid, n_size = n_size, x_integration = x_integration)
end

mutable struct MutableVariables
    """
    construct a type for mutable variables
    """
    # U::Array{Float64,2}
    V_S::Array{Float64,3}
    V_S_R::Array{Float64,3}
    V_S_D::Array{Float64,2}
    # V_C_R::Array{Float64,5}
    # V_C_D::Array{Float64,4}
    # V_C::Array{Float64,7}
    # c_S::Array{Float64,3}
    # c_C_1::Array{Float64,5}
    # c_C_2::Array{Float64,5}
    n_S::Array{Float64,3}
    # n_C_1_i::Array{Float64,7}
    # n_C_2_i::Array{Float64,7}
    # l_S::Array{Float64,3}
    # l_C_1::Array{Float64,5}
    # l_C_2::Array{Float64,5}
    a_S::Array{Float64,3}
    # a_C_i::Array{Int64,7}
    d_S_i::Array{Int64,3}
    # d_C_i::Array{Int64,7}
    P_S::Array{Float64,2}
    q_S::Array{Float64,2}
    # P_C::Array{Float64,3}
    # q_C::Array{Float64,3}
    μ_S::Array{Float64,3}
    # μ_C::Array{Float64,7}
end

function variables_function(
    parameters::NamedTuple;
    load_initial_value::Bool = false
    )
    """
    construct a mutable object containing endogenous variables
    """

    # unpack parameters
    @unpack  a_size, a_size_dist, z_size, η_size, ρ, r, ν = parameters

    # Singles' value functions
    V_S = zeros(a_size, z_size, η_size)
    V_S_R = zeros(a_size, z_size, η_size)
    V_S_D = zeros(z_size, η_size)

    # V_C_R = zeros(a_size, z_size, z_size, η_size, η_size)
    # V_C_D = zeros(z_size, z_size, η_size, η_size)
    # V_C = zeros(a_size, z_size, z_size, η_size, η_size, κ_size, κ_size)

    # Policy Functions
    # Consumption
    # c_S = zeros(a_size, z_size, η_size)
    # c_C_1 = zeros(a_size, z_size, z_size, η_size, η_size)
    # c_C_2 = zeros(a_size, z_size, z_size, η_size, η_size)

    # Labor
    n_S = ones(Float64,a_size, z_size, η_size).*0.5
    # n_C_1_i = ones(Float64,a_size, z_size, z_size, η_size, η_size, κ_size, κ_size).*0.5
    # n_C_2_i = ones(Float64,a_size, z_size, z_size, η_size, η_size, κ_size, κ_size).*0.5

    # Leisure
    # l_S = zeros(a_size, z_size, η_size)
    # l_C_1 = zeros(a_size, z_size, z_size, η_size, η_size)
    # l_C_2 = zeros(a_size, z_size, z_size, η_size, η_size)

    # Asset
    a_S = zeros(Float64,a_size, z_size, η_size)
    # a_C_i = zeros(Int64,a_size, z_size, z_size, η_size, η_size, κ_size, κ_size)

    # Default
    d_S_i = zeros(Int64,a_size, z_size, η_size)
    # d_C_i = zeros(Int64,a_size, z_size, z_size, η_size, η_size, κ_size, κ_size)

    # Loan pricing
    # Singles
    P_S = ones(a_size, z_size)
    q_S = ones(a_size, z_size) .* ρ/(1.0 + r)

    # Couples
    # P_C = ones(a_size, z_size, z_size)
    # q_C = ones(a_size, z_size, z_size) .* ρ/(1.0 + r)

    # cross-sectional distribution
    μ_S = ones(a_size_dist, z_size, η_size) ./ (a_size_dist*z_size*η_size)
    # μ_C = ones(a_size, z_size, z_size, η_size, η_size, κ_size, κ_size) ./ (a_size*z_size*z_size*η_size*η_size*κ_size*κ_size)

    # return the outcome
    # variables = MutableVariables(V_S, V_C, n_S, n_C_1_i, n_C_2_i, a_S, a_C_i, d_S_i, d_C_i, P_S, q_S, P_C, q_C, μ_S, μ_C)
    variables = MutableVariables(V_S, V_S_R, V_S_D, n_S, a_S, d_S_i, P_S, q_S, μ_S)
    return variables
end

function utility_function(
    c::Real,
    l::Real
    )
    """
    compute utility of CRRA utility function
    """
    @unpack α, γ_c, γ_l = parameters
    if (c > 0.0) && (l > 0.0)
        # return log(c) + α*log(l)
        return (c^(1.0-γ_c)-1.0)/(1.0-γ_c) + α*((l^(1.0-γ_l)-1.0)/(1.0-γ_l))
    else
        return -Inf
    end
end

# function Util_R_S_optim_function(
#     n::Vector, z, η, a, q_S, a_p, κ, V_expect, θ, α, T, β, ρ
#     )
#
#     @inbounds l = T - n[1]
#     @inbounds c = z*η*(n[1]^(θ))+a-q_S*a_p-κ
#
#     if (c > 0.0) && (l > 0.0)
#         return log(c) + α*log(l) + β*ρ*V_expect
#     else
#         return -Inf
#     end
# end
#
# function Util_D_S_optim_function(
#     n::Vector, z, η, V_expect, θ, α, T, β, ρ, ϕ
#     )
#
#     @inbounds l = T - n[1]
#     # c = z*η*(n^(θ))-ϕ
#     @inbounds c = (z*η*(n[1]^(θ)))*(1.0-ϕ)
#
#     if (c > 0.0) && (l > 0.0)
#         return log(c) + α*log(l) + β*ρ*V_expect
#     else
#         return -Inf
#     end
# end
#
# function Util_R_C_optim_function(
#     n::Vector, z_1, z_2, η_1, η_2, a, q_C, a_p, κ_1, κ_2, V_expect,θ,α,T,λ,β,ρ
#     )
#
#     @inbounds l_1 = T - n[1]
#     @inbounds l_2 = T - n[2]
#     @inbounds c = z_1*η_1*(n[1]^θ)+z_2*η_2*(n[2]^θ)+a-q_C*a_p-κ_1-κ_2
#
#     if (c > 0.0) && (l_1 > 0.0) && (l_2 > 0.0)
#         return λ*log(c/2) + (1-λ)*log(c/2) + λ*α*log(l_1) + (1-λ)*α*log(l_2) + β*ρ*V_expect
#     else
#         return -Inf
#     end
# end
#
# function Util_D_C_optim_function(
#     n::Vector, z_1, z_2, η_1, η_2, V_expect, θ, α, T, λ, β, ρ, ϕ
#     )
#
#     @inbounds l_1 = T - n[1]
#     @inbounds l_2 = T - n[2]
#     # c = z_1*η_1*(n_1^θ)+z_2*η_2*(n_2^θ)-ϕ
#     @inbounds c = (z_1*η_1*(n[1]^θ)+z_2*η_2*(n[2]^θ))*(1.0-ϕ)
#
#     if (c > 0.0) && (l_1 > 0.0) && (l_2 > 0.0)
#         return λ*log(c/2) + (1-λ)*log(c/2) + λ*α*log(l_1) + (1-λ)*α*log(l_2) + β*ρ*V_expect
#     else
#         return -Inf
#     end
# end

function value_function_singles!(
    V_S_p::Array{Float64,3},
    variables::MutableVariables,
    parameters::NamedTuple;
    slow_updating::Real = 1.0
    )
    """
    compute feasible set and (un)conditional value functions
    """

    @unpack  a_size, a_min, a_max, z_size, η_size, a_grid, z_grid, η_grid, T, θ, β, ρ, Γ_z, Γ_η, ϕ, χ, a_ind_zero, BM_indices, α, n_grid, n_size, x_integration = parameters

    # V_expect_mat = reshape(V_S_p,a_size,:)*transpose(kron(parameters.Γ_z,kron(reshape(parameters.Γ_η,1,:),reshape(parameters.Γ_κ,1,:))))

    # V_expect_mat = reshape(V_S_p,a_size,:)*transpose(kron(reshape(Γ_κ,1,:),kron(reshape(Γ_η,1,:),Γ_z)))

    V_expect_mat = reshape(V_S_p,a_size,:)*transpose(kron(reshape(parameters.Γ_η,1,:),parameters.Γ_z))

    variables.V_S .= -Inf

    variables.V_S_R .= -Inf
    variables.V_S_D .= -Inf

    # interp_V = LinearInterpolation((a_grid,z_grid,η_grid),V_S_p,extrapolation_bc=Line())
    #
    # x_min = [-x_integration,-x_integration]
    # x_max = [x_integration,x_integration]
    #
    # V_expect_mat=zeros(a_size,z_size)
    #
    # Threads.@threads for z_i in 1:z_size
    #     z = z_grid[z_i]
    #
    #     for a_p_i in 1:a_size
    #         a_p = a_grid[a_p_i]
    #
    #         # f(x) = interp_V(a_p,x[1],x[2])*pdf(Normal(ρ*z,0.0426),x[1])*pdf(Normal(0.0,0.0421),x[2])
    #
    #         f(x) = interp_V(a_p,exp(x[1] + ρ*log(z)),exp(x[2]))*pdf(Normal(0.0,0.0426),x[1])*pdf(Normal(0.0,0.0421),x[2])
    #
    #         # (V_expect,err) = hcubature(f::Function, x_min, x_max; reltol=1e-8, abstol=0, maxevals=0)
    #
    #         (V_expect,err) = hcubature(f::Function, x_min, x_max; abstol=1e-6, maxevals=0)
    #
    #         V_expect_mat[a_p_i,z_i] = V_expect
    #     end
    # end

    # interp_q = LinearInterpolation((a_grid,z_grid), variables.q_S)

    # For Repayment
    # variables.V_S_R .= -Inf
    # Threads.@threads for a_i in 1:a_size
    #     for  κ_i in 1:κ_size, η_i in 1:η_size, z_i in 1:z_size

    Threads.@threads for i in CartesianIndices((1:z_size,1:η_size))

        # lb_new = 1

        # non-defaulting value
        BM_bounds = zeros(Int,a_size,3)
        @inbounds BM_bounds[:,1] = 1:a_size
        @inbounds BM_bounds[1,2] = 1
        @inbounds BM_bounds[1,3] = a_size
        @inbounds BM_bounds[a_size,3] = a_size

        @inbounds z_i = i[1]
        @inbounds η_i = i[2]
        # @inbounds κ_i = i[3]

        @inbounds z = z_grid[z_i]
        @inbounds η = η_grid[η_i]

        interp_V_expect = LinearInterpolation(a_grid, V_expect_mat[:,z_i],extrapolation_bc=Interpolations.Line())
        interp_q = LinearInterpolation(a_grid, variables.q_S[:,z_i],extrapolation_bc=Interpolations.Line())

        # x_test = range(a_min, a_max, step=0.1)
        #
        # interp_V = CubicSplineInterpolation(x_test, V_expect_mat[:,z_i])
        # interp_q = CubicSplineInterpolation(x_test, variables.q_S[:,z_i])

        for BM_i in 1:a_size

            @views @inbounds a_i, lb_i, ub_i = BM_indices[BM_i,:]
            @inbounds lb = BM_bounds[lb_i,2]
            @inbounds ub = BM_bounds[ub_i,3]

            # lb = 1
            # ub = a_size

            if ub != 0

                @inbounds a = a_grid[a_i]

                value_temp = zeros(n_size,a_size)
                value_temp .= -Inf

                for n_i in 1:n_size

                    @inbounds n = n_grid[n_i]

                    for a_p_i in lb:ub
                        @inbounds q_S = variables.q_S[a_p_i,z_i]

                        @inbounds a_p = a_grid[a_p_i]

                        l = T - n
                        @inbounds c = z*η*(n^(θ))+a-variables.q_S[a_p_i,z_i]*a_p

                        # V_expect = 0.0
                        #
                        # for κ_p_i in 1:κ_size, η_p_i in 1:η_size, z_p_i in 1:z_size
                        #     @inbounds V_expect += Γ_z[z_i,z_p_i]*Γ_η[η_p_i]*Γ_κ[κ_p_i]*V_S_p[a_p_i,z_p_i,η_p_i,κ_p_i]
                        # end

                        # x_min = [z_grid[1],η_grid[1]]
                        # x_max = [z_grid[end],η_grid[end]]

                        # x_min = [-0.25,-0.25]
                        # x_max = [0.25,0.25]
                        #
                        # # f(x) = interp_V(a_p,x[1],x[2])*pdf(Normal(ρ*z,0.0426),x[1])*pdf(Normal(0.0,0.0421),x[2])
                        #
                        # f(x) = interp_V(a_p,exp(x[1] + ρ*log(z)),exp(x[2]))*pdf(Normal(0.0,0.0426),x[1])*pdf(Normal(0.0,0.0421),x[2])
                        #
                        # # (V_expect,err) = hcubature(f::Function, x_min, x_max; reltol=1e-8, abstol=0, maxevals=0)
                        #
                        # (V_expect,err) = hcubature(f::Function, x_min, x_max; abstol=1e-8, maxevals=0)

                        # @inbounds V_expect = V_expect_mat[a_p_i,z_i]


                        # n_min = max(((q_S*a_p+κ-a)/(z*η)),0.0)^(1/θ)
                        #
                        # if n_min > 1
                        #     optf = -Inf
                        # else
                        #
                        #     f(n::Vector,grad::Vector) = Util_R_S_optim_function(n,z,η,a,q_S,a_p,κ,V_expect,θ,α,T,β,ρ)
                        #
                        #     opt = Opt(:LN_COBYLA,1)
                        #     opt.max_objective = f
                        #     opt.lower_bounds = max(n_min,0.001)
                        #     opt.upper_bounds = 1.0
                        #     opt.xtol_rel = 1e-4
                        #
                        #     if variables.n_S_i[a_i,z_i,η_i,κ_i] < n_min
                        #         n_initial = n_min
                        #     else
                        #         @inbounds n_initial = variables.n_S_i[a_i,z_i,η_i,κ_i]
                        #     end
                        #
                        #     (optf,optx,ret) = optimize(opt,[n_initial])
                        # end

                        value_temp[n_i,a_p_i] = utility_function(c,l) + β*ρ*V_expect_mat[a_p_i,z_i]

                        # if isnan(optf) || isnan(optx[1])
                        #     println(optf)
                        #     println(optx)
                        #     println(ret)
                        #     error("NaN in opt S,R")
                        # end

                        # Test
                        # if optf > v_highest
                        #     v_highest = optf
                        #     if v_flag == 1
                        #         error("Non-concavity detected")
                        #     end
                        # elseif optf < v_highest
                        #     v_flag = 1
                        # end

                        # results = optimize(n -> -1*Util_optim_function(n,z,η,θ,a,q_S,a_p,κ,V_expect,T),0.0, 1.0)

                        # v_temp = utility_function(c,l) + β*ρ*V_expect

                        # v_temp = optf

                        # if v_temp > v_temp_pre
                        #     v_temp_pre = v_temp
                        #     lb_new = a_p_i
                        # end

                        # if (a_p_i == a_size) && (n_i == n_size) && (v_temp_pre == -Inf)
                        #     lb_new = 1
                        # end
                    end
                end

                    if !isinf(maximum(value_temp))

                        index_max = argmax(value_temp)

                        n_i_max = index_max[1]
                        a_p_i_max = index_max[2]

                        if a_p_i_max == 1
                            a_p_i_upper = 2
                            a_p_i_lower = 1
                            a_p_upper = a_grid[a_p_i_upper]
                            a_p_lower = a_grid[a_p_i_lower]
                            initial_a_p = (a_p_upper + a_p_lower)/2
                        elseif a_p_i_max == a_size
                            a_p_i_upper = a_size
                            a_p_i_lower = a_size-1
                            a_p_upper = a_grid[a_p_i_upper]
                            a_p_lower = a_grid[a_p_i_lower]
                            initial_a_p = (a_p_upper + a_p_lower)/2
                        else
                            a_p_i_upper = a_p_i_max+1
                            a_p_i_lower = a_p_i_max-1
                            a_p_upper = a_grid[a_p_i_upper]
                            a_p_lower = a_grid[a_p_i_lower]
                            initial_a_p = a_grid[a_p_i_max]
                        end

                        if n_i_max == 1
                            n_i_upper = 2
                            n_i_lower = 1
                            n_upper = n_grid[n_i_upper]
                            n_lower = n_grid[n_i_lower]
                            initial_n = (n_upper + n_lower)/2
                        elseif n_i_max == length(parameters.n_grid)
                            n_i_upper = length(parameters.n_grid)
                            n_i_lower = length(parameters.n_grid)-1
                            n_upper = n_grid[n_i_upper]
                            n_lower = n_grid[n_i_lower]
                            initial_n = (n_upper + n_lower)/2
                        else
                            n_i_upper = n_i_max+1
                            n_i_lower = n_i_max-1
                            n_upper = n_grid[n_i_upper]
                            n_lower = n_grid[n_i_lower]
                            initial_n = n_grid[n_i_max]
                        end

                        upper_bound = [n_upper,a_p_upper]
                        lower_bound = [n_lower,a_p_lower]

                        initial_x = [initial_n,initial_a_p]

                        # if all(a_p_temp .== -Inf)
                        #     # interp = x -> -Inf
                        #     optf = -Inf
                        # elseif (sum(a_p_temp .!= -Inf) == 1.0) || (sum(a_p_temp .!= -Inf) == 2.0)
                        #     optf = a_p_temp[a_p_i_max]
                        #     optx = a_grid[a_p_i_max]
                        # else

                        # l = T - n

                        # opt = optimize(x -> -(utility_function(z*η*(n^(θ))+a-interp_q(x)*x,l)+β*ρ*interp_V_expect(x)), a_p_lower, a_p_upper, GoldenSection())

                        # opt = optimize(x -> -(utility_function(z*η*(n^(θ))+a-interp_q(x)*x,T-n)+β*ρ*interp_V_expect(x)), a_p_lower, a_p_upper, GoldenSection())

                        # inner_optimizer = NelderMead()
                        # inner_optimizer = LBFGS(linesearch=LineSearches.BackTracking())

                        f(x) = -(utility_function(z*η*(x[1]^(θ))+a-interp_q(x[2])*x[2],T-x[1])+β*ρ*interp_V_expect(x[2]))

                        # opt = optimize(f, lower_bound, upper_bound, initial_x, Fminbox(inner_optimizer))
                        opt = optimize(f, lower_bound, upper_bound, initial_x, SAMIN(f_tol=1e-6,x_tol=1e-4,verbosity=0),Optim.Options(iterations=10^6))

                        # converged(opt) || error("Optimization failed to converge")
                        optx_n, optx_a_p = opt.minimizer
                        optf = -opt.minimum
                        # end

                        if optf > variables.V_S_R[a_i,z_i,η_i]
                            variables.V_S_R[a_i,z_i,η_i] = optf
                        end

                        if optf > variables.V_S[a_i,z_i,η_i]
                            @inbounds variables.V_S[a_i,z_i,η_i] = optf
                            @inbounds variables.d_S_i[a_i,z_i,η_i] = 1
                            @inbounds variables.a_S[a_i,z_i,η_i] = optx_a_p
                            # @inbounds variables.a_S[a_i,z_i,η_i,κ_i] = a_grid[a_p_i_max]
                            # @inbounds variables.n_S_i[a_i,z_i,η_i,κ_i] = optx[1]
                            @inbounds variables.n_S[a_i,z_i,η_i] = optx_n
                            @inbounds BM_bounds[a_i,2] = a_p_i_lower
                            @inbounds BM_bounds[a_i,3] = a_p_i_upper
                        # elseif optf < variables.V_S[a_i,z_i,η_i,κ_i]
                        #     break
                        end
                    end
                # end

                if variables.V_S[a_i,z_i,η_i] == -Inf
                    @inbounds BM_bounds[a_i,2] = 1
                    @inbounds BM_bounds[a_i,3] = 0
                end

            else
                @inbounds BM_bounds[a_i,2] = 1
                @inbounds BM_bounds[a_i,3] = 0
            end
        end
    end

    # For Default
    # variables.V_S_D .= -Inf
    # Threads.@threads for z_i in 1:z_size
    #     for η_i in 1:η_size

    Threads.@threads for i in CartesianIndices((1:z_size,1:η_size))

        @inbounds z_i = i[1]
        @inbounds η_i = i[2]

            @inbounds z = z_grid[z_i]
            @inbounds η = η_grid[η_i]

            # @inbounds V_expect = V_expect_mat[a_ind_zero,z_i]

            # interp_V = LinearInterpolation((z_grid,η_grid),V_S_p[a_ind_zero,:,:],extrapolation_bc=Line())
            #
            # x_min = [-x_integration,-x_integration]
            # x_max = [x_integration,x_integration]
            #
            # f(x) = interp_V(exp(x[1] + ρ*log(z)),exp(x[2]))*pdf(Normal(0.0,0.0426),x[1])*pdf(Normal(0.0,0.0421),x[2])
            #
            # (V_expect,err) = hcubature(f::Function, x_min, x_max; abstol=1e-8, maxevals=0)

            value_temp = zeros(n_size)
            value_temp .= -Inf

            for n_i in 1:n_size
                @inbounds n = n_grid[n_i]

                l = T - n
                # c = z*η*(n^(θ))-ϕ
                c = (z*η*(n^(θ)))*(1.0-ϕ)

                # V_expect = 0.0
                #
                # for κ_p_i in 1:κ_size, η_p_i in 1:η_size, z_p_i in 1:z_size
                #     @inbounds V_expect += Γ_z[z_i,z_p_i]*Γ_η[η_p_i]*Γ_κ[κ_p_i]*V_S_p[a_ind_zero,z_p_i,η_p_i,κ_p_i]
                # end

                # f(n::Vector,grad::Vector) = Util_D_S_optim_function(n,z,η,V_expect,θ,α,T,β,ρ,ϕ)
                #
                # opt = Opt(:LN_COBYLA,1)
                # opt.max_objective = f
                # opt.lower_bounds = 0.001
                # opt.upper_bounds = 1.0
                # opt.xtol_rel = 1e-4
                #
                # (optf,optx,ret) = optimize(opt,[0.5])

                value_temp[n_i] = utility_function(c,l) - χ + β*ρ*V_expect_mat[a_ind_zero,z_i]

                # if isnan(optf) || isnan(optx[1])
                #     error("NaN in opt S,D")
                # end

                # v_temp = utility_function(c,l) + β*ρ*V_expect
            end

            if !isinf(maximum(value_temp))

                n_i_max = argmax(value_temp)

                if n_i_max == 1
                    n_i_upper = 2
                    n_i_lower = 1
                    n_upper = n_grid[n_i_upper]
                    n_lower = n_grid[n_i_lower]
                    initial_x = (n_upper + n_lower)/2
                elseif n_i_max == length(parameters.n_grid)
                    n_i_upper = length(parameters.n_grid)
                    n_i_lower = length(parameters.n_grid)-1
                    n_upper = n_grid[n_i_upper]
                    n_lower = n_grid[n_i_lower]
                    initial_x = (n_upper + n_lower)/2
                else
                    n_i_upper = n_i_max+1
                    n_i_lower = n_i_max-1
                    n_upper = n_grid[n_i_upper]
                    n_lower = n_grid[n_i_lower]
                    initial_x = n_grid[n_i_max]
                end

                # upper_bound = [n_upper]
                # lower_bound = [n_lower]

                # inner_optimizer = GoldenSection()

                f(x) = -(utility_function(z*η*(x^(θ))*(1.0-ϕ),T-x)-χ+β*ρ*V_expect_mat[a_ind_zero,z_i])
                # f(x) = -(utility_function(z*η*(x^(θ))-ϕ,T-x)-χ+β*ρ*V_expect_mat[a_ind_zero,z_i])

                # opt = optimize(f, lower_bound, upper_bound, initial_x, Fminbox(inner_optimizer))
                opt = optimize(f, n_lower, n_upper, GoldenSection())

                converged(opt) || error("Optimization failed to converge")
                optx_n = opt.minimizer
                optf = -opt.minimum

                if optf > variables.V_S_D[z_i,η_i]
                    variables.V_S_D[z_i,η_i] = optf
                end

                for a_i in 1:a_size
                    if optf > variables.V_S[a_i,z_i,η_i]
                        @inbounds variables.V_S[a_i,z_i,η_i] = optf
                        @inbounds variables.d_S_i[a_i,z_i,η_i] = 2
                        @inbounds variables.a_S[a_i,z_i,η_i] = a_grid[a_ind_zero]
                        # @inbounds variables.n_S_i[a_i,z_i,η_i,κ_i] = optx[1]
                        @inbounds variables.n_S[a_i,z_i,η_i] = optx_n
                    end
                end
            end
            # end
#        end
    end

    if slow_updating != 1.0
        variables.V_S = slow_updating*variables.V_S + (1.0-slow_updating)*V_S_p
    end
end

function howard_singles!(
    variables::MutableVariables,
    parameters::NamedTuple;
    howard_iter::Real = 15
    # howard_iter::Real = 10
    # howard_iter::Real = 0
    )

    @unpack  a_size, a_min, a_max, z_size, η_size, a_grid, z_grid, η_grid, T, θ, β, ρ, Γ_z, Γ_η, ϕ, χ, a_ind_zero, BM_indices, x_integration = parameters

    iter = 1

    while iter <= howard_iter
        # copyto!(V_S_p_howard, variables.V_S)

        # V_expect_mat = reshape(variables.V_S,a_size,:)*transpose(kron(parameters.Γ_z,kron(reshape(parameters.Γ_η,1,:),reshape(parameters.Γ_κ,1,:))))

        # V_expect_mat = reshape(variables.V_S,a_size,:)*transpose(kron(reshape(parameters.Γ_κ,1,:),kron(reshape(parameters.Γ_η,1,:),parameters.Γ_z)))

        V_expect_mat = reshape(variables.V_S,a_size,:)*transpose(kron(reshape(parameters.Γ_η,1,:),parameters.Γ_z))

        # interp_V = LinearInterpolation((a_grid,z_grid,η_grid),variables.V_S,extrapolation_bc=Line())
        #
        # x_min = [-x_integration,-x_integration]
        # x_max = [x_integration,x_integration]
        #
        # V_expect_mat=zeros(a_size,z_size)
        #
        # Threads.@threads for z_i in 1:z_size
        #     z = z_grid[z_i]
        #
        #     for a_p_i in 1:a_size
        #         a_p = a_grid[a_p_i]
        #
        #         # f(x) = interp_V(a_p,x[1],x[2])*pdf(Normal(ρ*z,0.0426),x[1])*pdf(Normal(0.0,0.0421),x[2])
        #
        #         f(x) = interp_V(a_p,exp(x[1] + ρ*log(z)),exp(x[2]))*pdf(Normal(0.0,0.0426),x[1])*pdf(Normal(0.0,0.0421),x[2])
        #
        #         # (V_expect,err) = hcubature(f::Function, x_min, x_max; reltol=1e-8, abstol=0, maxevals=0)
        #
        #         (V_expect,err) = hcubature(f::Function, x_min, x_max; abstol=1e-6, maxevals=0)
        #
        #         V_expect_mat[a_p_i,z_i] = V_expect
        #     end
        # end

        Threads.@threads for i in CartesianIndices((1:z_size,1:η_size))

            @inbounds z_i = i[1]
            @inbounds η_i = i[2]

            @inbounds z = z_grid[z_i]
            @inbounds η = η_grid[η_i]

            interp_V = LinearInterpolation(a_grid, V_expect_mat[:,z_i])
            interp_q = LinearInterpolation(a_grid, variables.q_S[:,z_i])

            # x_test = range(a_min, a_max, step=0.1)
            #
            # interp_V = CubicSplineInterpolation(x_test, V_expect_mat[:,z_i])
            # interp_q = CubicSplineInterpolation(x_test, variables.q_S[:,z_i])

            for a_i in 1:a_size

                @inbounds a = a_grid[a_i]

                @inbounds a_p = variables.a_S[a_i,z_i,η_i]
                # @inbounds a_p = a_grid[a_p_i]
                @inbounds n = variables.n_S[a_i,z_i,η_i]

                if variables.d_S_i[a_i,z_i,η_i] == 1
                    l = T - n
                    @inbounds c = z*η*(n^(θ))+a-interp_q(a_p)*a_p

                    # V_expect = 0.0
                    #
                    # for κ_p_i in 1:κ_size, η_p_i in 1:η_size, z_p_i in 1:z_size
                    #     @inbounds V_expect += Γ_z[z_i,z_p_i]*Γ_η[η_p_i]*Γ_κ[κ_p_i]*V_S_p_howard[a_p_i,z_p_i,η_p_i,κ_p_i]
                    # end

                    # @inbounds V_expect = interp_V(a_p)

                    @inbounds variables.V_S[a_i,z_i,η_i] = utility_function(c,l) + β*ρ*interp_V(a_p)
                else
                    l = T - n
                    # c = z*η*(n^(θ))-ϕ
                    c = (z*η*(n^(θ)))*(1.0-ϕ)

                    # V_expect = 0.0
                    #
                    # for κ_p_i in 1:κ_size, η_p_i in 1:η_size, z_p_i in 1:z_size
                    #     @inbounds V_expect += Γ_z[z_i,z_p_i]*Γ_η[η_p_i]*Γ_κ[κ_p_i]*V_S_p_howard[a_ind_zero,z_p_i,η_p_i,κ_p_i]
                    # end

                    @inbounds V_expect = V_expect_mat[a_ind_zero,z_i]

                    @inbounds variables.V_S[a_i,z_i,η_i] = utility_function(c,l) - χ + β*ρ*V_expect
                end
            end
        end
        iter += 1
    end
end
#
# # Value function for couples
# function value_function_couples!(
#     V_C_p::Array{Float64,7},
#     variables::MutableVariables,
#     parameters::NamedTuple;
#     slow_updating::Real = 1.0
#     )
#     """
#     compute feasible set and (un)conditional value functions
#     """
#
#     @unpack  a_size, z_size, η_size, κ_size, a_grid, z_grid, η_grid, κ_grid, T, θ, β, ρ, Γ_z, Γ_η, Γ_κ, ϕ, a_ind_zero, λ, BM_indices, α, n_grid, n_size = parameters
#
#     V_expect_mat = reshape(V_C_p,a_size,:)*transpose(kron(reshape(parameters.Γ_κ,1,:),kron(reshape(parameters.Γ_κ,1,:),kron(reshape(parameters.Γ_η,1,:),kron(reshape(parameters.Γ_η,1,:),kron(parameters.Γ_z,parameters.Γ_z))))))
#
#     variables.V_C .= -Inf
#
#     # For Repayment
#     # variables.V_S_R .= -Inf
#     # Threads.@threads for a_i in 1:a_size
#     #     for κ_2_i in 1:κ_size, κ_1_i in 1:κ_size, η_2_i in 1:η_size, η_1_i in 1:η_size, z_2_i in 1:z_size, z_1_i in 1:z_size
#
#     Threads.@threads for i in CartesianIndices((1:z_size,1:η_size))
#         for κ_2_i in 1:κ_size, κ_1_i in 1:κ_size, η_2_i in 1:η_size, z_2_i in 1:z_size
#         # lb_new = 1
#
#         # non-defaulting value
#         BM_bounds = zeros(Int,a_size,3)
#         @inbounds BM_bounds[:,1] = 1:a_size
#         @inbounds BM_bounds[1,2] = 1
#         @inbounds BM_bounds[1,3] = a_size
#         @inbounds BM_bounds[a_size,3] = a_size
#
#         for BM_i in 1:a_size
#
#             @views @inbounds a_i, lb_i, ub_i = BM_indices[BM_i,:]
#             @inbounds lb = BM_bounds[lb_i,2]
#             @inbounds ub = BM_bounds[ub_i,3]
#
#             if ub != 0
#
#                 @inbounds z_1_i = i[1]
#                 # @inbounds z_2_i = i[2]
#                 @inbounds η_1_i = i[2]
#                 # @inbounds η_2_i = i[4]
#                 # @inbounds κ_1_i = i[5]
#                 # @inbounds κ_2_i = i[6]
#
#                 @inbounds a = a_grid[a_i]
#                 @inbounds z_1 = z_grid[z_1_i]
#                 @inbounds z_2 = z_grid[z_2_i]
#                 @inbounds η_1 = η_grid[η_1_i]
#                 @inbounds η_2 = η_grid[η_2_i]
#                 @inbounds κ_1 = κ_grid[κ_1_i]
#                 @inbounds κ_2 = κ_grid[κ_2_i]
#
#                 # lb = 1
#                 # if a_i > 1
#                 #     lb = lb_new
#                 # end
#                 #
#                 # v_temp_pre = -Inf
#
#                 for a_p_i in lb:ub, n_1_i in 1:n_size, n_2_i in 1:n_size
#
#                     @inbounds q_C = variables.q_C[a_p_i,z_1_i,z_2_i]
#
#                     @inbounds a_p = a_grid[a_p_i]
#                     @inbounds n_1 = n_grid[n_1_i]
#                     @inbounds n_2 = n_grid[n_2_i]
#
#                     @inbounds l_1 = T - n_1
#                     @inbounds l_2 = T - n_2
#                     @inbounds c = z_1*η_1*(n_1^θ)+z_2*η_2*(n_2^θ)+a-variables.q_C[a_p_i,z_1_i,z_2_i]*a_p-κ_1-κ_2
#
#                     # V_expect = 0.0
#                     #
#                     # for κ_2_p_i in 1:κ_size, κ_1_p_i in 1:κ_size, η_2_p_i in 1:η_size, η_1_p_i in 1:η_size, z_2_p_i in 1:z_size, z_1_p_i in 1:z_size
#                     #     @inbounds V_expect += Γ_z[z_1_i,z_1_p_i]*Γ_z[z_2_i,z_2_p_i]*Γ_η[η_1_p_i]*Γ_η[η_2_p_i]*Γ_κ[κ_1_p_i]*Γ_κ[κ_2_p_i]*V_C_p[a_p_i,z_1_p_i,z_2_p_i,η_1_p_i,η_2_p_i,κ_1_p_i,κ_2_p_i]
#                     # end
#
#                     @inbounds z_i = LinearIndices(Γ_z)[z_1_i,z_2_i]
#
#                     @inbounds V_expect = V_expect_mat[a_p_i,z_i]
#
#                     # n_1_min = max(((q_C*a_p+κ_1+κ_2-a-z_2*η_2*(0.99^θ))/(z_1*η_1)),0.0)^(1/θ)
#                     # n_2_min = max(((q_C*a_p+κ_1+κ_2-a-z_1*η_1*(0.99^θ))/(z_2*η_2)),0.0)^(1/θ)
#                     #
#                     # if (n_1_min > 1) || (n_2_min > 1)
#                     #     optf = -Inf
#                     # else
#                     #
#                     #     f(n::Vector,grad::Vector) = Util_R_C_optim_function(n,z_1,z_2,η_1,η_2,a,q_C,a_p,κ_1,κ_2,V_expect,θ,α,T,λ,β,ρ)
#                     #
#                     #     opt = Opt(:LN_COBYLA,2)
#                     #     opt.max_objective = f
#                     #     opt.lower_bounds = [max(n_1_min,0.001), max(n_2_min,0.001)]
#                     #     opt.upper_bounds = 1.0
#                     #     opt.xtol_rel = 1e-4
#                     #
#                     #     n_1_initial = max(n_1_min,0.99)
#                     #     n_2_initial = max(n_2_min,0.99)
#                     #
#                     #     (optf,optx,ret) = optimize(opt,[n_1_initial,n_2_initial])
#                     #
#                     #     # (optf,optx,ret) = optimize(opt,[variables.n_C_1_i[a_i,z_1_i,z_2_i,η_1_i,η_2_i,κ_1_i,κ_2_i],variables.n_C_2_i[a_i,z_1_i,z_2_i,η_1_i,η_2_i,κ_1_i,κ_2_i]])
#                     # end
#
#                     optf = λ*utility_function(c/2,l_1) + (1-λ)*utility_function(c/2,l_2)+ β*ρ*V_expect
#
#                     # if isnan(optf) || any(isnan.(optx))
#                     #     error("NaN in opt C,R")
#                     # end
#
#                     # v_temp = λ*utility_function(c/2,l_1) + (1-λ)*utility_function(c/2,l_2)+ β*ρ*V_expect
#
#                     # if v_temp > v_temp_pre
#                     #     v_temp_pre = v_temp
#                     #     lb_new = a_p_i
#                     # end
#
#                     # if (a_p_i == a_size) && (n_1_i == n_size) && (n_2_i == n_size) && (v_temp_pre == -Inf)
#                     #     lb_new = 1
#                     # end
#
#                     if optf > variables.V_C[a_i,z_1_i,z_2_i,η_1_i,η_2_i,κ_1_i,κ_2_i]
#                         @inbounds variables.V_C[a_i,z_1_i,z_2_i,η_1_i,η_2_i,κ_1_i,κ_2_i] = optf
#                         @inbounds variables.d_C_i[a_i,z_1_i,z_2_i,η_1_i,η_2_i,κ_1_i,κ_2_i] = 1
#                         @inbounds variables.a_C_i[a_i,z_1_i,z_2_i,η_1_i,η_2_i,κ_1_i,κ_2_i] = a_p_i
#                         # @inbounds variables.n_C_1_i[a_i,z_1_i,z_2_i,η_1_i,η_2_i,κ_1_i,κ_2_i] = optx[1]
#                         # @inbounds variables.n_C_2_i[a_i,z_1_i,z_2_i,η_1_i,η_2_i,κ_1_i,κ_2_i] = optx[2]
#                         @inbounds variables.n_C_1_i[a_i,z_1_i,z_2_i,η_1_i,η_2_i,κ_1_i,κ_2_i] = n_1
#                         @inbounds variables.n_C_2_i[a_i,z_1_i,z_2_i,η_1_i,η_2_i,κ_1_i,κ_2_i] = n_2
#                         @inbounds BM_bounds[a_i,2] = a_p_i
#                         @inbounds BM_bounds[a_i,3] = a_p_i
#                     # elseif optf < variables.V_C[a_i,z_1_i,z_2_i,η_1_i,η_2_i,κ_1_i,κ_2_i]
#                     #     break
#                     end
#                 end
#
#                 if variables.V_C[a_i,z_1_i,z_2_i,η_1_i,η_2_i,κ_1_i,κ_2_i] == -Inf
#                     @inbounds BM_bounds[a_i,2] = 1
#                     @inbounds BM_bounds[a_i,3] = 0
#
#                 end
#             else
#                 @inbounds BM_bounds[a_i,2] = 1
#                 @inbounds BM_bounds[a_i,3] = 0
#             end
#         end
#     end
#     end
#
#     # For Default
#     # variables.V_S_D .= -Inf
#     # Threads.@threads for z_1_i in 1:z_size
#     #     for η_2_i in 1:η_size, η_1_i in 1:η_size, z_2_i in 1:z_size
#
#     Threads.@threads for i in CartesianIndices((1:z_size,1:η_size))
#         for η_2_i in 1:η_size, z_2_i in 1:z_size
#
#             @inbounds z_1_i = i[1]
#             # @inbounds z_2_i = i[2]
#             @inbounds η_1_i = i[2]
#             # @inbounds η_2_i = i[4]
#
#             @inbounds z_1 = z_grid[z_1_i]
#             @inbounds z_2 = z_grid[z_2_i]
#             @inbounds η_1 = η_grid[η_1_i]
#             @inbounds η_2 = η_grid[η_2_i]
#
#             @inbounds z_i = LinearIndices(Γ_z)[z_1_i,z_2_i]
#
#             @inbounds V_expect = V_expect_mat[a_ind_zero,z_i]
#
#             for n_1_i in 1:n_size, n_2_i in 1:n_size
#                 @inbounds n_1 = n_grid[n_1_i]
#                 @inbounds n_2 = n_grid[n_2_i]
#
#                 l_1 = T - n_1
#                 l_2 = T - n_2
#                 # c = z_1*η_1*(n_1^θ)+z_2*η_2*(n_2^θ)-ϕ
#                 c = (z_1*η_1*(n_1^θ)+z_2*η_2*(n_2^θ))*(1.0-ϕ)
#
#                 # V_expect = 0.0
#                 #
#                 # for κ_2_p_i in 1:κ_size, κ_1_p_i in 1:κ_size, η_2_p_i in 1:η_size, η_1_p_i in 1:η_size, z_2_p_i in 1:z_size, z_1_p_i in 1:z_size
#                 #     @inbounds V_expect += Γ_z[z_1_i,z_1_p_i]*Γ_z[z_2_i,z_2_p_i]*Γ_η[η_1_p_i]*Γ_η[η_2_p_i]*Γ_κ[κ_1_p_i]*Γ_κ[κ_2_p_i]*V_C_p[a_ind_zero,z_1_p_i,z_2_p_i,η_1_p_i,η_2_p_i,κ_1_p_i,κ_2_p_i]
#                 # end
#
#                 # f(n::Vector,grad::Vector) = Util_D_C_optim_function(n,z_1,z_2,η_1,η_2,V_expect,θ,α,T,λ,β,ρ,ϕ)
#                 #
#                 # opt = Opt(:LN_COBYLA,2)
#                 # opt.max_objective = f
#                 # opt.lower_bounds = 0.001
#                 # opt.upper_bounds = 1.0
#                 # opt.xtol_rel = 1e-4
#                 #
#                 # (optf,optx,ret) = optimize(opt,[0.5,0.5])
#
#                 optf = λ*utility_function(c/2,l_1) + (1-λ)*utility_function(c/2,l_2)+ β*ρ*V_expect
#
#                 # if isnan(optf) || any(isnan.(optx))
#                 #     error("NaN in opt C,D")
#                 # end
#
#                 # v_temp = λ*utility_function(c/2,l_1) + (1-λ)*utility_function(c/2,l_2)+ β*ρ*V_expect
#
#                 for κ_2_i in 1:κ_size, κ_1_i in 1:κ_size, a_i in 1:a_size
#                     if optf > variables.V_C[a_i,z_1_i,z_2_i,η_1_i,η_2_i,κ_1_i,κ_2_i]
#                         @inbounds variables.V_C[a_i,z_1_i,z_2_i,η_1_i,η_2_i,κ_1_i,κ_2_i] = optf
#                         @inbounds variables.d_C_i[a_i,z_1_i,z_2_i,η_1_i,η_2_i,κ_1_i,κ_2_i] = 2
#                         @inbounds variables.a_C_i[a_i,z_1_i,z_2_i,η_1_i,η_2_i,κ_1_i,κ_2_i] = a_ind_zero
#                         # @inbounds variables.n_C_1_i[a_i,z_1_i,z_2_i,η_1_i,η_2_i,κ_1_i,κ_2_i] = optx[1]
#                         # @inbounds variables.n_C_2_i[a_i,z_1_i,z_2_i,η_1_i,η_2_i,κ_1_i,κ_2_i] = optx[2]
#                         @inbounds variables.n_C_1_i[a_i,z_1_i,z_2_i,η_1_i,η_2_i,κ_1_i,κ_2_i] = n_1
#                         @inbounds variables.n_C_2_i[a_i,z_1_i,z_2_i,η_1_i,η_2_i,κ_1_i,κ_2_i] = n_2
#                     end
#                 end
#             end
# #        end
#     end
#     end
#
#     # if slow_updating != 1.0
#     #     variables.W = slow_updating*variables.W + (1.0-slow_updating)*W_p
#     # end
# end
#
# function howard_couples!(
#     variables::MutableVariables,
#     parameters::NamedTuple;
#     howard_iter::Real = 15
#     # howard_iter::Real = 0
#     )
#
#     @unpack  a_size, z_size, η_size, κ_size, a_grid, z_grid, η_grid, κ_grid, T, θ, β, ρ, Γ_z, Γ_η, Γ_κ, ϕ, a_ind_zero, λ, BM_indices = parameters
#
#     iter = 1
#
#     while iter <= howard_iter
#         # copyto!(V_C_p_howard, variables.V_C)
#
#         V_expect_mat = reshape(variables.V_C,a_size,:)*transpose(kron(reshape(parameters.Γ_κ,1,:),kron(reshape(parameters.Γ_κ,1,:),kron(reshape(parameters.Γ_η,1,:),kron(reshape(parameters.Γ_η,1,:),kron(parameters.Γ_z,parameters.Γ_z))))))
#
#         Threads.@threads for i in CartesianIndices((1:z_size,1:η_size))
#             for κ_2_i in 1:κ_size, κ_1_i in 1:κ_size, η_2_i in 1:η_size, z_2_i in 1:z_size, a_i in 1:a_size
#             # @inbounds a_i = i[1]
#             @inbounds z_1_i = i[1]
#             # @inbounds z_2_i = i[3]
#             @inbounds η_1_i = i[2]
#             # @inbounds η_2_i = i[5]
#             # @inbounds κ_1_i = i[6]
#             # @inbounds κ_2_i = i[7]
#
#             @inbounds a = a_grid[a_i]
#             @inbounds z_1 = z_grid[z_1_i]
#             @inbounds z_2 = z_grid[z_2_i]
#             @inbounds η_1 = η_grid[η_1_i]
#             @inbounds η_2 = η_grid[η_2_i]
#             @inbounds κ_1 = κ_grid[κ_1_i]
#             @inbounds κ_2 = κ_grid[κ_2_i]
#
#             @inbounds a_p_i = variables.a_C_i[a_i,z_1_i,z_2_i,η_1_i,η_2_i,κ_1_i,κ_2_i]
#             @inbounds a_p = a_grid[a_p_i]
#             @inbounds n_1 = variables.n_C_1_i[a_i,z_1_i,z_2_i,η_1_i,η_2_i,κ_1_i,κ_2_i]
#             @inbounds n_2 = variables.n_C_2_i[a_i,z_1_i,z_2_i,η_1_i,η_2_i,κ_1_i,κ_2_i]
#
#             if variables.d_C_i[a_i,z_1_i,z_2_i,η_1_i,η_2_i,κ_1_i,κ_2_i] == 1
#                 @inbounds l_1 = T - n_1
#                 @inbounds l_2 = T - n_2
#                 @inbounds c = z_1*η_1*(n_1^θ)+z_2*η_2*(n_2^θ)+a-variables.q_C[a_p_i,z_1_i,z_2_i]*a_p-κ_1-κ_2
#
#                 # V_expect = 0.0
#                 #
#                 # for κ_2_p_i in 1:κ_size, κ_1_p_i in 1:κ_size, η_2_p_i in 1:η_size, η_1_p_i in 1:η_size, z_2_p_i in 1:z_size, z_1_p_i in 1:z_size
#                 #     @inbounds V_expect += Γ_z[z_1_i,z_1_p_i]*Γ_z[z_2_i,z_2_p_i]*Γ_η[η_1_p_i]*Γ_η[η_2_p_i]*Γ_κ[κ_1_p_i]*Γ_κ[κ_2_p_i]*V_C_p_howard[a_p_i,z_1_p_i,z_2_p_i,η_1_p_i,η_2_p_i,κ_1_p_i,κ_2_p_i]
#                 # end
#
#                 @inbounds z_i = LinearIndices(Γ_z)[z_1_i,z_2_i]
#
#                 @inbounds V_expect = V_expect_mat[a_p_i,z_i]
#
#                 @inbounds variables.V_C[a_i,z_1_i,z_2_i,η_1_i,η_2_i,κ_1_i,κ_2_i] = λ*utility_function(c/2,l_1) + (1-λ)*utility_function(c/2,l_2)+ β*ρ*V_expect
#             else
#                 l_1 = T - n_1
#                 l_2 = T - n_2
#                 # c = z_1*η_1*(n_1^θ)+z_2*η_2*(n_2^θ)-ϕ
#                 c = (z_1*η_1*(n_1^θ)+z_2*η_2*(n_2^θ))*(1.0-ϕ)
#
#                 # V_expect = 0.0
#                 #
#                 # for κ_2_p_i in 1:κ_size, κ_1_p_i in 1:κ_size, η_2_p_i in 1:η_size, η_1_p_i in 1:η_size, z_2_p_i in 1:z_size, z_1_p_i in 1:z_size
#                 #     @inbounds V_expect += Γ_z[z_1_i,z_1_p_i]*Γ_z[z_2_i,z_2_p_i]*Γ_η[η_1_p_i]*Γ_η[η_2_p_i]*Γ_κ[κ_1_p_i]*Γ_κ[κ_2_p_i]*V_C_p_howard[a_ind_zero,z_1_p_i,z_2_p_i,η_1_p_i,η_2_p_i,κ_1_p_i,κ_2_p_i]
#                 # end
#
#                 @inbounds z_i = LinearIndices(Γ_z)[z_1_i,z_2_i]
#
#                 @inbounds V_expect = V_expect_mat[a_ind_zero,z_i]
#
#                 @inbounds variables.V_C[a_i,z_1_i,z_2_i,η_1_i,η_2_i,κ_1_i,κ_2_i] = λ*utility_function(c/2,l_1) + (1-λ)*utility_function(c/2,l_2)+ β*ρ*V_expect
#             end
#         end
#     end
#         iter += 1
#     end
# end

function pricing_function!(
    q_S_p::Array{Float64,2},
    variables::MutableVariables,
    parameters::NamedTuple;
    slow_updating::Real = 1.0
    )

    @unpack  z_grid, η_grid, a_size, z_size, η_size, ρ, r, Γ_z, Γ_η, a_ind_zero, x_integration = parameters

    # x_min = [-x_integration,-x_integration]
    # x_max = [x_integration,x_integration]
    #
    # interp_V_S_D = LinearInterpolation((z_grid,η_grid),variables.V_S_D,extrapolation_bc=Interpolations.Line())

    # Singles
    # variables.P_S .= 0.0
    Threads.@threads for a_p_i in 1:(a_ind_zero-1)

        # # interp_d_S = LinearInterpolation((z_grid,η_grid),variables.d_S_i[a_p_i,:,:] .== 1,extrapolation_bc=Interpolations.Flat())
        #
        # interp_V_S_R = LinearInterpolation((z_grid,η_grid),variables.V_S_R[a_p_i,:,:],extrapolation_bc=Interpolations.Line())
        #
        # function interp_V_S_R_2(z,η)
        #     temp = interp_V_S_R(z,η)
        #     if isnan(temp)
        #         return -Inf
        #     else
        #         return temp
        #     end
        # end

        for z_i in 1:z_size

            # z = z_grid[z_i]
            #
            # # g(x) = interp_d_S(exp(x[1] + ρ*log(z)),exp(x[2]))*pdf(Normal(0.0,0.0426),x[1])*pdf(Normal(0.0,0.0421),x[2])
            #
            # g(x) = (interp_V_S_R_2(exp(x[1] + ρ*log(z)),exp(x[2])) > interp_V_S_D(exp(x[1] + ρ*log(z)),exp(x[2])))*pdf(Normal(0.0,0.0426),x[1])*pdf(Normal(0.0,0.0421),x[2])
            #
            # (R_expect,err) = hcubature(g::Function, x_min, x_max; abstol=1e-4, maxevals=0)

            R_expect = 0.0

            for η_p_i in 1:η_size, z_p_i in 1:z_size
                @inbounds R_expect += Γ_z[z_i,z_p_i]*Γ_η[η_p_i]*(variables.d_S_i[a_p_i,z_p_i,η_p_i] == 1)
            end

            @inbounds variables.P_S[a_p_i,z_i] = R_expect

            @inbounds variables.q_S[a_p_i,z_i] = ρ*R_expect/(1+r)
        end
    end

    # Couples
    # Threads.@threads for a_p_i in 1:(a_ind_zero-1)
    #     for z_2_i in 1:z_size, z_1_i in 1:z_size
    #         temp = 0.0
    #
    #         for κ_2_p_i in 1:κ_size, κ_1_p_i in 1:κ_size, η_2_p_i in 1:η_size, η_1_p_i in 1:η_size, z_2_p_i in 1:z_size, z_1_p_i in 1:z_size
    #             @inbounds temp += Γ_z[z_1_i,z_1_p_i]*Γ_z[z_2_i,z_2_p_i]*Γ_η[η_1_p_i]*Γ_η[η_2_p_i]*Γ_κ[κ_1_p_i]*Γ_κ[κ_2_p_i]*(variables.d_C_i[a_p_i,z_1_p_i,z_2_p_i,η_1_p_i,η_2_p_i,κ_1_p_i,κ_2_p_i] == 1)
    #         end
    #         @inbounds variables.P_C[a_p_i,z_1_i,z_2_i] = temp
    #
    #         @inbounds variables.q_C[a_p_i,z_1_i,z_2_i] = ρ*temp/(1+r)
    #     end
    # end

    # if slow_updating != 1.0
    #     # store previous pricing function
    #     q_b_p = similar(variables.q_b)
    #     copyto!(q_b_p, variables.q_b)
    # end
    #
    if slow_updating != 1.0
        variables.q_S = slow_updating*variables.q_S + (1.0-slow_updating)*q_S_p
    end
end

function stationary_distribution!(
    variables::MutableVariables,
    parameters::NamedTuple;
    tol::Real = tol_h,
    iter_max::Integer = iter_max,
    )

    # Unpack grids
    @unpack  a_size_dist, z_size, η_size, Γ_z, Γ_η, a_grid, a_grid_dist, a_ind_zero_dist, ρ = parameters

    # initialize matrices
    μ_S_p = similar(variables.μ_S)
    # μ_C_p = similar(variables.μ_C)

    # set up the criterion and the iteration number
    crit_μ = Inf
    iter_μ = 0
    prog_μ = ProgressThresh(tol, "Computing stationary distribution: ")

    while crit_μ > tol && iter_μ < iter_max

        # update distribution
        copyto!(μ_S_p, variables.μ_S)
        # copyto!(μ_C_p, variables.μ_C)

        # nullify distribution
        variables.μ_S .= 0.0
        # variables.μ_C .= 0.0

        # Threads.@threads for η_p_i in 1:η_size, z_p_i in 1:z_size
        Threads.@threads for i in CartesianIndices((1:z_size,1:η_size))
            @inbounds z_p_i = i[1]
            @inbounds η_p_i = i[2]

            # Singles
            for a_p_i in 1:a_size_dist

                μ_temp_survivor = 0.0
                μ_temp_newborn = 0.0

                for η_i in 1:η_size, z_i in 1:z_size

                    # Does this part work as intended?
                    # function interp_a_S_2(a)
                    #     # default_thresh_i = findfirst(variables.a_S[:,z_i,η_i] .!= 0.0)
                    #     # default_thresh = a_grid[default_thresh_i]
                    #
                    #     default_thresh_i = findfirst(variables.d_S_i[:,z_i,η_i] .≈ 1.0)
                    #     default_thresh = parameters.a_grid[default_thresh_i]
                    #
                    #     if a < default_thresh
                    #         return 0.0
                    #     else
                    #         interp_a_S = LinearInterpolation(a_grid, variables.a_S[:,z_i,η_i])
                    #         return interp_a_S(a)
                    #     end
                    # end

                    for a_i in 1:a_size_dist
                        # lower = findlast(interp_a_S_2(a_grid_dist[a_i]) .>= a_grid_dist)
                        # upper = findfirst(interp_a_S_2(a_grid_dist[a_i]) .<= a_grid_dist)
                        #
                        # if (a_p_i < lower) || (a_p_i > upper)
                        #     temp = 0.0
                        # elseif upper == lower
                        #     temp = 1.0
                        # elseif a_p_i == lower
                        #     temp = (a_grid_dist[upper] - interp_a_S_2(a_grid_dist[a_i]))/(a_grid_dist[upper]-a_grid_dist[lower])
                        # elseif a_p_i == upper
                        #     temp = (interp_a_S_2(a_grid_dist[a_i]) - a_grid_dist[lower])/(a_grid_dist[upper]-a_grid_dist[lower])
                        # end

                        lower = findlast(variables.a_S[a_i,z_i,η_i] .>= a_grid_dist)
                        upper = findfirst(variables.a_S[a_i,z_i,η_i] .<= a_grid_dist)

                        if (a_p_i < lower) || (a_p_i > upper)
                            temp = 0.0
                        elseif upper == lower
                            temp = 1.0
                        elseif a_p_i == lower
                            temp = (a_grid_dist[upper] - variables.a_S[a_i,z_i,η_i])/(a_grid_dist[upper]-a_grid_dist[lower])
                        elseif a_p_i == upper
                            temp = (variables.a_S[a_i,z_i,η_i] - a_grid_dist[lower])/(a_grid_dist[upper]-a_grid_dist[lower])
                        end

                        @inbounds μ_temp_survivor += ρ*Γ_z[z_i,z_p_i]*Γ_η[η_p_i]*temp*μ_S_p[a_i,z_i,η_i]

                        if a_p_i == a_ind_zero_dist
                            @inbounds μ_temp_newborn += (1-ρ)*Γ_η[η_p_i]*(z_p_i == 2)*μ_S_p[a_i,z_i,η_i]
                        end
                    end
                end

                # assign the result
                @inbounds variables.μ_S[a_p_i,z_p_i,η_p_i] += μ_temp_survivor + μ_temp_newborn
            end

            # Couples
            # for κ_2_p_i in 1:κ_size, κ_1_p_i in 1:κ_size, η_2_p_i in 1:η_size, z_2_p_i in 1:z_size, a_p_i in 1:a_size
            #
            #     μ_temp_survivor = 0.0
            #     μ_temp_newborn = 0.0
            #
            #     for κ_2_i in 1:κ_size, κ_1_i in 1:κ_size, η_2_i in 1:η_size, η_1_i in 1:η_size, z_2_i in 1:z_size, z_1_i in 1:z_size, a_i in 1:a_size
            #
            #         @inbounds μ_temp_survivor += ρ*Γ_z[z_1_i,z_p_i]*Γ_z[z_2_i,z_2_p_i]*Γ_η[η_p_i]*Γ_η[η_2_p_i]*Γ_κ[κ_1_p_i]*Γ_κ[κ_2_p_i]*(variables.a_C_i[a_i,z_1_i,z_2_i,η_1_i,η_2_i,κ_1_i,κ_2_i] == a_p_i)*μ_C_p[a_i,z_1_i,z_2_i,η_1_i,η_2_i,κ_1_i,κ_2_i]
            #         if a_p_i == a_ind_zero
            #             @inbounds μ_temp_newborn += (1-ρ)*Γ_η[η_p_i]*Γ_η[η_2_p_i]*(z_p_i == 3)*(z_2_p_i == 3)*(κ_1_p_i == 1)*(κ_2_p_i == 1)*μ_C_p[a_i,z_1_i,z_2_i,η_1_i,η_2_i,κ_1_i,κ_2_i]
            #         end
            #     end
            #
            #     # assign the result
            #     @inbounds variables.μ_C[a_p_i,z_p_i,z_2_p_i,η_p_i,η_2_p_i,κ_1_p_i,κ_2_p_i] += μ_temp_survivor + μ_temp_newborn
            # end
        end

        # report the progress
        # crit_μ = max(norm(variables.μ_S-μ_S_p, Inf),norm(variables.μ_C-μ_C_p, Inf))
        crit_μ = norm(variables.μ_S-μ_S_p, Inf)
        ProgressMeter.update!(prog_μ, crit_μ)

        # update the iteration number
        iter_μ += 1
    end
end

function solve_function!(
    variables::MutableVariables,
    parameters::NamedTuple;
    tol::Real = tol_h,
    iter_max::Integer = iter_max,
    one_loop::Bool = true
    )
    """
    solve model
    """

    if one_loop == true

        # initialize the iteration number and criterion
        iter = 0
        crit = Inf
        prog = ProgressThresh(tol, "Solving model (one-loop): ")

        # initialize the next-period functions
        V_S_p = similar(variables.V_S)
        # V_C_p = similar(variables.V_C)
        q_S_p = similar(variables.q_S)
        # q_C_p = similar(variables.q_C)

        # μ_p = similar(variables.μ)

        # V_S_p_howard = similar(variables.V_S)
        # V_C_p_howard = similar(variables.V_C)

        while crit > tol && iter < iter_max

            # copy previous unconditional value and loan pricing functions
            copyto!(V_S_p, variables.V_S)
            # copyto!(V_C_p, variables.V_C)
            copyto!(q_S_p, variables.q_S)
            # copyto!(q_C_p, variables.q_C)

            # copyto!(μ_p, variables.μ)

            # update value functions
            value_function_singles!(V_S_p, variables, parameters; slow_updating = 1.0)

            howard_singles!(variables, parameters)

            # value_function_couples!(V_C_p, variables, parameters; slow_updating = 1.0)

            # howard_couples!(variables, parameters)

            # compute payday loan price
            pricing_function!(q_S_p, variables, parameters; slow_updating = 1.0)

            # Compute stationary distribution
            # stationary_distribution!(μ_p, variables, parameters)

            # check convergence
            # crit = max(norm(variables.V_S .- V_S_p, Inf), norm(variables.V_C .- V_C_p, Inf), norm(variables.q_S .- q_S_p, Inf), norm(variables.q_C .- q_C_p, Inf))
            crit = max(norm(variables.V_S .- V_S_p, Inf), norm(variables.q_S .- q_S_p, Inf))

            # V_S_crit = norm(variables.V_S .- V_S_p, Inf)
            # V_S_crit_ind = argmax(abs.(variables.V_S .- V_S_p))
            # V_S_ind = variables.V_S[V_S_crit_ind]
            # V_S_p_ind = V_S_p[V_S_crit_ind]
            #
            # V_C_crit = norm(variables.V_C .- V_C_p, Inf)
            # V_C_crit_ind = argmax(abs.(variables.V_C .- V_C_p))
            # V_C_ind = variables.V_C[V_C_crit_ind]
            # V_C_p_ind = V_C_p[V_C_crit_ind]
            #
            # q_S_crit = norm(variables.q_S .- q_S_p, Inf)
            # q_S_crit_ind = argmax(abs.(variables.q_S .- q_S_p))
            #
            # q_C_crit = norm(variables.q_C .- q_C_p, Inf)
            # q_C_crit_ind = argmax(abs.(variables.q_C .- q_C_p))
            #
            # crit = max(V_S_crit, V_C_crit, q_S_crit, q_C_crit)

            # report progress
            ProgressMeter.update!(prog, crit)

            # println("V_S_crit = $V_S_crit at $V_S_crit_ind, V_S = $V_S_ind, V_S_p = $V_S_p_ind")
            # println("V_C_crit = $V_C_crit at $V_C_crit_ind, V_C = $V_C_ind, V_C_p = $V_C_p_ind")
            # println("q_S_crit = $q_S_crit at $q_S_crit_ind")
            # println("q_C_crit = $q_C_crit at $q_C_crit_ind")
            # println("V_S_crit = $V_S_crit")
            # println("q_S_crit = $q_S_crit")

            # update the iteration number
            iter += 1
            # println("iter = ", iter)
        end

    else

        # initialize the iteration number and criterion
        iter_q = 0
        crit_q = Inf
        prog_q = ProgressThresh(tol, "Solving loan pricing and type updating functions: ")

        # initialize the next-period function
        q_S_p = similar(variables.q_S)
        q_C_p = similar(variables.q_C)
        # μ_p = similar(variables.μ)

        while crit_q > tol && iter_q < iter_max

            # copy previous loan pricing function and type updating function
            copyto!(q_S_p, variables.q_S)
            copyto!(q_C_p, variables.q_C)
            # copyto!(μ_p, variables.μ)

            # initialize the iteration number and criterion
            iter_W = 0
            crit_W = Inf
            prog_W = ProgressThresh(tol, "Solving unconditional value function: ")

            # initialize the next-period function
            V_S_p = similar(variables.V_S)
            V_C_p = similar(variables.V_C)

            while crit_W > tol && iter_W < iter_max

                # copy previous loan pricing function
                copyto!(V_S_p, variables.V_S)
                copyto!(V_C_p, variables.V_C)

                # update value functions
                value_function_singles!(V_S_p, variables, parameters; slow_updating = 1.0)

                value_function_couples!(V_C_p, variables, parameters; slow_updating = 1.0)

                # check convergence
                crit_W = max(norm(variables.V_S .- V_S_p, Inf), norm(variables.V_C .- V_C_p, Inf))

                # report preogress
                ProgressMeter.update!(prog_W, crit_W)

                # update the iteration number
                iter_W += 1

            end

            # compute payday price
            pricing_function!(variables, parameters; slow_updating = 1.0)

            # Compute stationary distribution
            # stationary_distribution!(μ_p, variables, parameters)

            # check convergence
            crit_q = max(norm(variables.q_S .- q_S_p, Inf), norm(variables.q_C .- q_C_p, Inf))

            # report preogress
            ProgressMeter.update!(prog_q, crit_q)

            # update the iteration number
            iter_q += 1
            println("")
        end
    end
end

parameters = parameters_function()
variables = variables_function(parameters; load_initial_value = true)
solve_function!(variables, parameters; tol = 1E-4, iter_max = 5000, one_loop = true)
stationary_distribution!(variables, parameters; tol = 1E-4, iter_max = 5000)

# check whether the sum of choice probability, given any individual state, equals one
# all(sum(variables.σ, dims=1) .≈ 1.0)

# save and load workspace
V_S = variables.V_S
# V_C = variables.V_C
n_S = variables.n_S
# n_C_1_i = variables.n_C_1_i
# n_C_2_i = variables.n_C_2_i
# l_S::Array{Float64,3}
# l_C_1::Array{Float64,5}
# l_C_2::Array{Float64,5}
a_S = variables.a_S
# a_C_i = variables.a_C_i
d_S_i = variables.d_S_i
# d_C_i = variables.d_C_i
P_S = variables.P_S
q_S = variables.q_S
# P_C = variables.P_C
# q_C = variables.q_C
μ_S = variables.μ_S
# μ_C = variables.μ_C

using JLD2
# @save "workspace.jld2" parameters V_S V_C n_S n_C_1_i n_C_2_i a_S a_C_i d_S_i d_C_i P_S q_S P_C q_C μ_S μ_C
@save "workspace.jld2" parameters V_S n_S a_S d_S_i P_S q_S μ_S

# cd("C:/Users/JanSun/Dropbox/Bankruptcy-Family/Results/7")
# @load "workspace.jld2"
