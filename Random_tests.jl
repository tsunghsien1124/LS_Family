# Check monotonicity of asset policy
# Singles
temp = 0
for κ_i in 1:parameters.κ_size, η_i in 1:parameters.η_size, z_i in 1:parameters.z_size, a_i in 2:parameters.a_size
    if (d_S_i[a_i-1,z_i,η_i,κ_i] == 1) # && (d_S_i[a_i,z_i,η_i,κ_i] == 1)
        temp += (a_S_i[a_i,z_i,η_i,κ_i].<a_S_i[a_i-1,z_i,η_i,κ_i])
        # if (a_S_i[a_i,z_i,η_i,κ_i].<a_S_i[a_i-1,z_i,η_i,κ_i]) == 1
        #     println(a_i,z_i,η_i,κ_i)
        # end
    end
end

# Couples
temp = 0
for κ_2_i in 1:parameters.κ_size, κ_1_i in 1:parameters.κ_size, η_2_i in 1:parameters.η_size, η_1_i in 1:parameters.η_size, z_2_i in 1:parameters.z_size, z_1_i in 1:parameters.z_size, a_i in 2:parameters.a_size
    if (d_C_i[a_i-1,z_1_i,z_2_i,η_1_i,η_2_i,κ_1_i,κ_2_i] == 1) # && (d_C_i[a_i,z_1_i,z_2_i,η_1_i,η_2_i,κ_1_i,κ_2_i] == 1)
        temp += (a_C_i[a_i,z_1_i,z_2_i,η_1_i,η_2_i,κ_1_i,κ_2_i].<a_C_i[a_i-1,z_1_i,z_2_i,η_1_i,η_2_i,κ_1_i,κ_2_i])
        # if (a_C_i[a_i,z_1_i,z_2_i,η_1_i,η_2_i,κ_1_i,κ_2_i].<a_C_i[a_i-1,z_1_i,z_2_i,η_1_i,η_2_i,κ_1_i,κ_2_i]) == 1
        #     println(parameters.a_grid[a_i])
        #     println(parameters.z_grid[z_1_i])
        #     println(parameters.z_grid[z_2_i])
        #     println(parameters.η_grid[η_1_i])
        #     println(parameters.η_grid[η_2_i])
        #     println(parameters.κ_grid[κ_1_i])
        #     println(parameters.κ_grid[κ_2_i])
        #     break
        # end
    end
end


# NLopt tests
f(x) = x[1]^2+x[2]^2
lower = [1.0,1.0]
upper = [2.0, 2.0]
results = optimize(f,lower,upper)
optimize(f)

rosenbrock(x) =  (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
result = optimize(rosenbrock, zeros(2), BFGS())

f(x) = x^2

function f_test(x::Vector,a)
    return x[1]^2+x[2]^2+a
end

f(x::Vector,grad::Vector) = f_test(x,2)

opt = Opt(:LN_COBYLA,2)
opt.max_objective = f
opt.lower_bounds = 1.0
opt.upper_bounds = 2.0
opt.xtol_rel = 1e-4

(optf,optx,ret) = optimize(opt,[1.5, 1.5])

function f_test(x::Vector,a)
    return x[1]^2+a
end

f(x::Vector,grad::Vector) = f_test(x,2)

opt = Opt(:LN_COBYLA,1)
opt.max_objective = f
opt.lower_bounds = 1.0
opt.upper_bounds = 2.0
opt.xtol_rel = 1e-4

(optf,optx,ret) = optimize(opt,[1.5])


f(n::Vector,grad::Vector) = Util_R_S_optim_function(n,1.0,1.0,0.0,1.0,0.0,0.0,0.0,parameters.θ,parameters.α,parameters.T,parameters.β,parameters.ρ)

opt = Opt(:LN_COBYLA,1)
opt.max_objective = f
opt.lower_bounds = 0.0
opt.upper_bounds = 1.0
opt.xtol_rel = 1e-2

(optf,optx,ret) = optimize(opt,[0.5])

for i in 1:3, ii in 1:5
    println(i)
        println(ii)
        if ii == 3
            break
        end
end


f(n::Vector,grad::Vector) = Util_R_S_optim_function(n,grad,1.0,1.0,-0.5,1.0,0.0,0.5,0.0,parameters.θ,parameters.α,parameters.T,parameters.β,parameters.ρ)

opt = Opt(:LD_MMA,1)
opt.max_objective = f
opt.lower_bounds = 0.0
opt.upper_bounds = 1.0
opt.xtol_rel = 1e-3

(optf,optx,ret) = optimize(opt,[0.0001])


function f_test(x::Vector,a)
    grad[1] = 2*x[1]
    return x[1]^2+a
end

f(x::Vector,grad::Vector) = f_test(x,2)

opt = Opt(:LN_COBYLA,1)
opt.max_objective = f
opt.lower_bounds = 1.0
opt.upper_bounds = 2.0
opt.xtol_rel = 1e-4

(optf,optx,ret) = optimize(opt,[1.5])


function f_test(x::Vector)
    if x[1] < 1.5
        return -(10^10)
    else
        return x[1]
    end
end

f(x::Vector,grad::Vector) = f_test(x)

opt = Opt(:LN_COBYLA,1)
opt.max_objective = f
opt.lower_bounds = 1.3
opt.upper_bounds = 2.0
opt.xtol_rel = 1e-4

(optf,optx,ret) = optimize(opt,[1.4])





function f_test(x::Vector,grad::Vector)
    if x[1] < 1.5
        grad[1] = 10^2
        return -Inf
    else
        grad[1] = 1
        return x[1]
    end
end

f(x::Vector,grad::Vector) = f_test(x,grad)

opt = Opt(:LD_LBFGS,1)
opt.max_objective = f
opt.lower_bounds = 1.0
opt.upper_bounds = 2.0
opt.xtol_rel = 1e-4

(optf,optx,ret) = optimize(opt,[1.01])



function f_test(x::Vector,grad::Vector)
    if (x[1] < 1.5) || (x[2] < 1.5)
        grad[1] = 10^10
        grad[2] = 10^10
        return -(10^10)
    else
        grad[1] = 1
        grad[2] = -1
        return x[1]-x[2]
    end
end

f(x::Vector,grad::Vector) = f_test(x,grad)

opt = Opt(:LD_LBFGS,2)
opt.max_objective = f
opt.lower_bounds = 1.0
opt.upper_bounds = 2.0
opt.xtol_rel = 1e-4

(optf,optx,ret) = optimize(opt,[1.6, 1.6])













z = parameters.z_grid[1]
η = parameters.η_grid[1]
a = parameters.a_grid[10]
q_S = 0.9375
a_p = -1.0
κ = parameters.κ_grid[1]
V_expect = 0.0
n_min = 0.001
n_initial = 0.5

f(n::Vector,grad::Vector) = Util_R_S_optim_function(n,grad,z,η,a,q_S,a_p,κ,V_expect,parameters.θ,parameters.α,parameters.T,parameters.β,parameters.ρ)

opt = Opt(:LD_MMA,1)
opt.max_objective = f
opt.lower_bounds = n_min
opt.upper_bounds = 1.0
opt.xtol_rel = 1e-4

(optf,optx,ret) = optimize(opt,[n_initial])

grad = zeros(101)
n_g =collect(0:0.01:1)

for i in 1:101
    n = n_g[i]
    grad[i] = (parameters.θ*z*η*(n^(parameters.θ-1.0))/(z*η*(n^parameters.θ)+a-q_S*a_p-κ))-(parameters.α/(parameters.T-n))
end

0.0^(parameters.θ-1)
