using Parameters
using LinearAlgebra

using ProgressMeter
# using NLopt
# using Optim
# using Plots
# using StatsFuns

println("Julia is running with $(Threads.nthreads()) threads...")

# include("GQ_algorithm.jl")

function parameters_function(;
    # β::Real = 0.9,                              # Discount factor, externally set to annual discount factor 0.96 (standard)
    # β::Real = 0.8847,
    β::Real = 0.9231,
    # β::Real = 0.830584,
    λ::Real = 0.5,                              # Utility weight for couples
    # r::Real = 0.04,
    r_s::Real = 0.106790788,                    # Savings rate, Exogenously calibrated from Livshits et al. (2007)
    # r_b::Real = 0.191016,                       # Borrowing rate, Exogenously calibrated from Livshits et al. (2007)
    r_b::Real = 0.13685,
    r_max::Real = 4.359375,                     # Maximum interest rate, Exogenously calibrated from Livshits et al. (2007)
    # ρ::Real = 0.975,                            # survival probability
    # ρ::Real = 0.8,
    a_size_neg::Integer = 101,
    # a_size_neg::Integer = 81,                   # number of assets
    # a_size_neg::Integer = 51,
    # a_size_neg::Integer = 31,                       # Baseline grid
    a_size_pos::Integer = 140,
    # a_size_pos::Integer = 100,
    # a_size_pos::Integer = 80,
    # a_size_pos::Integer = 50,                          # Baseline grid
    a_min::Real = -5.00,                        # minimum of assets
    # a_min::Real = -4.00,                               # Baseline grid
    a_max::Real = 13.00,                         # maximum of assets
    # a_max::Real = 10.00,                        # Baseline grid
    # a_degree::Real = 1.0,                      # governs how grid is spaced
    d_size::Integer = 2,                         # Number of default/repayment options
    # T::Real = 1.2,
    T::Real = 1.5,                              # Overall time endowment, exogenously calibrated from Alon et al. (2020)
    # α::Real = 0.5,                           # Weight of leisure in utility function # From Alon et al. (2020)
    # α::Real = 1.27,
    # ω::Real = 0.496,                            # Consumption weight, taken from Borella et al. (2018), needs to be internally calibrated
    ω::Real = 0.5631,
    γ_c::Real = 2.0,                             # CRRA parameter on consumption, Exogenously calibrated, standard value
    γ_l::Real = 3.0,                             # CRRA parameter on leisure
    # ϕ::Real = 0.9,                           # Wage garnishment
    ϕ::Real = 0.395,                                # Wage garnishment, internally calibrated
    # θ::Real = 0.55,                              # Returns to labor # From Alon et al. (2020)
    θ::Real = 1.0,
    ν::Real = 0.4,                               # Share of singles in economy
    # ζ::Real = 0.005,                             # Extreme value parameter
    # ζ::Real = 0.001,
    ζ::Real = 0.0001,
    # ψ::Real = 0.0,                              # Exogenous divorce probability
    ψ::Real = 0.033,                                # Own calibration
    lifespan::Integer = 16,
    # κ_div::Real = 0.0,
    # κ_div::Real = 0.04,
    # κ_div::Real = 0.0714,                       # Divorce cost, exogenoulsy calibrated from Livshits et al (2007), but still with $70k mean income
    κ_div::Real = 0.0818,                       # own calibration
    # ρ_s::Real = 1.4023                          # Economies of scale, exogenously calibrated in Voena (2015) from McClements scale
    ρ_s::Real = 1.0
    )
    """
    contruct an immutable object containing all parameters
    """

    # AEJ 2010 process
#     z_grid = [0.43951170247150,	0.68224440442049,	0.89325839105602,	1.16953770235719,	1.81544779969480]
#     z_size = length(z_grid)
#     Γ_z = [0.70073993281691	0.23546001015495	0.05603155653834	0.00752487973592 0.00024362075390;
# 0.23546001015494	0.41202882280749	0.26024386056642	0.08474242673522	0.00752487973592;
# 0.05603155653834	0.26024386056642	0.36744916579047	0.26024386056642	0.05603155653834;
# 0.00752487973592	0.08474242673522	0.26024386056642	0.41202882280749	0.23546001015494;
# 0.00024362075390	0.00752487973592	0.05603155653834	0.23546001015495	0.70073993281691]

    # De Nardi (final version)
    z_grid = zeros(5, 2)

    z_grid[:,1] = [0.4155809704471233, 0.6446556991504251, 1.0, 1.5512156354436546, 2.4062699476448612] # Women
    z_grid[:,2] = [0.33417655181070505, 0.5780800565758215, 1.0, 1.7298642093335026, 2.992430182733024] # Men
    # Double the standard deviation
    # z_grid[:,1] = [0.2888668145978083, 0.5374633146530173, 1.0, 1.8605921050548226, 3.4618029813923363] # Women
    # z_grid[:,2] = [0.21222686836997753, 0.4606808747603676, 1.0, 2.1707000546097555, 4.711938727082796] # Men
    # Triple the standard deviation
    # z_grid[:,1] = [0.21852051512973156, 0.4674617793250391, 1.0, 2.139212325430936, 4.576229373275633] # Women
    # z_grid[:,2] = [ 0.14979668426409884, 0.38703576613033946, 1.0, 2.58374054159955, 6.675715186305135] # Men
    # Quintuple the standard deviation
    # z_grid[:,1] = [0.1403744659551959, 0.374665805692481, 1.0, 2.66904527930361, 7.1238027029728865] # Women
    # z_grid[:,2] = [0.08621402432980975, 0.29362224767515444, 1.0, 3.40573647915923, 11.59904096547591] # Men
    # Six times the standard deviation
    # z_grid[:,1] = [0.1163860511451547, 0.34115399916336125, 1.0, 2.931227546657458, 8.592094930283501] # Women
    # z_grid[:,2] = [0.06823136029176047, 0.2612113326250614, 1.0, 3.828317822011896, 14.656017346333906] # Men
    # Seven times the standard deviation
    # z_grid[:,1] = [0.09796201139925671, 0.31298883590194826, 1.0, 3.1950021383934457, 10.208038664338691] # Women
    # z_grid[:,2] = [0.055024842442246154, 0.23457374627661587, 1.0, 4.263051666578119, 18.173609511914474] # Men
    # Ten times the standard deviation
    # z_grid[:,1] = [0.06224196432415489, 0.24948339488662344, 1.0, 4.008282797556308, 16.066330985185818] # Women
    # z_grid[:,2] = [0.03123775793273678, 0.17674206610973173, 1.0, 5.65796260059075, 32.01254078968364] # Men
    # Thirteen times the standard deviation
    # z_grid[:,1] = [0.042173864939848586, 0.20536276424865485, 1.0, 4.869431922864031, 23.711367251407296] # Women
    # z_grid[:,2] = [0.01921635281939909, 0.13862306020067183, 1.0, 7.213806985305274, 52.039011221239164] # Men
    # Fifteen times the standard deviation
    # z_grid[:,1] = [0.03334722459119466, 0.18261222464882973, 1.0, 5.476084648347273, 29.98750307586468] # Women
    # z_grid[:,2] = [0.014333982595171268, 0.1197246114847372, 1.0, 8.352501524947378, 69.76428172424829] # Men

    Γ_z = zeros(5,5,2)
    Γ_z[:,:,1] = [ 0.802665     0.181378     0.0153698  0.000578853  8.17523e-6;
                0.0453446    0.81035      0.136468   0.00769308   0.000144713;
                0.00256163   0.0909786    0.812919   0.0909786    0.00256163;
                0.000144713  0.00769308   0.136468   0.81035      0.0453446;
                8.17523e-6   0.000578853  0.0153698  0.181378     0.802665] # Women
    Γ_z[:,:,2] = [ 0.851414    0.139747     0.0086015  0.000235301  2.41382e-6;
                0.0349367   0.855715     0.104987   0.00430316   5.88253e-5;
                0.00143358  0.0699911    0.857151   0.0699911    0.00143358;
                5.88253e-5  0.00430316   0.104987   0.855715     0.0349367;
                2.41382e-6  0.000235301  0.0086015  0.139747     0.851414] # Men

    z_size = 5

    for gender in 1:2
        for i in 1:z_size
            sum_temp = sum(Γ_z[i,:,gender])
            for j in 1:z_size
                Γ_z[i,j,gender] = Γ_z[i,j,gender] / sum_temp
            end
        end
    end

    Γ_z_initial = [0.2, 0.2, 0.2, 0.2, 0.2]

    # Expense shocks
    # κ_grid = [0.0 0.264 0.8218]
    # # κ_grid = [0.0 0.5 1.6]
    # # κ_grid = [0.0 0.375 1.2]
    # Γ_κ = [(1.0-0.07104-0.0046), 0.07104, 0.0046]
    # κ_size = length(κ_grid)

    # Recalibrated based on AER 2007
    # κ_grid = [0.0 0.1181 0.4225]
    # Divide size by 2
    # κ_grid = [0.0 0.2108 0.7545]
    # Γ_κ = [(1.0-0.03374-0.0046), 0.03374, 0.0046]
    # Divide probs by 2
    # κ_grid = [0.0 0.2885 1.4637]
    # Γ_κ = [(1.0-0.02466-0.00237), 0.02466, 0.00237]
    # κ_size = length(κ_grid)

    # κ_grid = zeros(lifespan, 2, 3)
    # Γ_κ = [0.95, 0.03, 0.02]
    # κ_size = 3

    # Own MEPS expense shocks
    # Subset into 90th and 98th percentile
    # κ_grid = zeros(lifespan, 2, 3)
    # κ_grid[1,:,:] = [0.021422835 0.19294307 1.0315356
    #                 0.009614893 0.09464257 0.4320062]
    # κ_grid[2,:,:] = [0.021422835 0.19294307 1.0315356
    #                 0.009614893 0.09464257 0.4320062]
    #
    # κ_grid[3,:,:] = [0.01977526 0.1479777 0.3350743;
    # 0.01204864 0.1222332 0.5182777]
    # κ_grid[4,:,:] = [0.01977526 0.1479777 0.3350743;
    # 0.01204864 0.1222332 0.5182777]
    #
    # κ_grid[5,:,:] = [0.01747482 0.1387107 0.3802541;
    # 0.01405247 0.1294446 0.6139952]
    # κ_grid[6,:,:] = [0.01747482 0.1387107 0.3802541;
    # 0.01405247 0.1294446 0.6139952]
    #
    # κ_grid[7,:,:] = [0.01656116 0.1022988 0.3337965;
    # 0.01395420 0.1042894 0.4086041]
    # κ_grid[8,:,:] = [0.01656116 0.1022988 0.3337965;
    # 0.01395420 0.1042894 0.4086041]
    #
    # κ_grid[9,:,:] = [0.01376288 0.08569271 0.4867520;
    # 0.01697912 0.14074635 0.5639222]
    # κ_grid[10,:,:] = [0.01376288 0.08569271 0.4867520;
    # 0.01697912 0.14074635 0.5639222]
    #
    # κ_grid[11,:,:] = [0.02225225 0.1590459 0.5245908;
    # 0.02010363 0.1596131 1.2977394]
    # κ_grid[12,:,:] = [0.02225225 0.1590459 0.5245908;
    # 0.02010363 0.1596131 1.2977394]
    #
    # κ_grid[13,:,:] = [0.02688415 0.1790724 0.568906;
    # 0.02735435 0.2270097 1.165659]
    # κ_grid[14,:,:] = [0.02688415 0.1790724 0.568906;
    # 0.02735435 0.2270097 1.165659]
    #
    # κ_grid[15,:,:] = [0.02974287 0.1534460 0.3821857;
    # 0.02547362 0.1759906 0.8823126]
    # κ_grid[16,:,:] = [0.02974287 0.1534460 0.3821857;
    # 0.02547362 0.1759906 0.8823126]
    #
    # Γ_κ = [0.9, 0.08, 0.02]
    # κ_size = 3

    # Subset into 90th and 98th percentile, base is no shock
    # κ_grid = zeros(lifespan, 2, 3)
    # κ_grid[1,:,:] = [0 0.19294307 1.0315356
    #                 0 0.09464257 0.4320062]
    # κ_grid[2,:,:] = [0 0.19294307 1.0315356
    #                 0 0.09464257 0.4320062]
    #
    # κ_grid[3,:,:] = [0 0.1479777 0.3350743;
    # 0 0.1222332 0.5182777]
    # κ_grid[4,:,:] = [0 0.1479777 0.3350743;
    # 0 0.1222332 0.5182777]
    #
    # κ_grid[5,:,:] = [0 0.1387107 0.3802541;
    # 0 0.1294446 0.6139952]
    # κ_grid[6,:,:] = [0 0.1387107 0.3802541;
    # 0 0.1294446 0.6139952]
    #
    # κ_grid[7,:,:] = [0 0.1022988 0.3337965;
    # 0 0.1042894 0.4086041]
    # κ_grid[8,:,:] = [0 0.1022988 0.3337965;
    # 0 0.1042894 0.4086041]
    #
    # κ_grid[9,:,:] = [0 0.08569271 0.4867520;
    # 0 0.14074635 0.5639222]
    # κ_grid[10,:,:] = [0 0.08569271 0.4867520;
    # 0 0.14074635 0.5639222]
    #
    # κ_grid[11,:,:] = [0 0.1590459 0.5245908;
    # 0 0.1596131 1.2977394]
    # κ_grid[12,:,:] = [0 0.1590459 0.5245908;
    # 0 0.1596131 1.2977394]
    #
    # κ_grid[13,:,:] = [0 0.1790724 0.568906;
    # 0 0.2270097 1.165659]
    # κ_grid[14,:,:] = [0 0.1790724 0.568906;
    # 0 0.2270097 1.165659]
    #
    # κ_grid[15,:,:] = [0 0.1534460 0.3821857;
    # 0 0.1759906 0.8823126]
    # κ_grid[16,:,:] = [0 0.1534460 0.3821857;
    # 0 0.1759906 0.8823126]
    #
    # Γ_κ = [0.9, 0.08, 0.02]
    # κ_size = 3

    # Subset into 95th and 98th percentile, base is no shock (final version)
    κ_grid = zeros(lifespan, 2, 3)
    κ_grid[1,:,:] = [0 0.2478430 1.0315356;
                    0 0.1430821 0.4320062]
    κ_grid[2,:,:] = [0 0.2478430 1.0315356;
                    0 0.1430821 0.4320062]

    κ_grid[3,:,:] = [0 0.2005534 0.3350743;
    0 0.1696046 0.5182777]
    κ_grid[4,:,:] = [0 0.2005534 0.3350743;
    0 0.1696046 0.5182777]

    κ_grid[5,:,:] = [0 0.2045652 0.3802541;
    0 0.1796054 0.6139952]
    κ_grid[6,:,:] = [0 0.2045652 0.3802541;
    0 0.1796054 0.6139952]

    κ_grid[7,:,:] = [0 0.1333683 0.3337965;
    0 0.1509138 0.4086041]
    κ_grid[8,:,:] = [0 0.1333683 0.3337965;
    0 0.1509138 0.4086041]

    κ_grid[9,:,:] = [0 0.1135000 0.4867520;
    0 0.2006192 0.5639222]
    κ_grid[10,:,:] = [0 0.1135000 0.4867520;
    0 0.2006192 0.5639222]

    κ_grid[11,:,:] = [0 0.2215764 0.5245908;
    0 0.2364737 1.2977394]
    κ_grid[12,:,:] = [0 0.2215764 0.5245908;
    0 0.2364737 1.2977394]

    κ_grid[13,:,:] = [0 0.2369242 0.568906;
    0 0.3228740 1.165659]
    κ_grid[14,:,:] = [0 0.2369242 0.568906;
    0 0.3228740 1.165659]

    κ_grid[15,:,:] = [0 0.1922297 0.3821857;
    0 0.2315047 0.8823126]
    κ_grid[16,:,:] = [0 0.1922297 0.3821857;
    0 0.2315047 0.8823126]

    Γ_κ = [0.95, 0.03, 0.02]
    κ_size = 3

    # Subset into terciles (was still computed with Livshits et al. (2003) total charges adjustment)
    # κ_grid = zeros(lifespan, 2, 3)
    # κ_grid[1,:,:] = [0.0021021542 0.015140008 0.08591016;
    #                 0.0007336637 0.006571398 0.05369331]
    # κ_grid[2,:,:] = [0.0021021542 0.015140008 0.08591016;
    #                 0.0007336637 0.006571398 0.05369331]
    #
    # κ_grid[3,:,:] = [0.0020874607 0.015187949 0.07637816;
    #                 0.0006733213 0.007494999 0.06724905]
    # κ_grid[4,:,:] = [0.0020874607 0.015187949 0.07637816;
    #                 0.0006733213 0.007494999 0.06724905]
    #
    # κ_grid[5,:,:] = [0.001989376 0.01285704 0.07796850;
    #                 0.001269721 0.00907863 0.07351243]
    # κ_grid[6,:,:] = [0.001989376 0.01285704 0.07796850;
    #                 0.001269721 0.00907863 0.07351243]
    #
    # κ_grid[7,:,:] = [0.002078878 0.01257278 0.06691124;
    #                 0.001636876 0.01116755 0.05884386]
    # κ_grid[8,:,:] = [0.002078878 0.01257278 0.06691124;
    #                 0.001636876 0.01116755 0.05884386]
    #
    # κ_grid[9,:,:] = [0.002085603 0.01100013 0.06051587;
    #                 0.001828983 0.01206171 0.08207369]
    # κ_grid[10,:,:] = [0.002085603 0.01100013 0.06051587;
    #                 0.001828983 0.01206171 0.08207369]
    #
    # κ_grid[11,:,:] = [0.004101226 0.01960561 0.09228008;
    #                 0.002879350 0.01506069 0.10441197]
    # κ_grid[12,:,:] = [0.004101226 0.01960561 0.09228008;
    #                 0.002879350 0.01506069 0.10441197]
    #
    # κ_grid[13,:,:] = [0.004734507 0.02280651 0.1063080;
    #                 0.003936898 0.02084274 0.1405701]
    # κ_grid[14,:,:] = [0.004734507 0.02280651 0.1063080;
    #                 0.003936898 0.02084274 0.1405701]
    #
    # κ_grid[15,:,:] = [0.006271379 0.02550519 0.1001296;
    #                 0.004275191 0.02083723 0.1119401]
    # κ_grid[16,:,:] = [0.006271379 0.02550519 0.1001296;
    #                 0.004275191 0.02083723 0.1119401]
    #
    # Γ_κ = [1/3, 1/3, 1/3]
    # κ_size = 3

    # Lifecycle productivity
    h_grid = [0.774482122, 0.819574547, 0.873895492, 0.9318168, 0.986069673, 1.036889326, 1.082870993, 1.121249981, 1.148476948, 1.161069822, 1.156650443, 1.134940682, 1.09844343, 1.05261516, 1.005569967, 0.9519]

    # fixed_cost_working_grid = []

    # Equivalence scales
    # es_grid = [1.264, 1.343, 1.436, 1.533, 1.614, 1.661, 1.671, 1.657, 1.621, 1.572, 1.514, 1.468, 1.417, 1.37, 1.327, 1.2910]

    es_grid = ones(lifespan)

    # Asset grid
    a_grid_neg = collect(range(a_min, 0.0, length=a_size_neg))
    a_grid_pos = collect(range(0.0, a_max, length=a_size_pos))
    a_grid = cat(a_grid_neg[1:(end-1)], a_grid_pos, dims = 1)

    # a_grid_neg = reverse((exp.(collect(range(log(1.0), stop=log(-a_min+1), length = a_size_neg))).-1)*(-1))
    # a_grid_pos = exp.(collect(range(log(1.0), stop=log(a_max+1), length = a_size_pos))).-1
    # a_grid = cat(a_grid_neg[1:(end-1)], a_grid_pos, dims = 1)

    a_size_neg = length(a_grid_neg)
    a_size = length(a_grid)
    a_ind_zero = findall(iszero, a_grid)[]

    # BM_indices = BM_function(a_size)

    # Labor
    n_grid = collect(range(0.0, 1.0, length=3))
    n_size = length(n_grid)

    # Asset distribution for newborns
    # Start life with debt
    # μ_asset_newborns = zeros(a_size)
    # μ_asset_newborns[45:a_ind_zero-1] = collect(range(0.0,0.1,length=(a_ind_zero-45)))
    # μ_asset_newborns[a_ind_zero] = 1.0 - sum(μ_asset_newborns)

    # Start life with savings
    μ_asset_newborns = zeros(a_size)
    μ_asset_newborns[a_ind_zero+1:a_ind_zero+30] = collect(range(0.05,0.0,length=30))
    μ_asset_newborns[a_ind_zero] = 1.0 - sum(μ_asset_newborns)

    # numerically relevant lower bound
    L_ζ = ζ*log(eps(Float64))

    # return the outcome
    return (β=β, λ=λ, r_s=r_s, r_b=r_b, r_max=r_max, T=T, ω=ω, γ_c=γ_c, γ_l=γ_l, ϕ=ϕ, θ=θ, ν=ν, κ_grid = κ_grid, Γ_κ = Γ_κ, κ_size = κ_size,
    a_grid=a_grid, a_size=a_size, a_ind_zero=a_ind_zero,
    z_grid=z_grid, z_size=z_size, Γ_z=Γ_z, Γ_z_initial=Γ_z_initial, d_size=d_size, ζ=ζ, L_ζ=L_ζ, ψ=ψ, μ_asset_newborns=μ_asset_newborns, lifespan = lifespan, h_grid=h_grid, es_grid=es_grid, κ_div=κ_div, ρ_s=ρ_s, n_grid=n_grid, n_size = n_size)
end

mutable struct MutableVariables
    """
    construct a type for mutable variables
    """
    V_S::Array{Float64,5}
    a_S_i::Array{Int64,5}
    n_S_i::Array{Int64,5}
    d_S_i::Array{Int64,5}
    V_div::Array{Float64,5}
    a_div_i::Array{Int64,5}
    n_div_i::Array{Int64,5}
    d_div_i::Array{Int64,5}
    V_C::Array{Float64,6}
    a_C_i::Array{Int64,6}
    n_C_i::Array{Int64,7}
    d_C_i::Array{Int64,6}
    P_S::Array{Float64,4}
    q_S::Array{Float64,4}
    P_C::Array{Float64,4}
    q_C::Array{Float64,4}
end

function variables_function(
    parameters::NamedTuple;
    load_initial_value::Bool = false
    )
    """
    construct a mutable object containing endogenous variables
    """

    # unpack parameters
    @unpack  a_size, z_size, κ_size, d_size, r_s, r_b, ν, lifespan, a_ind_zero, n_size = parameters

    # Singles' value functions
    V_S = zeros(a_size, z_size, κ_size, lifespan, 2)
    V_S .= -Inf

    # Singles' policy functions
    # Assets
    a_S_i = zeros(Int64, a_size, z_size, κ_size, lifespan, 2)

    # Labor
    n_S_i = ones(Int64, a_size, z_size, κ_size, lifespan, 2)

    # Default
    d_S_i = zeros(Int64, a_size, z_size, κ_size, lifespan, 2)

    # Divorced value functions
    V_div = zeros(a_size, z_size, κ_size, lifespan, 2)
    V_div .= -Inf

    # Divorced' policy functions
    # Assets
    a_div_i = zeros(Int64, a_size, z_size, κ_size, lifespan, 2)

    # Labor
    n_div_i = zeros(Int64, a_size, z_size, κ_size, lifespan, 2)

    # Default
    d_div_i = zeros(Int64, a_size, z_size, κ_size, lifespan, 2)

    # Couples' value functions
    V_C = zeros(a_size, z_size, z_size, κ_size, κ_size, lifespan)
    V_C .= -Inf

    # Couples' policy functions
    # Assets
    a_C_i = zeros(Int64, a_size, z_size, z_size, κ_size, κ_size, lifespan)

    # labor
    n_C_i = zeros(Int64, a_size, z_size, z_size, κ_size, κ_size, lifespan, 2)

    # Default
    d_C_i = zeros(Int64, a_size, z_size, z_size, κ_size, κ_size, lifespan)

    # Loan pricing
    # Singles
    P_S = ones(a_size, z_size, lifespan, 2)
    q_S = ones(a_size, z_size, lifespan, 2)
    q_S[1:(a_ind_zero-1), :, :, :] .= 1.0/(1.0 + r_b)
    q_S[a_ind_zero:end, :, :, :] .= 1.0/(1.0 + r_s)

    # Couples
    P_C = ones(a_size, z_size, z_size, lifespan)
    q_C = ones(a_size, z_size, z_size, lifespan)
    q_C[1:(a_ind_zero-1), :, :, :] .= 1.0/(1.0 + r_b)
    q_C[a_ind_zero:end, :, :, :] .= 1.0/(1.0 + r_s)

    # return the outcome
    variables = MutableVariables(V_S,a_S_i,n_S_i,d_S_i,V_div,a_div_i,n_div_i,d_div_i,V_C,a_C_i,n_C_i,d_C_i,P_S,q_S,P_C,q_C)
    return variables
end

function utility_function(
    c::Real,
    l::Real
    )
    """
    compute utility of CRRA utility function
    """
    @unpack ω, γ_c = parameters
    if (c > 0.0) && (l > 0.0)
        # return (c^(1.0-γ_c))/(1.0-γ_c)
        return (((c^ω)*(l^(1-ω)))^(1.0-γ_c))/(1.0-γ_c)
    else
        return -Inf
    end
end

function value_function_singles!(
    age::Integer,
    gender::Integer,
    variables::MutableVariables,
    parameters::NamedTuple;
    slow_updating::Real = 1.0
    )
    """
    compute feasible set and (un)conditional value functions
    """

    @unpack  a_size, z_size, κ_size, a_grid, z_grid, κ_grid, T, θ, β, Γ_z, Γ_κ, ϕ, a_ind_zero, ζ, L_ζ, lifespan, h_grid, es_grid, n_grid, n_size = parameters

    if age != lifespan
        V_expect_mat = reshape(variables.V_S[:,:,:,age+1,gender],a_size,:)*transpose(kron(reshape(Γ_κ,1,:),Γ_z[:,:,gender]))
    end

    # Repayment
    Threads.@threads for i in CartesianIndices((1:z_size,1:κ_size))

        @inbounds z_i = i[1]
        @inbounds κ_i = i[2]

        @inbounds z = z_grid[z_i,gender]
        # @inbounds κ = κ_grid[age,2,κ_i] # 2 is for singles
        @inbounds κ = κ_grid[age,1,κ_i] # 1 is for couples

        for a_i in 1:a_size

            @inbounds a = a_grid[a_i]

            for n_i in 1:n_size, a_p_i in 1:a_size

                n = n_grid[n_i]
                l = T - n

                @inbounds a_p = a_grid[a_p_i]

                @inbounds c = z*h_grid[age]*n+a-variables.q_S[a_p_i,z_i,age,gender]*a_p-κ

                if (c <= 0.0) || (variables.q_S[a_p_i,z_i,age,gender] ≈ 0.0)
                    @inbounds temp = -Inf
                else
                    if age == lifespan
                        if a_p_i != a_ind_zero
                            temp = -Inf
                        else
                            @inbounds temp = utility_function(c,l)
                        end
                    else
                        @inbounds temp = utility_function(c,l) + β*V_expect_mat[a_p_i,z_i]
                    end
                end

                if temp > variables.V_S[a_i,z_i,κ_i,age,gender]
                    @inbounds variables.V_S[a_i,z_i,κ_i,age,gender] = temp
                    @inbounds variables.d_S_i[a_i,z_i,κ_i,age,gender] = 1 # Repayment
                    @inbounds variables.a_S_i[a_i,z_i,κ_i,age,gender] = a_p_i
                    @inbounds variables.n_S_i[a_i,z_i,κ_i,age,gender] = n_i
                end
            end
        end
    end

    # Default
    Threads.@threads for z_i in 1:z_size

        @inbounds z = z_grid[z_i, gender]

        # Formal default
        for n_i in 2:n_size
            n = n_grid[n_i]
            l = T - n
            c = (z*h_grid[age]*n)*(1.0-ϕ)
            # c = (z*h_grid[age]*n) - ϕ

            if (c <= 0.0)
                temp = -Inf
            else
                if age == lifespan
                    @inbounds temp = utility_function(c,l)
                else
                    @inbounds temp = utility_function(c,l) + β*V_expect_mat[a_ind_zero,z_i]
                end
            end

            for κ_i in 1:κ_size, a_i in 1:a_size
                if temp > variables.V_S[a_i,z_i,κ_i,age,gender]
                    @inbounds variables.V_S[a_i,z_i,κ_i,age,gender] = temp
                    @inbounds variables.d_S_i[a_i,z_i,κ_i,age,gender] = 2 # Default
                    @inbounds variables.a_S_i[a_i,z_i,κ_i,age,gender] = a_ind_zero
                    @inbounds variables.n_S_i[a_i,z_i,κ_i,age,gender] = n_i
                end
            end
        end
    end
end

function value_function_divorced!(
    age::Integer,
    gender::Integer,
    variables::MutableVariables,
    parameters::NamedTuple;
    slow_updating::Real = 1.0
    )
    """
    compute feasible set and (un)conditional value functions
    """

    @unpack  a_size, z_size, κ_size, a_grid, z_grid, κ_grid, T, θ, β, Γ_z, Γ_κ, ϕ, a_ind_zero, ζ, L_ζ, κ_div, lifespan, h_grid, es_grid, n_grid, n_size = parameters

    if age != lifespan
        V_expect_mat = reshape(variables.V_S[:,:,:,age+1,gender],a_size,:)*transpose(kron(reshape(Γ_κ,1,:),Γ_z[:,:,gender]))
    end

    # Repayment
    Threads.@threads for i in CartesianIndices((1:z_size,1:κ_size))

        @inbounds z_i = i[1]
        @inbounds κ_i = i[2]

        @inbounds z = z_grid[z_i,gender]
        # @inbounds κ = κ_grid[age,2,κ_i] # 2 is for singles
        @inbounds κ = κ_grid[age,1,κ_i] # 1 is for couples

        for a_i in 1:a_size

            @inbounds a = a_grid[a_i]

            for n_i in 1:n_size, a_p_i in 1:a_size

                n = n_grid[n_i]
                l = T - n

                @inbounds a_p = a_grid[a_p_i]

                @inbounds c = z*h_grid[age]*n+a-variables.q_S[a_p_i,z_i,age,gender]*a_p-κ-κ_div

                if (c <= 0.0) || (variables.q_S[a_p_i,z_i,age,gender] ≈ 0.0)
                    @inbounds temp = -Inf
                else
                    if age == lifespan
                        if a_p_i != a_ind_zero
                            temp = -Inf
                        else
                            @inbounds temp = utility_function(c,l)
                        end
                    else
                        @inbounds temp = utility_function(c,l) + β*V_expect_mat[a_p_i,z_i]
                    end
                end

                if temp > variables.V_div[a_i,z_i,κ_i,age,gender]
                    @inbounds variables.V_div[a_i,z_i,κ_i,age,gender] = temp
                    @inbounds variables.d_div_i[a_i,z_i,κ_i,age,gender] = 1 # Repayment
                    @inbounds variables.a_div_i[a_i,z_i,κ_i,age,gender] = a_p_i
                    @inbounds variables.n_div_i[a_i,z_i,κ_i,age,gender] = n_i
                end
            end
        end
    end

    # Default
    Threads.@threads for z_i in 1:z_size

        @inbounds z = z_grid[z_i, gender]

        # Formal default
        for n_i in 2:n_size
            n = n_grid[n_i]
            l = T - n
            c = (z*h_grid[age]*n)*(1.0-ϕ)
            # c = (z*h_grid[age]*n) - ϕ

            if (c <= 0.0)
                temp = -Inf
            else
                if age == lifespan
                    @inbounds temp = utility_function(c,l)
                else
                    @inbounds temp = utility_function(c,l) + β*V_expect_mat[a_ind_zero,z_i]
                end
            end

            for κ_i in 1:κ_size, a_i in 1:a_size
                if temp > variables.V_div[a_i,z_i,κ_i,age,gender]
                    @inbounds variables.V_div[a_i,z_i,κ_i,age,gender] = temp
                    @inbounds variables.d_div_i[a_i,z_i,κ_i,age,gender] = 2 # Default
                    @inbounds variables.a_div_i[a_i,z_i,κ_i,age,gender] = a_ind_zero
                    @inbounds variables.n_div_i[a_i,z_i,κ_i,age,gender] = n_i
                end
            end
        end
    end
end

# Value function for couples
function value_function_couples!(
    age::Integer,
    variables::MutableVariables,
    parameters::NamedTuple;
    slow_updating::Real = 1.0
    )
    """
    compute feasible set and (un)conditional value functions
    """

    @unpack  a_size, z_size, κ_size, a_grid, z_grid, κ_grid, T, θ, β, Γ_z, Γ_κ, ϕ, a_ind_zero, λ, ζ, L_ζ, ψ, lifespan, h_grid, es_grid, ρ_s, n_grid, n_size = parameters

    if age != lifespan
        V_expect_C_mat = reshape(variables.V_C[:,:,:,:,:,age+1],a_size,:)*transpose(kron(reshape(parameters.Γ_κ,1,:),kron(reshape(parameters.Γ_κ,1,:),kron(parameters.Γ_z[:,:,2],parameters.Γ_z[:,:,1]))))

        V_expect_div_mat_women = reshape(variables.V_div[:,:,:,age+1,1],a_size,:)*transpose(kron(reshape(parameters.Γ_κ,1,:),parameters.Γ_z[:,:,1]))

        V_expect_div_mat_men = reshape(variables.V_div[:,:,:,age+1,2],a_size,:)*transpose(kron(reshape(parameters.Γ_κ,1,:),parameters.Γ_z[:,:,2]))
    end

    # Repayment
    Threads.@threads for i in CartesianIndices((1:z_size,1:κ_size))
        for κ_2_i in 1:κ_size, z_2_i in 1:z_size

            @inbounds z_1_i = i[1]
            @inbounds κ_1_i = i[2]

            @inbounds z_1 = z_grid[z_1_i,1]
            @inbounds z_2 = z_grid[z_2_i,2]
            @inbounds κ_1 = κ_grid[age,1,κ_1_i] # 1 is for couples
            @inbounds κ_2 = κ_grid[age,1,κ_2_i] # 1 is for couples
            # @inbounds κ_1 = κ_grid[age,2,κ_1_i] # 2 is for singles
            # @inbounds κ_2 = κ_grid[age,2,κ_2_i] # 2 is for singles
            # @inbounds κ_2 = 0.0
            # @inbounds κ_2 = κ_1 # Perfectly correlated expenses within couples

            @inbounds z_i = LinearIndices(Γ_z[:,:,1])[z_1_i,z_2_i]

            for a_i in 1:a_size

                @inbounds a = a_grid[a_i]

                for n_2_i in 1:n_size, n_1_i in 1:n_size, a_p_i in 1:a_size
                    n_1 = n_grid[n_1_i]
                    l_1 = T - n_1

                    n_2 = n_grid[n_2_i]
                    l_2 = T - n_2

                    @inbounds a_p = a_grid[a_p_i]

                    @inbounds x = z_1*h_grid[age]*n_1+z_2*h_grid[age]*n_2+a-variables.q_C[a_p_i,z_1_i,z_2_i,age]*a_p-κ_1-κ_2

                    # With economies of scale
                    c = x/(2^(1/ρ_s))
                    # Without economies of scale
                    # c = x/2

                    if (c <= 0.0) || (variables.q_C[a_p_i,z_1_i,z_2_i,age] ≈ 0.0)
                        @inbounds temp = -Inf
                    else
                        if age == lifespan
                            if a_p_i != a_ind_zero
                                temp = -Inf
                            else
                                # @inbounds temp = λ*utility_function(c/es_grid[age],l_1) + (1-λ)*utility_function(c/es_grid[age],l_2)
                                @inbounds temp = utility_function(c/es_grid[age],l_1) + utility_function(c/es_grid[age],l_2)
                            end
                        else

                            if a_p/2 in a_grid
                                a_p_i_half_1 = findfirst(a_grid .== a_p/2)
                                a_p_i_half_2 = a_p_i_half_1
                            else
                                a_p_i_half_1 = findlast(a_grid .< a_p/2)
                                a_p_i_half_2 = findfirst(a_grid .> a_p/2)
                            end

                            # @inbounds temp = λ*utility_function(c/es_grid[age],l_1) + (1-λ)*utility_function(c/es_grid[age],l_2)+ β*(1-ψ)*V_expect_C_mat[a_p_i,z_i] + β*ψ*(λ*V_expect_div_mat_women[a_p_i_half_1, z_1_i] + (1-λ)*V_expect_div_mat_men[a_p_i_half_2, z_2_i])
                            @inbounds temp = utility_function(c/es_grid[age],l_1) + utility_function(c/es_grid[age],l_2)+ β*(1-ψ)*V_expect_C_mat[a_p_i,z_i] + β*ψ*(V_expect_div_mat_women[a_p_i_half_1, z_1_i] + V_expect_div_mat_men[a_p_i_half_2, z_2_i])
                        end
                    end

                    if temp > variables.V_C[a_i,z_1_i,z_2_i,κ_1_i,κ_2_i,age]
                        @inbounds variables.V_C[a_i,z_1_i,z_2_i,κ_1_i,κ_2_i,age] = temp
                        @inbounds variables.d_C_i[a_i,z_1_i,z_2_i,κ_1_i,κ_2_i,age] = 1 # Repayment
                        @inbounds variables.a_C_i[a_i,z_1_i,z_2_i,κ_1_i,κ_2_i,age] = a_p_i
                        @inbounds variables.n_C_i[a_i,z_1_i,z_2_i,κ_1_i,κ_2_i,age,1] = n_1_i
                        @inbounds variables.n_C_i[a_i,z_1_i,z_2_i,κ_1_i,κ_2_i,age,2] = n_2_i
                    end

                end
            end
        end
    end

    # Default
    Threads.@threads for i in CartesianIndices((1:z_size,1:z_size))

        @inbounds z_1_i = i[1]
        @inbounds z_2_i = i[2]

        @inbounds z_1 = z_grid[z_1_i,1]
        @inbounds z_2 = z_grid[z_2_i,2]

        @inbounds z_i = LinearIndices(Γ_z[:,:,1])[z_1_i,z_2_i]

        for n_2_i in 1:n_size, n_1_i in 1:n_size
            n_1 = n_grid[n_1_i]
            l_1 = T - n_1

            n_2 = n_grid[n_2_i]
            l_2 = T - n_2

            x = (z_1*h_grid[age]*n_1+z_2*h_grid[age]*n_2)*(1.0-ϕ)
            # x = (z_1*h_grid[age]*n_1+z_2*h_grid[age]*n_2) - ϕ

            # With economies of scale
            c = x/(2^(1/ρ_s))
            # Without economies of scale
            # c = x/2

            if (c <= 0.0)
                temp = -Inf
            else
                if age == lifespan
                    # temp = λ*utility_function(c/es_grid[age],l_1) + (1-λ)*utility_function(c/es_grid[age],l_2)
                    temp = utility_function(c/es_grid[age],l_1) + utility_function(c/es_grid[age],l_2)
                else
                    # @inbounds temp = λ*utility_function(c/es_grid[age],l_1) + (1-λ)*utility_function(c/es_grid[age],l_2)+ β*(1-ψ)*V_expect_C_mat[a_ind_zero,z_i] + β*ψ*(λ*V_expect_div_mat_women[a_ind_zero, z_1_i] + (1-λ)*V_expect_div_mat_men[a_ind_zero, z_2_i])
                    @inbounds temp = utility_function(c/es_grid[age],l_1) + utility_function(c/es_grid[age],l_2)+ β*(1-ψ)*V_expect_C_mat[a_ind_zero,z_i] + β*ψ*(V_expect_div_mat_women[a_ind_zero, z_1_i] + V_expect_div_mat_men[a_ind_zero, z_2_i])
                end
            end

            for κ_2_i in 1:κ_size, κ_1_i in 1:κ_size, a_i in 1:a_size
                if temp > variables.V_C[a_i,z_1_i,z_2_i,κ_1_i,κ_2_i,age]
                    @inbounds variables.V_C[a_i,z_1_i,z_2_i,κ_1_i,κ_2_i,age] = temp
                    @inbounds variables.d_C_i[a_i,z_1_i,z_2_i,κ_1_i,κ_2_i,age] = 2 # Default
                    @inbounds variables.a_C_i[a_i,z_1_i,z_2_i,κ_1_i,κ_2_i,age] = a_ind_zero
                    @inbounds variables.n_C_i[a_i,z_1_i,z_2_i,κ_1_i,κ_2_i,age,1] = n_1_i
                    @inbounds variables.n_C_i[a_i,z_1_i,z_2_i,κ_1_i,κ_2_i,age,2] = n_2_i
                end
            end
        end
    end
end

function pricing_function!(
    age::Integer,
    variables::MutableVariables,
    parameters::NamedTuple;
    slow_updating::Real = 1.0
    )

    @unpack  a_size, z_size, z_grid, κ_grid, κ_size, a_grid, r_b, Γ_z, Γ_κ, a_ind_zero, ψ, lifespan, h_grid, r_max, κ_div, n_grid, n_size = parameters

    # Singles
    # Women
    if age == lifespan
        @inbounds variables.q_S[1:a_ind_zero-1,:,age, :] .= 0.0
    else
        Threads.@threads for gender in 1:2
            for z_i in 1:z_size, a_p_i in 1:(a_ind_zero-1)

                P_expect = 0.0

                for κ_p_i in 1:κ_size, z_p_i in 1:z_size

                    if variables.d_S_i[a_p_i,z_p_i,κ_p_i,age+1,gender] == 2 # Default

                        n = n_grid[variables.n_S_i[a_p_i,z_p_i,κ_p_i,age+1,gender]]

                        # recovery = ((z_grid[z_p_i,gender]*h_grid[age+1]*n)*parameters.ϕ)/(abs(a_grid[a_p_i])+κ_grid[age+1,2,κ_p_i]) # 2 is for singles
                        # recovery = (parameters.ϕ)/(abs(a_grid[a_p_i])+κ_grid[age+1,2,κ_p_i]) # 2 is for singles
                        recovery = ((z_grid[z_p_i,gender]*h_grid[age+1]*n)*parameters.ϕ)/(abs(a_grid[a_p_i])+κ_grid[age+1,1,κ_p_i]) # 1 is for couples

                        if recovery > 1.0
                            recovery = 1.0
                        end

                        P_expect += Γ_z[z_i,z_p_i,gender]*Γ_κ[κ_p_i]*recovery
                    else
                        P_expect += Γ_z[z_i,z_p_i,gender]*Γ_κ[κ_p_i]
                    end
                end

                @inbounds variables.P_S[a_p_i,z_i,age,gender] = P_expect

                @inbounds variables.q_S[a_p_i,z_i,age,gender] = P_expect/(1+r_b)

                # Interest rate ceiling
                if variables.q_S[a_p_i,z_i,age,gender] < 1/(1+r_max)
                    variables.q_S[a_p_i,z_i,age,gender] = 0.0
                end
            end
        end
    end

    # Couples
    if age == lifespan
        @inbounds variables.q_C[1:a_ind_zero-1,:,:,age] .= 0.0
    else
        Threads.@threads for a_p_i in 1:(a_ind_zero-1)

            a_p = a_grid[a_p_i]

            if a_p/2 in a_grid
                a_p_i_half_1 = findfirst(a_grid .== a_p/2)
                a_p_i_half_2 = a_p_i_half_1
            else
                a_p_i_half_1 = findlast(a_grid .< a_p/2)
                a_p_i_half_2 = findfirst(a_grid .> a_p/2)
            end

            a_p_half_1 = a_grid[a_p_i_half_1]
            a_p_half_2 = a_grid[a_p_i_half_2]

            for z_2_i in 1:z_size, z_1_i in 1:z_size
                P_expect_remain = 0.0
                P_expect_div_1 = 0.0
                P_expect_div_2 = 0.0

                # Remain married
                for κ_2_p_i in 1:κ_size, κ_1_p_i in 1:κ_size, z_2_p_i in 1:z_size, z_1_p_i in 1:z_size

                    if variables.d_C_i[a_p_i,z_1_p_i,z_2_p_i,κ_1_p_i,κ_2_p_i,age+1] == 2 # Default
                            n_1 = n_grid[variables.n_C_i[a_p_i,z_1_p_i,z_2_p_i,κ_1_p_i,κ_2_p_i,age+1,1]]
                            n_2 = n_grid[variables.n_C_i[a_p_i,z_1_p_i,z_2_p_i,κ_1_p_i,κ_2_p_i,age+1,2]]

                            recovery = ((z_grid[z_1_p_i,1]*h_grid[age+1]*n_1+z_grid[z_2_p_i,2]*h_grid[age+1]*n_2)*parameters.ϕ)/(abs(a_grid[a_p_i])+κ_grid[age+1,1,κ_1_p_i]+κ_grid[age+1,1,κ_2_p_i]) # 1 is for couples
                            # recovery = (parameters.ϕ)/(abs(a_grid[a_p_i])+κ_grid[age+1,1,κ_1_p_i]+κ_grid[age+1,1,κ_2_p_i]) # 1 is for couples, fixed bankruptcy cost
                            # recovery = ((z_grid[z_1_p_i,1]*h_grid[age+1]*n_1+z_grid[z_2_p_i,2]*h_grid[age+1]*n_2)*parameters.ϕ)/(abs(a_grid[a_p_i])+κ_grid[age+1,2,κ_1_p_i]+κ_grid[age+1,2,κ_2_p_i]) # 2 is for singles
                            # recovery = ((z_grid[z_1_p_i,1]*h_grid[age+1]*n_1+z_grid[z_2_p_i,2]*h_grid[age+1]*n_2)*parameters.ϕ)/(abs(a_grid[a_p_i])+κ_grid[age+1,2,κ_1_p_i]) # 2 is for singles, only one singles shock
                            # recovery = ((z_grid[z_1_p_i,1]*h_grid[age+1]*n_1+z_grid[z_2_p_i,2]*h_grid[age+1]*n_2)*parameters.ϕ)/(abs(a_grid[a_p_i])+κ_grid[age+1,1,κ_1_p_i]+κ_grid[age+1,1,κ_1_p_i]) # 1 is for couples, two perfectly correlated shocks

                            if recovery > 1.0
                                recovery = 1.0
                            end

                            P_expect_remain += Γ_z[z_1_i,z_1_p_i,1]*Γ_z[z_2_i,z_2_p_i,2]*Γ_κ[κ_1_p_i]*Γ_κ[κ_2_p_i]*recovery
                    else
                            P_expect_remain += Γ_z[z_1_i,z_1_p_i,1]*Γ_z[z_2_i,z_2_p_i,2]*Γ_κ[κ_1_p_i]*Γ_κ[κ_2_p_i]
                    end
                end

                # Divorced female
                for κ_1_p_i in 1:κ_size, z_1_p_i in 1:z_size

                    if variables.d_div_i[a_p_i_half_1,z_1_p_i,κ_1_p_i,age+1,1] == 2 # Default
                        n = n_grid[variables.n_div_i[a_p_i_half_1,z_1_p_i,κ_1_p_i,age+1,1]]

                        # recovery = ((z_grid[z_1_p_i,1]*h_grid[age+1]*n)*parameters.ϕ)/(abs(a_grid[a_p_i_half_1])+κ_grid[age+1,2,κ_1_p_i]+κ_div) # 2 is for singles
                        # recovery = (parameters.ϕ)/(abs(a_grid[a_p_i_half_1])+κ_grid[age+1,2,κ_1_p_i]+κ_div) # 2 is for singles
                        recovery = ((z_grid[z_1_p_i,1]*h_grid[age+1]*n)*parameters.ϕ)/(abs(a_grid[a_p_i_half_1])+κ_grid[age+1,1,κ_1_p_i]+κ_div) # 1 is for couples

                        if recovery > 1.0
                            recovery = 1.0
                        end

                        P_expect_div_1 += Γ_z[z_1_i,z_1_p_i,1]*Γ_κ[κ_1_p_i]*recovery

                    else
                        P_expect_div_1 += Γ_z[z_1_i,z_1_p_i,1]*Γ_κ[κ_1_p_i]
                    end
                end

                # Divorced male
                for κ_2_p_i in 1:κ_size, z_2_p_i in 1:z_size

                    if variables.d_div_i[a_p_i_half_2,z_2_p_i,κ_2_p_i,age+1,2] == 2 # Default
                        n = n_grid[variables.n_div_i[a_p_i_half_2,z_2_p_i,κ_2_p_i,age+1,2]]

                        # recovery = ((z_grid[z_2_p_i,2]*h_grid[age+1]*n)*parameters.ϕ)/(abs(a_grid[a_p_i_half_2])+κ_grid[age+1,2,κ_2_p_i]+κ_div) # 2 is for singles
                        # recovery = (parameters.ϕ)/(abs(a_grid[a_p_i_half_2])+κ_grid[age+1,2,κ_2_p_i]+κ_div) # 2 is for singles
                        recovery = ((z_grid[z_2_p_i,2]*h_grid[age+1]*n)*parameters.ϕ)/(abs(a_grid[a_p_i_half_2])+κ_grid[age+1,1,κ_2_p_i]+κ_div) # 1 is for couples

                        if recovery > 1.0
                            recovery = 1.0
                        end

                        P_expect_div_2 += Γ_z[z_2_i,z_2_p_i,2]*Γ_κ[κ_2_p_i]*recovery
                    else
                        P_expect_div_2 += Γ_z[z_2_i,z_2_p_i,2]*Γ_κ[κ_2_p_i]
                    end
                end

                @inbounds variables.P_C[a_p_i,z_1_i,z_2_i,age] = (1.0-ψ)*P_expect_remain + ψ*((a_p_half_1/a_p)*P_expect_div_1 + (a_p_half_2/a_p)*P_expect_div_2)

                @inbounds variables.q_C[a_p_i,z_1_i,z_2_i,age] = variables.P_C[a_p_i,z_1_i,z_2_i,age]/(1+r_b)

                # Interest rate ceiling
                if variables.q_C[a_p_i,z_1_i,z_2_i,age] < 1/(1+r_max)
                    variables.q_C[a_p_i,z_1_i,z_2_i,age] = 0.0
                end
            end
        end
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
        for age in parameters.lifespan:-1:1

            # compute loan pricing
            pricing_function!(age, variables, parameters; slow_updating = 1.0)

            # update value functions
            for gender in 1:2
                value_function_singles!(age, gender, variables, parameters; slow_updating = 1.0)

                value_function_divorced!(age, gender, variables, parameters; slow_updating = 1.0)
            end

            value_function_couples!(age, variables, parameters; slow_updating = 1.0)

            # check convergence
            # crit = max(norm(variables.W_S .- W_S_p, Inf), norm(variables.W_C .- W_C_p, Inf), norm(variables.q_S .- q_S_p, Inf), norm(variables.q_C .- q_C_p, Inf))
            # crit = max(norm(variables.W_S .- W_S_p, Inf), norm(variables.q_S .- q_S_p, Inf))

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
            # ProgressMeter.update!(prog, crit)

            # println("V_S_crit = $V_S_crit at $V_S_crit_ind, V_S = $V_S_ind, V_S_p = $V_S_p_ind")
            # println("V_C_crit = $V_C_crit at $V_C_crit_ind, V_C = $V_C_ind, V_C_p = $V_C_p_ind")
            # println("q_S_crit = $q_S_crit at $q_S_crit_ind")
            # println("q_C_crit = $q_C_crit at $q_C_crit_ind")

            println("Age = ", age)
        end
end

parameters = parameters_function()
variables = variables_function(parameters; load_initial_value = true)
@time solve_function!(variables, parameters; tol = 1E-4, iter_max = 1000, one_loop = true)

# save and load workspace
V_S = variables.V_S
a_S_i = variables.a_S_i
n_S_i = variables.n_S_i
d_S_i = variables.d_S_i
V_div = variables.V_div
a_div_i = variables.a_div_i
n_div_i = variables.n_div_i
d_div_i = variables.d_div_i
V_C = variables.V_C
a_C_i = variables.a_C_i
n_C_i = variables.n_C_i
d_C_i = variables.d_C_i
P_S = variables.P_S
q_S = variables.q_S
P_C = variables.P_C
q_C = variables.q_C

# W_S_test = variables.W_S
# σ_S_test = variables.σ_S
# # v_div_R = variables.v_div_R
# # v_div_D = variables.v_div_D
# W_div_test = variables.W_div
# σ_div_test = variables.σ_div
# # v_C_R = variables.v_C_R
# # v_C_D = variables.v_C_D
# W_C_test = variables.W_C
# σ_C_test = variables.σ_C
# P_S_test = variables.P_S
# q_S_test = variables.q_S
# P_C_test = variables.P_C
# q_C_test = variables.q_C
# # μ_S = variables.μ_S
# # μ_C = variables.μ_C

using JLD2
# cd("C:/Users/jsun/Dropbox/Bankruptcy-Family/Results/Current/No-divorce-larger-grid-smaller-taste-new/phi-0.99")
# cd("C:/Users/jsun/Dropbox/Bankruptcy-Family/Results/Current/No-divorce-no-expense/phi-0.99")
# cd("C:/Users/jsun/Dropbox/Bankruptcy-Family/Results/Current/No-divorce-no-expense-six-times-standard-dev/phi-0.7")
cd("C:/Users/jsun/Dropbox/Bankruptcy-Family/Results/Current/Calibration-14/Singles_have_couples_expenses_no_scale")

@save "workspace.jld2" parameters V_S a_S_i n_S_i d_S_i V_div a_div_i n_div_i d_div_i V_C a_C_i n_C_i d_C_i q_S q_C

# @load "workspace.jld2"

# all(W_S .≈ W_S_test)
# all(σ_S .≈ σ_S_test)
#
# all(W_div .≈ W_div_test)
# all(σ_div .≈ σ_div_test)
#
# maximum(abs.(σ_div .- σ_div_test))
# sum(abs.(σ_div .- σ_div_test))
# sum(σ_div .≈ σ_div_test)
#
# all(W_C .≈ W_C_test)
# all(σ_C .≈ σ_C_test)
# maximum(abs.(σ_C .- σ_C_test))
#
#
# all(q_S .≈ q_S_test)
# all(q_C .≈ q_C_test)
