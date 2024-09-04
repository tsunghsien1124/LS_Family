# Simulation
using QuantEcon: Categorical
using Random
using ProgressMeter

# path_ws = "C:/Users/jsun/Dropbox/Bankruptcy-Family/Results/Current/Taste_shock_exo_divorce_0.1_normal_grid_no_dist_with_div_value"
# path_ws = "C:/Users/JanSun/Dropbox/Bankruptcy-Family/Results/Current/Taste_shock_exo_divorce_0.1_normal_grid_no_dist_with_div_value"

# path_ws = "C:/Users/jsun/Dropbox/Bankruptcy-Family/Results/Current/Taste_shock_exo_divorce_0.1_normal_grid_no_dist_with_div_value_phi_0.1"
# path_ws = "C:/Users/jsun/Dropbox/Bankruptcy-Family/Results/Current/Taste_shock_exo_divorce_0.1_normal_grid_no_dist_with_div_value_phi_0.3"
# path_ws = "C:/Users/jsun/Dropbox/Bankruptcy-Family/Results/Current/Taste_shock_exo_divorce_0.1_normal_grid_no_dist_with_div_value_phi_0.5"
# path_ws = "C:/Users/jsun/Dropbox/Bankruptcy-Family/Results/Current/Taste_shock_exo_divorce_0.1_normal_grid_no_dist_with_div_value_phi_0.7"
# path_ws = "C:/Users/jsun/Dropbox/Bankruptcy-Family/Results/Current/Taste_shock_exo_divorce_0.1_normal_grid_no_dist_with_div_value_phi_0.9"
# path_ws = "C:/Users/jsun/Dropbox/Bankruptcy-Family/Results/Current/Taste_shock_exo_divorce_0.1_normal_grid_no_dist_with_div_value_no_bankruptcy"

# cd(path_ws)

num_hh = 40000
num_S = Int(num_hh*parameters.ν)
num_C = num_hh - num_S

monte_sim = 10

# set seed
Random.seed!(95969)

# Panels for endogenous states and choices
# Singles
panel_S_a = zeros(Int, num_S, parameters.lifespan, 2, monte_sim)
panel_S_n = zeros(Int, num_S, parameters.lifespan, 2, monte_sim)
panel_S_d = zeros(Int, num_S, parameters.lifespan, 2, monte_sim)
panel_S_c = zeros(num_S, parameters.lifespan, 2, monte_sim)

# Couples
panel_C_a = zeros(Int, num_C, parameters.lifespan, 2, monte_sim)
panel_C_n = zeros(Int, num_C, parameters.lifespan, 2, monte_sim)
panel_C_d = zeros(Int, num_C, parameters.lifespan, 2, monte_sim)
panel_C_c = zeros(num_C, parameters.lifespan, 2, monte_sim)
# panel_C_div_ind = zeros(Int, num_C, parameters.lifespan, monte_sim)

# Draw shocks for entire panel
# Singles
# persistent productivity
shock_S_z = zeros(Int, num_S, parameters.lifespan, 2, monte_sim)
# transitory productivity
shock_S_η = zeros(Int, num_S, parameters.lifespan, 2, monte_sim)
# Expense shocks
shock_S_κ = zeros(Int, num_S, parameters.lifespan, 2, monte_sim)

# Couples
# Divorce shocks
shock_C_div = zeros(Int, num_C, parameters.lifespan, monte_sim)
shock_C_div[:,1,:] .= 2
# persistent productivity
shock_C_z = zeros(Int, num_C, parameters.lifespan, 2, monte_sim)
# transitory productivity
shock_C_η = zeros(Int, num_C, parameters.lifespan, 2, monte_sim)
# Expense shocks
shock_C_κ = zeros(Int, num_C, parameters.lifespan, 2, monte_sim)

# Auxiliary matrices
# Singles
# Women
σ_S_women_temp = reshape(σ_S_women,:,parameters.a_size,parameters.z_size,parameters.η_size,parameters.κ_size,parameters.lifespan)
temp_S_women = zeros(parameters.a_size+1,parameters.n_size_women)
# Men
σ_S_men_temp = reshape(σ_S_men,:,parameters.a_size,parameters.z_size,parameters.η_size,parameters.κ_size,parameters.lifespan)
temp_S_men = zeros(parameters.a_size+1,parameters.n_size_men)

# Divorced
# Women
σ_div_women_temp = reshape(σ_div_women,:,parameters.a_size,parameters.z_size,parameters.η_size,parameters.κ_size,parameters.lifespan)
temp_div_women = zeros(parameters.a_size+1,parameters.n_size_women)
# Men
σ_div_men_temp = reshape(σ_div_men,:,parameters.a_size,parameters.z_size,parameters.η_size,parameters.κ_size,parameters.lifespan)
temp_div_men = zeros(parameters.a_size+1,parameters.n_size_men)

# Couples
σ_C_temp = reshape(σ_C,:,parameters.a_size,parameters.z_size,parameters.z_size,parameters.η_size,parameters.η_size,parameters.κ_size,parameters.κ_size,parameters.lifespan)
temp_C = zeros(parameters.a_size+1,parameters.n_size_women,parameters.n_size_men)

# Loop over HHs and Time periods
@showprogress 1 "Computing..." for monte_i in 1:monte_sim
    for age in 1:parameters.lifespan
        # Singles
        for hh_i in 1:num_S, gender in 1:2
            if age == 1
                # Initiate exogenous states for newborns
                # Persistent productivity
                z_i = rand(Categorical(parameters.Γ_z_initial))
                shock_S_z[hh_i,age,gender,monte_i] = z_i
                # Transitory productivity
                η_i = rand(Categorical(parameters.Γ_η))
                shock_S_η[hh_i,age,gender,monte_i] = η_i
                # Expense shock
                κ_i = 1
                shock_S_κ[hh_i,age,gender,monte_i] = κ_i

                # Initialize asset state
                a_i = parameters.a_ind_zero
                panel_S_a[hh_i,age,gender,monte_i] = a_i

                # Compute choices
                if gender == 1
                    action_i = rand(Categorical(σ_S_women_temp[:,a_i,z_i,η_i,κ_i,age]))
                    a_p_i = CartesianIndices(temp_S_women)[action_i][1]
                    n_i = CartesianIndices(temp_S_women)[action_i][2]
                else
                    action_i = rand(Categorical(σ_S_men_temp[:,a_i,z_i,η_i,κ_i,age]))
                    a_p_i = CartesianIndices(temp_S_men)[action_i][1]
                    n_i = CartesianIndices(temp_S_men)[action_i][2]
                end

                if a_p_i == 1 # Default
                    panel_S_d[hh_i,age,gender,monte_i] = 2
                    panel_S_a[hh_i,age+1,gender,monte_i] = parameters.a_ind_zero
                    panel_S_n[hh_i,age,gender,monte_i] = n_i
                    if gender == 1
                        panel_S_c[hh_i,age,1,monte_i] = (parameters.z_grid[z_i,1]*parameters.η_grid[η_i]*parameters.h_grid[age]*parameters.n_grid_women[n_i])*(1.0-parameters.ϕ)
                    else gender == 2
                        panel_S_c[hh_i,age,2,monte_i] = (parameters.z_grid[z_i,2]*parameters.η_grid[η_i]*parameters.h_grid[age]*parameters.n_grid_men[n_i])*(1.0-parameters.ϕ)
                    end
                else
                    panel_S_d[hh_i,age,gender,monte_i] = 1
                    panel_S_a[hh_i,age+1,gender,monte_i] = a_p_i - 1
                    panel_S_n[hh_i,age,gender,monte_i] = n_i
                    if gender == 1
                        panel_S_c[hh_i,age,1,monte_i] = parameters.z_grid[z_i,1]*parameters.η_grid[η_i]*parameters.h_grid[age]*parameters.n_grid_women[n_i]+parameters.a_grid[a_i]-q_S[a_p_i-1,z_i,age,1]*parameters.a_grid[a_p_i-1] - parameters.κ_grid[κ_i]
                    else gender == 2
                        panel_S_c[hh_i,age,2,monte_i] = parameters.z_grid[z_i,2]*parameters.η_grid[η_i]*parameters.h_grid[age]*parameters.n_grid_men[n_i]+parameters.a_grid[a_i]-q_S[a_p_i-1,z_i,age,2]*parameters.a_grid[a_p_i-1] - parameters.κ_grid[κ_i]
                    end
                end

            else

                # extract exogenous states
                # Persistent productivity
                z_i = rand(Categorical(parameters.Γ_z[shock_S_z[hh_i,age-1,gender,monte_i],:,gender]))
                shock_S_z[hh_i,age,gender,monte_i] = z_i

                # Transitory productivity
                η_i = rand(Categorical(parameters.Γ_η))
                shock_S_η[hh_i,age,gender,monte_i] = η_i

                # Expense shock
                κ_i = rand(Categorical(parameters.Γ_κ))
                shock_S_κ[hh_i,age,gender,monte_i] = κ_i

                # extract endogenous states
                a_i = panel_S_a[hh_i, age, gender,monte_i]

                # Compute choices
                if gender == 1
                    action_i = rand(Categorical(σ_S_women_temp[:,a_i,z_i,η_i,κ_i,age]))
                    a_p_i = CartesianIndices(temp_S_women)[action_i][1]
                    n_i = CartesianIndices(temp_S_women)[action_i][2]
                else
                    action_i = rand(Categorical(σ_S_men_temp[:,a_i,z_i,η_i,κ_i,age]))
                    a_p_i = CartesianIndices(temp_S_men)[action_i][1]
                    n_i = CartesianIndices(temp_S_men)[action_i][2]
                end

                if a_p_i == 1 # Default
                    panel_S_d[hh_i,age,gender,monte_i] = 2
                    if age < parameters.lifespan
                        panel_S_a[hh_i,age+1,gender,monte_i] = parameters.a_ind_zero
                    end
                    panel_S_n[hh_i,age,gender,monte_i] = n_i
                    if gender == 1
                        panel_S_c[hh_i,age,1,monte_i] = (parameters.z_grid[z_i,1]*parameters.η_grid[η_i]*parameters.h_grid[age]*parameters.n_grid_women[n_i])*(1.0-parameters.ϕ)
                    else
                        panel_S_c[hh_i,age,2,monte_i] = (parameters.z_grid[z_i,2]*parameters.η_grid[η_i]*parameters.h_grid[age]*parameters.n_grid_men[n_i])*(1.0-parameters.ϕ)
                    end
                else # Repayment
                    panel_S_d[hh_i,age,gender,monte_i] = 1
                    if age < parameters.lifespan
                        panel_S_a[hh_i,age+1,gender,monte_i] = a_p_i - 1
                    end
                    panel_S_n[hh_i,age,gender,monte_i] = n_i
                    if gender == 1
                        panel_S_c[hh_i,age,1,monte_i] = parameters.z_grid[z_i,1]*parameters.η_grid[η_i]*parameters.h_grid[age]*parameters.n_grid_women[n_i]+parameters.a_grid[a_i]-q_S[a_p_i-1,z_i,age,1]*parameters.a_grid[a_p_i-1] - parameters.κ_grid[κ_i]
                    else
                        panel_S_c[hh_i,age,2,monte_i] = parameters.z_grid[z_i,2]*parameters.η_grid[η_i]*parameters.h_grid[age]*parameters.n_grid_men[n_i]+parameters.a_grid[a_i]-q_S[a_p_i-1,z_i,age,2]*parameters.a_grid[a_p_i-1] - parameters.κ_grid[κ_i]
                    end
                end
            end
        end

        # Couples
        for hh_i in 1:num_C
            if (age == 1) || (shock_C_div[hh_i,age,monte_i] == 2)
                if age == 1
                    # Initiate exogenous states for newborns
                    shock_C_div[hh_i,age+1,monte_i] = rand(Categorical([parameters.ψ, 1-parameters.ψ])) # 1: divorce, 2: no divorce

                    # Persistent productivity
                    z_1_i = rand(Categorical(parameters.Γ_z_initial))
                    z_2_i = rand(Categorical(parameters.Γ_z_initial))

                    shock_C_z[hh_i,age,1,monte_i] = z_1_i
                    shock_C_z[hh_i,age,2,monte_i] = z_2_i

                    # Transitory productivity
                    η_1_i = rand(Categorical(parameters.Γ_η))
                    η_2_i = rand(Categorical(parameters.Γ_η))

                    shock_C_η[hh_i,age,1,monte_i] = η_1_i
                    shock_C_η[hh_i,age,2,monte_i] = η_2_i

                    # Expense shock
                    κ_1_i = 1
                    κ_2_i = 1

                    shock_C_κ[hh_i,age,1,monte_i] = κ_1_i
                    shock_C_κ[hh_i,age,2,monte_i] = κ_2_i

                    # Initialize asset state
                    a_i = parameters.a_ind_zero
                    panel_C_a[hh_i,age,:,monte_i] .= a_i
                else
                    # extract exogenous states
                    if age < parameters.lifespan
                        shock_C_div[hh_i,age+1,monte_i] = rand(Categorical([parameters.ψ, 1-parameters.ψ]))# 1: divorce, 2: no divorce
                    end

                    # Persistent productivity
                    z_1_i = rand(Categorical(parameters.Γ_z[shock_C_z[hh_i,age-1,1,monte_i],:,1]))
                    z_2_i = rand(Categorical(parameters.Γ_z[shock_C_z[hh_i,age-1,2,monte_i],:,2]))

                    shock_C_z[hh_i,age,1,monte_i] = z_1_i
                    shock_C_z[hh_i,age,2,monte_i] = z_2_i

                    # Transitory productivity
                    η_1_i = rand(Categorical(parameters.Γ_η))
                    η_2_i = rand(Categorical(parameters.Γ_η))
                    shock_C_η[hh_i,age,1,monte_i] = η_1_i
                    shock_C_η[hh_i,age,2,monte_i] = η_2_i

                    # Expense shocks
                    κ_1_i = rand(Categorical(parameters.Γ_κ))
                    κ_2_i = rand(Categorical(parameters.Γ_κ))

                    shock_C_κ[hh_i,age,1,monte_i] = κ_1_i
                    shock_C_κ[hh_i,age,2,monte_i] = κ_2_i

                    # extract endogenous states
                    a_i = panel_C_a[hh_i,age,1,monte_i]

                end

                # Compute choices
                action_i = rand(Categorical(σ_C_temp[:,a_i,z_1_i,z_2_i,η_1_i,η_2_i,κ_1_i,κ_2_i,age]))
                a_p_i = CartesianIndices(temp_C)[action_i][1]
                n_1_i = CartesianIndices(temp_C)[action_i][2]
                n_2_i = CartesianIndices(temp_C)[action_i][3]

                if a_p_i == 1 # Default
                    panel_C_d[hh_i,age,:,monte_i] .= 2
                    if age < parameters.lifespan
                        panel_C_a[hh_i,age+1,:,monte_i] .= parameters.a_ind_zero
                    end
                    panel_C_n[hh_i,age,1,monte_i] = n_1_i
                    panel_C_n[hh_i,age,2,monte_i] = n_2_i
                    x = (parameters.z_grid[z_1_i,1]*parameters.η_grid[η_1_i]*parameters.h_grid[age]*parameters.n_grid_women[n_1_i]+parameters.z_grid[z_2_i,2]*parameters.η_grid[η_2_i]*parameters.h_grid[age]*parameters.n_grid_men[n_2_i])*(1.0-parameters.ϕ)
                    panel_C_c[hh_i,age,:,monte_i] .= x/(2^(1/parameters.ρ_s))

                else
                    panel_C_d[hh_i,age,:,monte_i] .= 1
                    if age < parameters.lifespan
                        panel_C_a[hh_i,age+1,:,monte_i] .= a_p_i - 1
                    end
                    panel_C_n[hh_i,age,1,monte_i] = n_1_i
                    panel_C_n[hh_i,age,2,monte_i] = n_2_i
                    x = parameters.z_grid[z_1_i,1]*parameters.η_grid[η_1_i]*parameters.h_grid[age]*parameters.n_grid_women[n_1_i]+parameters.z_grid[z_2_i,2]*parameters.η_grid[η_2_i]*parameters.h_grid[age]*parameters.n_grid_men[n_2_i]+parameters.a_grid[a_i]-q_C[a_p_i-1,z_1_i,z_2_i,age]*parameters.a_grid[a_p_i-1]-parameters.κ_grid[κ_1_i]- parameters.κ_grid[κ_2_i]
                    panel_C_c[hh_i,age,:,monte_i] .= x/(2^(1/parameters.ρ_s))
                end

            elseif shock_C_div[hh_i,age,monte_i] == 1 # Divorced
                if age < parameters.lifespan
                    shock_C_div[hh_i,age+1,monte_i] = 1 # 1: divorce, 2: no divorce
                end

                if shock_C_div[hh_i,age-1,monte_i] == 2 # Married in last period
                    a = parameters.a_grid[panel_C_a[hh_i,age,1,monte_i]]

                    if a/2 in parameters.a_grid
                        a_i_half_1 = findfirst(parameters.a_grid .== a/2)
                        a_i_half_2 = a_i_half_1
                    else
                        a_i_half_1 = findlast(parameters.a_grid .< a/2)
                        a_i_half_2 = findfirst(parameters.a_grid .> a/2)
                    end

                    panel_C_a[hh_i,age,1,monte_i] = a_i_half_1 # Change asset position to per capita
                    panel_C_a[hh_i,age,2,monte_i] = a_i_half_2
                end

                for gender in 1:2
                    # Persistent productivity
                    z_i = rand(Categorical(parameters.Γ_z[shock_C_z[hh_i,age-1,gender,monte_i],:,gender]))

                    shock_C_z[hh_i,age,gender,monte_i] = z_i

                    # Transitory productivity
                    η_i = rand(Categorical(parameters.Γ_η))

                    shock_C_η[hh_i,age,gender,monte_i] = η_i

                    # Expense shocks
                    κ_i = rand(Categorical(parameters.Γ_κ))

                    shock_C_κ[hh_i,age,gender,monte_i] = κ_i

                    # extract endogenous states
                    a_i = panel_C_a[hh_i,age,gender,monte_i]

                    # Compute choices
                    if gender == 1
                        action_i = rand(Categorical(σ_div_women_temp[:,a_i,z_i,η_i,κ_i,age]))
                        a_p_i = CartesianIndices(temp_div_women)[action_i][1]
                        n_i = CartesianIndices(temp_div_women)[action_i][2]
                    else
                        action_i = rand(Categorical(σ_div_men_temp[:,a_i,z_i,η_i,κ_i,age]))
                        a_p_i = CartesianIndices(temp_div_men)[action_i][1]
                        n_i = CartesianIndices(temp_div_men)[action_i][2]
                    end

                    if a_p_i == 1 # Default
                        panel_C_d[hh_i,age,gender,monte_i] = 2
                        if age < parameters.lifespan
                            panel_C_a[hh_i,age+1,gender,monte_i] = parameters.a_ind_zero
                        end
                        panel_C_n[hh_i,age,gender,monte_i] = n_i

                        if gender == 1
                            panel_C_c[hh_i,age,1,monte_i] = (parameters.z_grid[z_i,1]*parameters.η_grid[η_i]*parameters.h_grid[age]*parameters.n_grid_women[n_i])*(1.0-parameters.ϕ)
                        else
                            panel_C_c[hh_i,age,2,monte_i] = (parameters.z_grid[z_i,2]*parameters.η_grid[η_i]*parameters.h_grid[age]*parameters.n_grid_men[n_i])*(1.0-parameters.ϕ)
                        end

                    else
                        panel_C_d[hh_i,age,gender,monte_i] = 1
                        if age < parameters.lifespan
                            panel_C_a[hh_i,age+1,gender,monte_i] = a_p_i - 1
                        end
                        panel_C_n[hh_i,age,gender,monte_i] = n_i

                        if gender == 1
                            panel_C_c[hh_i,age,1,monte_i] = parameters.z_grid[z_i,1]*parameters.η_grid[η_i]*parameters.h_grid[age]*parameters.n_grid_women[n_i]+parameters.a_grid[a_i]-q_S[a_p_i-1,z_i,age,1]*parameters.a_grid[a_p_i-1]-parameters.κ_grid[κ_i]-parameters.κ_div
                        else
                            panel_C_c[hh_i,age,2,monte_i] = parameters.z_grid[z_i,2]*parameters.η_grid[η_i]*parameters.h_grid[age]*parameters.n_grid_men[n_i]+parameters.a_grid[a_i]-q_S[a_p_i-1,z_i,age,2]*parameters.a_grid[a_p_i-1]-parameters.κ_grid[κ_i]-parameters.κ_div
                        end
                    end

                end

            end
        end
    end
end

@save "simulations.jld2" panel_S_a panel_S_d panel_S_n panel_S_c shock_S_z shock_S_η shock_S_κ panel_C_a panel_C_n panel_C_d panel_C_c shock_C_div shock_C_z shock_C_η shock_C_κ num_hh num_S num_C monte_sim

@load "simulations.jld2"

# With moments
# @save "simulations.jld2" panel_S_a panel_S_d panel_S_n panel_S_c panel_S_age shock_S_ρ shock_S_z shock_S_η panel_div_1_a panel_div_1_d panel_div_1_n panel_div_1_c panel_div_1_age panel_div_1_ind shock_div_1_ρ shock_div_1_z shock_div_1_η panel_div_2_a panel_div_2_d panel_div_2_n panel_div_2_c panel_div_2_age panel_div_2_ind shock_div_2_ρ shock_div_2_z shock_div_2_η panel_C_a panel_C_d panel_C_1_n panel_C_2_n panel_C_c panel_C_age shock_C_ρ shock_C_div shock_C_z_1 shock_C_z_2 shock_C_η_1 shock_C_η_2 moments

# period_all = size(panel_S_a)[2]
# # period = size(panel_asset)[2]
# period = size(panel_S_a)[2]-20
