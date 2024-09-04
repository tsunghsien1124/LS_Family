# Simulation
using QuantEcon: Categorical
using Random
using ProgressMeter

num_hh = 40000
# num_S = Int(num_hh*parameters.ν)
num_S = Int(num_hh*parameters.ν)
num_C = num_hh - num_S
num_periods = 2000+1
burn_in = 100

# set seed
Random.seed!(325)

# Panels for endogenous states and choices
# Singles
panel_S_a = zeros(Int, num_S, num_periods)
panel_S_n = zeros(Int, num_S, num_periods)
panel_S_d = zeros(Int, num_S, num_periods)
panel_S_age = zeros(Int, num_S, num_periods)

# Couples
panel_C_a = zeros(Int, num_C, num_periods)
panel_C_1_n = zeros(Int, num_C, num_periods)
panel_C_2_n = zeros(Int, num_C, num_periods)
panel_C_d = zeros(Int, num_C, num_periods)
panel_C_age = zeros(Int, num_C, num_periods)

# Draw shocks for entire panel
# Singles
# survival rate
shock_S_ρ = rand(Categorical([parameters.ρ, 1-parameters.ρ]), (num_S, num_periods))
# persistent productivity
shock_S_z = zeros(Int, num_S, num_periods)
# transitory productivity
shock_S_η = zeros(Int, num_S, num_periods)
# Expense shocks
shock_S_κ = zeros(Int, num_S, num_periods)

# Couples
shock_C_ρ = rand(Categorical([parameters.ρ, 1-parameters.ρ]), (num_C, num_periods))
# persistent productivity
shock_C_z_1 = zeros(Int, num_C, num_periods)
shock_C_z_2 = zeros(Int, num_C, num_periods)
# transitory productivity
shock_C_η_1 = zeros(Int, num_C, num_periods)
shock_C_η_2 = zeros(Int, num_C, num_periods)
# Expense shocks
shock_C_κ_1 = zeros(Int, num_C, num_periods)
shock_C_κ_2 = zeros(Int, num_C, num_periods)

# Loop over HHs and Time periods
@showprogress 1 "Computing..." for period_i in 1:(num_periods-1)
    # Singles
    Threads.@threads for hh_i in 1:num_S
        if period_i == 1 || shock_S_ρ[hh_i,period_i] == 2

            # Initiate exogenous states for newborns
            # Persistent productivity
            z_i = 3
            shock_S_z[hh_i,period_i] = z_i
            # Transitory productivity
            η_i = rand(Categorical(parameters.Γ_η))
            shock_S_η[hh_i,period_i] = η_i
            # Expense shock
            κ_i = 1
            shock_S_κ[hh_i,period_i] = κ_i
            # Age
            panel_S_age[hh_i,period_i] = 0

            # Initialize asset state
            a_i = parameters.a_ind_zero
            panel_S_a[hh_i,period_i] = a_i

            # Compute choices
            # a
            panel_S_a[hh_i,period_i+1] = a_S_i[a_i,z_i,η_i,κ_i]
            # default decision
            panel_S_d[hh_i,period_i] = d_S_i[a_i,z_i,η_i,κ_i]
            # Labor Supply
            panel_S_n[hh_i,period_i] = n_S_i[a_i,z_i,η_i,κ_i]

        else

            # extract exogenous states
            # Persistent productivity
            z_i = rand(Categorical(parameters.Γ_z[shock_S_z[hh_i,period_i-1],:]))
            shock_S_z[hh_i,period_i] = z_i

            # Transitory productivity
            η_i = rand(Categorical(parameters.Γ_η))
            shock_S_η[hh_i,period_i] = η_i

            # Expense shock
            κ_i = rand(Categorical(parameters.Γ_κ))
            shock_S_κ[hh_i,period_i] = κ_i

            # age
            panel_S_age[hh_i,period_i] = panel_S_age[hh_i,period_i-1] + 1

            # extract endogenous states
            a_i = panel_S_a[hh_i, period_i]

            # Compute choices
            # a
            panel_S_a[hh_i,period_i+1] = a_S_i[a_i,z_i,η_i,κ_i]
            # default decision
            panel_S_d[hh_i,period_i] = d_S_i[a_i,z_i,η_i,κ_i]
            # Labor Supply
            panel_S_n[hh_i,period_i] = n_S_i[a_i,z_i,η_i,κ_i]

        end
    end

    # Couples
    Threads.@threads for hh_i in 1:num_C
        if period_i == 1 || shock_C_ρ[hh_i,period_i] == 2

            # Initiate exogenous states for newborns
            # Persistent productivity
            z_1_i = 3
            z_2_i = 3
            shock_C_z_1[hh_i,period_i] = z_1_i
            shock_C_z_2[hh_i,period_i] = z_2_i
            # Transitory productivity
            η_1_i = rand(Categorical(parameters.Γ_η))
            η_2_i = rand(Categorical(parameters.Γ_η))
            shock_C_η_1[hh_i,period_i] = η_1_i
            shock_C_η_2[hh_i,period_i] = η_2_i
            # Expense shock
            κ_1_i = 1
            shock_C_κ_1[hh_i,period_i] = κ_1_i
            κ_2_i = 1
            shock_C_κ_2[hh_i,period_i] = κ_2_i
            # Age
            panel_C_age[hh_i,period_i] = 0

            # Initialize asset state
            a_i = parameters.a_ind_zero
            panel_C_a[hh_i,period_i] = a_i

            # Compute choices
            # a
            panel_C_a[hh_i,period_i+1] = a_C_i[a_i,z_1_i,z_2_i,η_1_i,η_2_i,κ_1_i,κ_2_i]
            # default decision
            panel_C_d[hh_i,period_i] = d_C_i[a_i,z_1_i,z_2_i,η_1_i,η_2_i,κ_1_i,κ_2_i]
            # Labor Supply
            panel_C_1_n[hh_i,period_i] = n_C_1_i[a_i,z_1_i,z_2_i,η_1_i,η_2_i,κ_1_i,κ_2_i]
            panel_C_2_n[hh_i,period_i] = n_C_2_i[a_i,z_1_i,z_2_i,η_1_i,η_2_i,κ_1_i,κ_2_i]

        else

            # extract exogenous states
            # Persistent productivity
            z_1_i = rand(Categorical(parameters.Γ_z[shock_C_z_1[hh_i,period_i-1],:]))
            z_2_i = rand(Categorical(parameters.Γ_z[shock_C_z_2[hh_i,period_i-1],:]))
            shock_C_z_1[hh_i,period_i] = z_1_i
            shock_C_z_2[hh_i,period_i] = z_2_i

            # Transitory productivity
            η_1_i = rand(Categorical(parameters.Γ_η))
            η_2_i = rand(Categorical(parameters.Γ_η))
            shock_C_η_1[hh_i,period_i] = η_1_i
            shock_C_η_2[hh_i,period_i] = η_2_i

            # Expense shocks
            κ_1_i = rand(Categorical(parameters.Γ_κ))
            shock_C_κ_1[hh_i,period_i] = κ_1_i
            κ_2_i = rand(Categorical(parameters.Γ_κ))
            shock_C_κ_2[hh_i,period_i] = κ_2_i

            # age
            panel_C_age[hh_i,period_i] = panel_C_age[hh_i,period_i-1] + 1

            # extract endogenous states
            a_i = panel_C_a[hh_i, period_i]

            # Compute choices
            # a
            panel_C_a[hh_i,period_i+1] = a_C_i[a_i,z_1_i,z_2_i,η_1_i,η_2_i,κ_1_i,κ_2_i]
            # default decision
            panel_C_d[hh_i,period_i] = d_C_i[a_i,z_1_i,z_2_i,η_1_i,η_2_i,κ_1_i,κ_2_i]
            # Labor Supply
            panel_C_1_n[hh_i,period_i] = n_C_1_i[a_i,z_1_i,z_2_i,η_1_i,η_2_i,κ_1_i,κ_2_i]
            panel_C_2_n[hh_i,period_i] = n_C_2_i[a_i,z_1_i,z_2_i,η_1_i,η_2_i,κ_1_i,κ_2_i]
        end
    end
end

# Cut burn-in and last period
# Singles
panel_S_a = panel_S_a[:,burn_in+1:end-1]
panel_S_d = panel_S_d[:,burn_in+1:end-1]
panel_S_n = panel_S_n[:,burn_in+1:end-1]
panel_S_age = panel_S_age[:,burn_in+1:end-1]
shock_S_ρ = shock_S_ρ[:,burn_in+1:end-1]
shock_S_z = shock_S_z[:,burn_in+1:end-1]
shock_S_η = shock_S_η[:,burn_in+1:end-1]
shock_S_κ = shock_S_κ[:,burn_in+1:end-1]

# Couples
panel_C_a = panel_C_a[:,burn_in+1:end-1]
panel_C_d = panel_C_d[:,burn_in+1:end-1]
panel_C_1_n = panel_C_1_n[:,burn_in+1:end-1]
panel_C_2_n = panel_C_2_n[:,burn_in+1:end-1]
panel_C_age = panel_C_age[:,burn_in+1:end-1]
shock_C_ρ = shock_C_ρ[:,burn_in+1:end-1]
shock_C_z_1 = shock_C_z_1[:,burn_in+1:end-1]
shock_C_z_2 = shock_C_z_2[:,burn_in+1:end-1]
shock_C_η_1 = shock_C_η_1[:,burn_in+1:end-1]
shock_C_η_2 = shock_C_η_2[:,burn_in+1:end-1]
shock_C_κ_1 = shock_C_κ_1[:,burn_in+1:end-1]
shock_C_κ_2 = shock_C_κ_2[:,burn_in+1:end-1]

period_all = size(panel_S_a)[2]
# period = size(panel_asset)[2]
period = size(panel_S_a)[2]-20

#=
#===============#
# Check moments #
#===============#
#======================#
# 1) Full default rate #
#======================#
fraction_full_default_sim_all_periods = zeros(period_all)
for i in 1:period_all
    fraction_full_default_sim_all_periods[i] = sum(panel_default[:,i] .== 1)/num_hh*100
end
fraction_full_default_sim_all_periods_ave = sum(fraction_full_default_sim_all_periods)/period_all

using Plots
# Plot across time
plot(1:period_all, fraction_full_default_sim_all_periods)

# In specific period
fraction_full_default_sim = sum(panel_default[:,period] .== 1)/num_hh*100

#=========================#
# 2) Partial default rate #
#=========================#
fraction_partial_default_sim_all_periods = zeros(period_all)
for i in 1:period_all
    fraction_partial_default_sim_all_periods[i] = sum(panel_default[:,i] .== 2)/num_hh*100
end
fraction_partial_default_sim_all_periods_ave = sum(fraction_partial_default_sim_all_periods)/period_all

# Plot across time
plot(1:period_all, fraction_partial_default_sim_all_periods)

# In specific period
fraction_partial_default_sim = sum(panel_default[:,period] .== 2)/num_hh*100
=#
#================================#
# 3) Fraction of bank loan users #
#================================#
# Singles
fraction_S_loan_users_sim_all_periods = zeros(period_all)
for i in 1:period_all
    fraction_S_loan_users_sim_all_periods[i] = sum(panel_S_a[:,i] .< parameters.a_ind_zero)/num_S*100
end
fraction_S_loan_users_sim_all_periods_ave = sum(fraction_S_loan_users_sim_all_periods)/period_all

# Plot across time
plot(1:period_all, fraction_S_loan_users_sim_all_periods)

# In specific period
fraction_S_loan_users_sim = sum(panel_S_a[:,period] .< parameters.a_ind_zero)/num_S*100

#=
#============================#
# 7) Average bank loan price #
#============================#
ave_bank_rate_sim_all_periods = zeros(period_all-1)
for i in 2:period_all
    ave_bank_price_sim_num = 0.0
    ave_bank_price_sim_den = 0.0

    for hh_i in 1:num_hh
        a_p_i, b_p_i = parameters.asset_ind[panel_asset[hh_i,i],:]

        if a_p_i < parameters.a_ind_zero

            action_a_i = 1  + a_p_i

            a_i, b_i = parameters.asset_ind[panel_asset[hh_i,i-1],:]

            ave_bank_price_sim_num += q_a[action_a_i-1,shock_e[hh_i,i-1],a_i,panel_score[hh_i,i-1]]
            ave_bank_price_sim_den += 1.0
        end
    end

    ave_bank_price_sim = ave_bank_price_sim_num/ave_bank_price_sim_den
    ave_bank_rate_sim = (1.0/ave_bank_price_sim) - 1.0

    ave_bank_rate_sim_all_periods[i-1] = ave_bank_rate_sim
end
ave_bank_rate_sim_all_periods_ave = sum(ave_bank_rate_sim_all_periods)/(period_all-1)

# For specific period
ave_bank_price_sim_num = 0.0
ave_bank_price_sim_den = 0.0

for hh_i in 1:num_hh
    a_p_i, b_p_i = parameters.asset_ind[panel_asset[hh_i,period],:]

    if a_p_i < parameters.a_ind_zero
        #=
        if panel_default[hh_i,period-1] == 1
            action_a_i = 1
        elseif panel_default[hh_i,period-1] == 2
            action_a_i = 1 + a_p_i
        else
            action_a_i = 1 + parameters.a_size + a_p_i
        end
        =#

        action_a_i = 1  + a_p_i

        a_i, b_i = parameters.asset_ind[panel_asset[hh_i,period-1],:]

        ave_bank_price_sim_num += q_a[action_a_i-1,shock_e[hh_i,period-1],a_i,panel_score[hh_i,period-1]]
        ave_bank_price_sim_den += 1.0
    end
end

ave_bank_price_sim = ave_bank_price_sim_num/ave_bank_price_sim_den
ave_bank_rate_sim = (1.0/ave_bank_price_sim) - 1.0

#========================#
# Simulated Distribution #
#========================#
μ_sim = zeros(parameters.β_size, parameters.z_size, parameters.e_size, parameters.asset_size, parameters.s_size)

@showprogress 1 "Computing..." for s_i in 1:parameters.s_size, asset_i in 1:parameters.asset_size, e_i in 1:parameters.e_size, z_i in 1:parameters.z_size, β_i in 1:parameters.β_size
    # HH's type
    ind_hh_type = [(shock_β .== β_i) .& (shock_z .== z_i) .& (shock_e .== e_i) .& (panel_asset .== asset_i) .& (panel_score .== s_i)][1]

    # simulated distribution
    μ_sim[β_i,z_i,e_i,asset_i,s_i] = sum(ind_hh_type)/(num_hh*period_all)
end
=#

#===============#
# Event studies #
#===============#
event_window = 5

# Effect of switch from z_3 to z_1
z_before = 3
z_after = 2

# event_S = zeros(num_S,period_all-2*event_window)
# for hh_i in 1:num_S, period_i in (event_window+1):(period_all-event_window)
#     event_S[hh_i,period_i-event_window] = (shock_S_z[hh_i,period_i] == z_after) & (shock_S_z[hh_i,period_i-1] == z_before)
# end

# On default probability (singles vs. couples)
d_grid = [0.0 1.0]
# Singles
event_S = zeros(num_S,period_all)
for hh_i in 1:num_S, period_i in 2:period_all
    event_S[hh_i,period_i] = (shock_S_z[hh_i,period_i] == z_after) & (shock_S_z[hh_i,period_i-1] == z_before)
end

# event_S = event_S[:,(event_window+1):(period_all-event_window)]

num_events_S =  Int(sum(event_S[:,3:(period_all-event_window)]))

event_S_default = zeros(num_events_S,event_window+3)

counter = 0

for hh_i in 1:num_S, period_i in 2:period_all
    if (event_S[hh_i,period_i] == 1) && (period_i > 2)  && (period_i <= period_all-event_window)
        counter += 1
        event_S_default[counter,:] = d_grid[panel_S_d[hh_i,(period_i-2):(period_i+event_window)]]
    end
end

event_S_default_ave = sum(event_S_default,dims=1)./counter

event_S_default_ave_norm = event_S_default_ave ./ event_S_default_ave[2]

# Couples
event_C = zeros(num_C,period_all)
for hh_i in 1:num_C, period_i in 2:period_all
    event_C[hh_i,period_i] = (shock_C_z_1[hh_i,period_i] == z_after) & (shock_C_z_1[hh_i,period_i-1] == z_before)
end

# event_S = event_S[:,(event_window+1):(period_all-event_window)]

num_events_C =  Int(sum(event_C[:,3:(period_all-event_window)]))

event_C_default = zeros(num_events_C,event_window+3)

counter = 0

for hh_i in 1:num_C, period_i in 2:period_all
    if (event_C[hh_i,period_i] == 1) && (period_i > 2)  && (period_i <= period_all-event_window)
        counter += 1
        event_C_default[counter,:] = d_grid[panel_C_d[hh_i,(period_i-2):(period_i+event_window)]]
    end
end

event_C_default_ave = sum(event_C_default,dims=1)./counter

event_C_default_ave_norm = event_C_default_ave ./ event_C_default_ave[2]

# On labour supply (singles vs. couples)
# Singles
event_S_labor = zeros(num_events_S,event_window+3)

counter = 0

for hh_i in 1:num_S, period_i in 2:period_all
    if (event_S[hh_i,period_i] == 1) && (period_i > 2)  && (period_i <= period_all-event_window)
        counter += 1
        event_S_labor[counter,:] = parameters.n_grid[panel_S_n[hh_i,(period_i-2):(period_i+event_window)]]
    end
end

event_S_labor_ave = sum(event_S_labor,dims=1)./counter

event_S_labor_ave_norm = event_S_labor_ave ./ event_S_labor_ave[2]

# Couples
event_C_labor_1 = zeros(num_events_C,event_window+3)
event_C_labor_2 = zeros(num_events_C,event_window+3)

counter = 0

for hh_i in 1:num_C, period_i in 2:period_all
    if (event_C[hh_i,period_i] == 1) && (period_i > 2)  && (period_i <= period_all-event_window)
        counter += 1
        event_C_labor_1[counter,:] = parameters.n_grid[panel_C_1_n[hh_i,(period_i-2):(period_i+event_window)]]
        event_C_labor_2[counter,:] = parameters.n_grid[panel_C_2_n[hh_i,(period_i-2):(period_i+event_window)]]
    end
end

event_C_labor_1_ave = sum(event_C_labor_1,dims=1)./counter
event_C_labor_2_ave = sum(event_C_labor_2,dims=1)./counter

event_C_labor_1_ave_norm = event_C_labor_1_ave ./ event_C_labor_1_ave[2]
event_C_labor_2_ave_norm = event_C_labor_2_ave ./ event_C_labor_2_ave[2]

# On asset choices
# Singles
# event_S_assets = zeros(num_events_S,event_window+3)
#
# counter = 0
# 
# for hh_i in 1:num_S, period_i in 2:period_all
#     if (event_S[hh_i,period_i] == 1) && (period_i > 2)  && (period_i <= period_all-event_window-1) && (panel_S_d[hh_i,period_i] == 1)
#         counter += 1
#         event_S_assets[counter,:] = parameters.a_grid[panel_S_a[hh_i,(period_i-1):(period_i+event_window+1)]]
#     end
# end
#
# event_S_assets_ave = sum(event_S_assets,dims=1)./counter
#
# event_S_assets_ave_norm = event_S_assets_ave ./ event_S_assets_ave[2]

# Couples

################################################################################
# Effect of transitory shock on default rate
η_shock = 1
# Singles
d_grid = [0.0 1.0]
# Singles
event_η_S = zeros(num_S,period_all)
for hh_i in 1:num_S, period_i in 2:period_all
    event_η_S[hh_i,period_i] = (shock_S_η[hh_i,period_i] == η_shock)
end

# event_S = event_S[:,(event_window+1):(period_all-event_window)]

num_events_η_S =  Int(sum(event_η_S[:,3:(period_all-event_window)]))

event_η_S_default = zeros(num_events_η_S,event_window+3)

counter = 0

for hh_i in 1:num_S, period_i in 2:period_all
    if (event_η_S[hh_i,period_i] == 1) && (period_i > 2)  && (period_i <= period_all-event_window)
        counter += 1
        event_η_S_default[counter,:] = d_grid[panel_S_d[hh_i,(period_i-2):(period_i+event_window)]]
    end
end

event_η_S_default_ave = sum(event_η_S_default,dims=1)./counter

event_η_S_default_ave_norm = event_η_S_default_ave ./ event_η_S_default_ave[2]

# Couples
event_η_C = zeros(num_C,period_all)
for hh_i in 1:num_C, period_i in 2:period_all
    event_η_C[hh_i,period_i] = (shock_C_η_1[hh_i,period_i] == η_shock)
end

# event_S = event_S[:,(event_window+1):(period_all-event_window)]

num_events_η_C =  Int(sum(event_η_C[:,3:(period_all-event_window)]))

event_η_C_default = zeros(num_events_η_C,event_window+3)

counter = 0

for hh_i in 1:num_C, period_i in 2:period_all
    if (event_η_C[hh_i,period_i] == 1) && (period_i > 2)  && (period_i <= period_all-event_window)
        counter += 1
        event_η_C_default[counter,:] = d_grid[panel_C_d[hh_i,(period_i-2):(period_i+event_window)]]
    end
end

event_η_C_default_ave = sum(event_η_C_default,dims=1)./counter

event_η_C_default_ave_norm = event_η_C_default_ave ./ event_η_C_default_ave[2]

################################################################################
# Effect of expense shock on default rate
κ_shock = 2
d_grid = [0.0 1.0]
# Singles
event_κ_S = zeros(num_S,period_all)
for hh_i in 1:num_S, period_i in 2:period_all
    event_κ_S[hh_i,period_i] = (shock_S_κ[hh_i,period_i] == κ_shock)
end

# event_S = event_S[:,(event_window+1):(period_all-event_window)]

num_events_κ_S =  Int(sum(event_κ_S[:,3:(period_all-event_window)]))

event_κ_S_default = zeros(num_events_κ_S,event_window+3)

counter = 0

for hh_i in 1:num_S, period_i in 2:period_all
    if (event_κ_S[hh_i,period_i] == 1) && (period_i > 2)  && (period_i <= period_all-event_window)
        counter += 1
        event_κ_S_default[counter,:] = d_grid[panel_S_d[hh_i,(period_i-2):(period_i+event_window)]]
    end
end

event_κ_S_default_ave = sum(event_κ_S_default,dims=1)./counter

event_κ_S_default_ave_norm = event_κ_S_default_ave ./ event_κ_S_default_ave[2]

event_κ_S_default_ave_norm_sum = zeros(3)
event_κ_S_default_ave_norm_sum[1] = event_κ_S_default_ave[2] ./ event_κ_S_default_ave[2]
event_κ_S_default_ave_norm_sum[2] = event_κ_S_default_ave[3] ./ event_κ_S_default_ave[2]
event_κ_S_default_ave_norm_sum[3] = (event_κ_S_default_ave[3]+event_κ_S_default_ave[4]) ./ event_κ_S_default_ave[2]

# Couples
event_κ_1_C = zeros(num_C,period_all)
for hh_i in 1:num_C, period_i in 2:period_all
    event_κ_1_C[hh_i,period_i] = (shock_C_κ_1[hh_i,period_i] == κ_shock)
end

# event_S = event_S[:,(event_window+1):(period_all-event_window)]

num_events_κ_1_C =  Int(sum(event_κ_1_C[:,3:(period_all-event_window)]))

event_κ_1_C_default = zeros(num_events_κ_1_C,event_window+3)

counter = 0

for hh_i in 1:num_C, period_i in 2:period_all
    if (event_κ_1_C[hh_i,period_i] == 1) && (period_i > 2)  && (period_i <= period_all-event_window)
        counter += 1
        event_κ_1_C_default[counter,:] = d_grid[panel_C_d[hh_i,(period_i-2):(period_i+event_window)]]
    end
end

event_κ_1_C_default_ave = sum(event_κ_1_C_default,dims=1)./counter

event_κ_1_C_default_ave_norm = event_κ_1_C_default_ave ./ event_κ_1_C_default_ave[2]

event_κ_1_C_default_ave_norm_sum = zeros(3)
event_κ_1_C_default_ave_norm_sum[1] = event_κ_1_C_default_ave[2] ./ event_κ_1_C_default_ave[2]
event_κ_1_C_default_ave_norm_sum[2] = event_κ_1_C_default_ave[3] ./ event_κ_1_C_default_ave[2]
event_κ_1_C_default_ave_norm_sum[3] = (event_κ_1_C_default_ave[3]+event_κ_1_C_default_ave[4]) ./ event_κ_1_C_default_ave[2]

# On labour supply (singles vs. couples)
# Singles
event_κ_S_labor = zeros(num_events_κ_S,event_window+3)

counter = 0

for hh_i in 1:num_S, period_i in 2:period_all
    if (event_κ_S[hh_i,period_i] == 1) && (period_i > 2)  && (period_i <= period_all-event_window)
        counter += 1
        event_κ_S_labor[counter,:] = parameters.n_grid[panel_S_n[hh_i,(period_i-2):(period_i+event_window)]]
    end
end

event_κ_S_labor_ave = sum(event_κ_S_labor,dims=1)./counter

event_κ_S_labor_ave_norm = event_κ_S_labor_ave ./ event_κ_S_labor_ave[2]

# Couples
event_κ_1_C_labor_1 = zeros(num_events_κ_1_C,event_window+3)
event_κ_1_C_labor_2 = zeros(num_events_κ_1_C,event_window+3)

counter = 0

for hh_i in 1:num_C, period_i in 2:period_all
    if (event_κ_1_C[hh_i,period_i] == 1) && (period_i > 2)  && (period_i <= period_all-event_window)
        counter += 1
        event_κ_1_C_labor_1[counter,:] = parameters.n_grid[panel_C_1_n[hh_i,(period_i-2):(period_i+event_window)]]
        event_κ_1_C_labor_2[counter,:] = parameters.n_grid[panel_C_2_n[hh_i,(period_i-2):(period_i+event_window)]]
    end
end

event_κ_1_C_labor_1_ave = sum(event_κ_1_C_labor_1,dims=1)./counter
event_κ_1_C_labor_2_ave = sum(event_κ_1_C_labor_2,dims=1)./counter

event_κ_1_C_labor_ave_sum = (event_κ_1_C_labor_1_ave + event_κ_1_C_labor_2_ave) ./ 2

event_κ_1_C_labor_1_ave_norm = event_κ_1_C_labor_1_ave ./ event_κ_1_C_labor_1_ave[2]
event_κ_1_C_labor_2_ave_norm = event_κ_1_C_labor_2_ave ./ event_κ_1_C_labor_2_ave[2]

event_κ_1_C_labor_ave_sum_norm = event_κ_1_C_labor_ave_sum ./ event_κ_1_C_labor_ave_sum[2]

# Effect of default on labor supply
# Singles


#=
# Effect of partial default on type score
event_PD = findall(panel_default .== 2)
event_PD_score = []

for event_i in 1:length(event_PD)
    hh_i = event_PD[event_i][1]
    period_i = event_PD[event_i][2]

    if 1 < period_i < num_periods-burn_in-1
        if all(shock_ρ[hh_i,period_i-1:period_i+1] .== 1)
            append!(event_PD_score, parameters.s_grid[panel_score[hh_i,period_i-1:period_i+1]])
        end
    end
end

event_PD_score = reshape(event_PD_score, (3,:))
event_PD_score_mean = sum(event_PD_score,dims=2) ./ size(event_PD_score)[2]

# Effect of partial default on type score conditional on high earners
event_PD_high_e = findall(panel_default .== 2)
event_PD_high_e_score = []

for event_i in 1:length(event_PD_high_e)
    hh_i = event_PD_high_e[event_i][1]
    period_i = event_PD_high_e[event_i][2]

    if 1 < period_i < num_periods-burn_in-1
        if all(shock_ρ[hh_i,period_i-1:period_i+1] .== 1)
            if shock_e[hh_i,period_i] == 3
                append!(event_PD_high_e_score, parameters.s_grid[panel_score[hh_i,period_i-1:period_i+1]])
            end
        end
    end
end

event_PD_high_e_score = reshape(event_PD_high_e_score, (3,:))
event_PD_high_e_score_mean = sum(event_PD_high_e_score,dims=2) ./ size(event_PD_high_e_score)[2]

# Effect of partial default on type score conditional on HHs with both bank and payday loans
event_PD_both_loans = findall(panel_default .== 2)
event_PD_both_loans_score = []

for event_i in 1:length(event_PD_both_loans)
    hh_i = event_PD_both_loans[event_i][1]
    period_i = event_PD_both_loans[event_i][2]

    if 1 < period_i < num_periods-burn_in-1
        if all(shock_ρ[hh_i,period_i-1:period_i+1] .== 1)
            if (panel_asset[hh_i,period_i] .< findfirst(parameters.asset_ind[:,1] .== parameters.a_ind_zero)) .& (isodd.(panel_asset[hh_i,period_i]))
                append!(event_PD_both_loans_score, parameters.s_grid[panel_score[hh_i,period_i-1:period_i+1]])
            end
        end
    end
end

event_PD_both_loans_score = reshape(event_PD_both_loans_score, (3,:))
event_PD_both_loans_score_mean = sum(event_PD_both_loans_score,dims=2) ./ size(event_PD_both_loans_score)[2]

# Effect of full default on type score
event_FD = findall(panel_default .== 1)
event_FD_score = []

for event_i in 1:length(event_FD)
    hh_i = event_FD[event_i][1]
    period_i = event_FD[event_i][2]

    if 1 < period_i < num_periods-burn_in-1
        if all(shock_ρ[hh_i,period_i-1:period_i+1] .== 1)
            append!(event_FD_score, parameters.s_grid[panel_score[hh_i,period_i-1:period_i+1]])
        end
    end
end

event_FD_score = reshape(event_FD_score, (3,:))
event_FD_score_mean = sum(event_FD_score,dims=2) ./ size(event_FD_score)[2]

# Effect on type score of switching from saving to borrowing
event_switch_score = []
asset_zero = findlast(parameters.asset_ind[:,1] .== parameters.a_ind_zero)

for hh_i in 1:num_hh
    for period_i in 1:size(panel_asset)[2]-1
        if (panel_asset[hh_i,period_i] > asset_zero) & (panel_asset[hh_i,period_i+1] < asset_zero-1)
            if all(shock_ρ[hh_i,period_i:period_i+1] .== 1)
                append!(event_switch_score, parameters.s_grid[panel_score[hh_i,period_i:period_i+1]])
            end
        end
    end
end

event_switch_score = reshape(event_switch_score, (2,:))
event_switch_score_mean = sum(event_switch_score,dims=2) ./ size(event_switch_score)[2]

# Effect on type score of switching from high e to low e
event_switch_earnings_score = []

for hh_i in 1:num_hh
    for period_i in 1:size(panel_asset)[2]-1
        if (shock_e[hh_i,period_i] - shock_e[hh_i,period_i+1]) > 0
            if all(shock_ρ[hh_i,period_i:period_i+1] .== 1)
                append!(event_switch_earnings_score, parameters.s_grid[panel_score[hh_i,period_i:period_i+1]])
            end
        end
    end
end

event_switch_earnings_score = reshape(event_switch_earnings_score, (2,:))
event_switch_earnings_score_mean = sum(event_switch_earnings_score,dims=2) ./ size(event_switch_earnings_score)[2]

# Effect on type score using both loans to borrow vs. using only bank loans (cond. on total borrowing amount and HH state)
event_window = 1

#=
β_i_target = 2
z_i_target = 1
e_i_target = 1
# asset_i_target = parameters.a_ind_zero*parameters.b_size
asset_i_target = 99
s_i_target = 15
=#
e_i_target = 2
tb_i_target = parameters.a_ind_zero-2

# use both loans
a_both_i = tb_i_target + 1
asset_both_i = (a_both_i-1)*parameters.b_size + 1
event_both_loans = findall(panel_asset .== asset_both_i)

# use only bank loans
a_bank_i = tb_i_target
asset_bank_i = a_bank_i*parameters.b_size
event_bank_loans = findall(panel_asset .== asset_bank_i)

# event containers
event_both_loans_score = []
event_bank_loans_score = []

for event_i in 1:length(event_both_loans)

    hh_i = event_both_loans[event_i][1]
    period_i = event_both_loans[event_i][2]
    period_i = period_i - 1

    if event_window < period_i < (period_all-event_window)
        #=
        β_i = shock_β[hh_i,period_i-1]
        z_i = shock_z[hh_i,period_i-1]
        e_i = shock_e[hh_i,period_i-1]
        asset_i = panel_asset[hh_i,period_i-1]
        s_i = panel_score[hh_i,period_i-1]
        =#
        # if (β_i == β_i_target) && (z_i == z_i_target) && (e_i == e_i_target) && (asset_i == asset_i_target) && (s_i == s_i_target)  && all(shock_ρ[hh_i,(period_i-event_window):(period_i+event_window)] .== 1)

        e_i = shock_e[hh_i,period_i]

        if (e_i == e_i_target) && all(shock_ρ[hh_i,(period_i-event_window):(period_i+event_window)] .== 1)
            append!(event_both_loans_score, parameters.s_grid[panel_score[hh_i,(period_i-event_window):(period_i+event_window)]])
        end
    end
end

event_both_loans_score = reshape(event_both_loans_score,2*event_window+1,:)

event_both_loans_score_ave = sum(event_both_loans_score,dims=2)/size(event_both_loans_score)[2]

for event_i in 1:length(event_bank_loans)
    hh_i = event_bank_loans[event_i][1]
    period_i = event_bank_loans[event_i][2]
    period_i = period_i - 1

    if event_window < period_i < (period_all-event_window)

        #=
        β_i = shock_β[hh_i,period_i-1]
        z_i = shock_z[hh_i,period_i-1]
        e_i = shock_e[hh_i,period_i-1]
        asset_i = panel_asset[hh_i,period_i-1]
        s_i = panel_score[hh_i,period_i-1]
        =#

        # if (β_i == β_i_target) && (z_i == z_i_target) && (e_i == e_i_target) && (asset_i == asset_i_target) && (s_i == s_i_target) && all(shock_ρ[hh_i,(period_i-event_window):(period_i+event_window)] .== 1)

        e_i = shock_e[hh_i,period_i]

        if (e_i == e_i_target) && all(shock_ρ[hh_i,(period_i-event_window):(period_i+event_window)] .== 1)
            append!(event_bank_loans_score, parameters.s_grid[panel_score[hh_i,(period_i-event_window):(period_i+event_window)]])
        end
    end
end

event_bank_loans_score = reshape(event_bank_loans_score,2*event_window+1,:)

event_bank_loans_score_ave = sum(event_bank_loans_score,dims=2)/size(event_bank_loans_score)[2]

################################################################################
# Across all possible tb_i

#e_i_target = 2

# event containers
event_both_loans_score = []
event_bank_loans_score = []

for tb_i_target in 1:(parameters.a_size_neg-1)

    # use both loans
    a_both_i = tb_i_target + 1
    asset_both_i = (a_both_i-1)*parameters.b_size + 1
    event_both_loans = findall(panel_asset .== asset_both_i)

    for event_i in 1:length(event_both_loans)

        hh_i = event_both_loans[event_i][1]
        period_i = event_both_loans[event_i][2]
        period_i = period_i - 1

        if event_window < period_i < (period_all-event_window)
            #=
            β_i = shock_β[hh_i,period_i-1]
            z_i = shock_z[hh_i,period_i-1]
            e_i = shock_e[hh_i,period_i-1]
            asset_i = panel_asset[hh_i,period_i-1]
            s_i = panel_score[hh_i,period_i-1]
            =#
            # if (β_i == β_i_target) && (z_i == z_i_target) && (e_i == e_i_target) && (asset_i == asset_i_target) && (s_i == s_i_target)  && all(shock_ρ[hh_i,(period_i-event_window):(period_i+event_window)] .== 1)

            # e_i = shock_e[hh_i,period_i]

            if all(shock_ρ[hh_i,(period_i-event_window):(period_i+event_window)] .== 1)
                append!(event_both_loans_score, parameters.s_grid[panel_score[hh_i,(period_i-event_window):(period_i+event_window)]])
            end
        end
    end
end

event_both_loans_score = reshape(event_both_loans_score,2*event_window+1,:)

event_both_loans_score_ave = sum(event_both_loans_score,dims=2)/size(event_both_loans_score)[2]

for tb_i_target in 1:(parameters.a_size_neg-1)

    # use only bank loans
    a_bank_i = tb_i_target
    asset_bank_i = a_bank_i*parameters.b_size
    event_bank_loans = findall(panel_asset .== asset_bank_i)

    for event_i in 1:length(event_bank_loans)

        hh_i = event_bank_loans[event_i][1]
        period_i = event_bank_loans[event_i][2]
        period_i = period_i - 1

        if event_window < period_i < (period_all-event_window)

            #=
            β_i = shock_β[hh_i,period_i-1]
            z_i = shock_z[hh_i,period_i-1]
            e_i = shock_e[hh_i,period_i-1]
            asset_i = panel_asset[hh_i,period_i-1]
            s_i = panel_score[hh_i,period_i-1]
            =#

            # if (β_i == β_i_target) && (z_i == z_i_target) && (e_i == e_i_target) && (asset_i == asset_i_target) && (s_i == s_i_target) && all(shock_ρ[hh_i,(period_i-event_window):(period_i+event_window)] .== 1)

            # e_i = shock_e[hh_i,period_i]

            if all(shock_ρ[hh_i,(period_i-event_window):(period_i+event_window)] .== 1)
                append!(event_bank_loans_score, parameters.s_grid[panel_score[hh_i,(period_i-event_window):(period_i+event_window)]])
            end
        end
    end
end

event_bank_loans_score = reshape(event_bank_loans_score,2*event_window+1,:)

event_bank_loans_score_ave = sum(event_bank_loans_score,dims=2)/size(event_bank_loans_score)[2]


################################################################################

using Plots
plot(event_both_loans_score_ave, lc = :blue, legend = :topleft)
plot!(event_bank_loans_score_ave, lc = :red)

mid_point = event_window+1

both_normalized = ((event_both_loans_score_ave.-event_both_loans_score_ave[mid_point])./event_both_loans_score_ave[mid_point])*100
bank_normalized = ((event_bank_loans_score_ave.-event_bank_loans_score_ave[mid_point])./event_bank_loans_score_ave[mid_point])*100
plot(both_normalized, lc = :blue, legend=:topleft)
plot!(bank_normalized, lc = :red)

both_normalized = ((event_both_loans_score_ave.-event_both_loans_score_ave[mid_point]))
bank_normalized = ((event_bank_loans_score_ave.-event_bank_loans_score_ave[mid_point]))
plot(both_normalized, lc = :blue, label = "both", legend=:topleft)
plot!(bank_normalized, lc = :red, label = "bank")








# reference
# σ = ones(action_size, β_size, z_size, e_size, asset_size, s_size) ./ action_size
# Q_s = zeros(s_size, action_a_size, e_size, a_size, s_size)
# μ = ones(β_size, z_size, e_size, asset_size, s_size)

# containers
# effect_both = zeros(parameters.a_size_neg-1,parameters.β_size,parameters.z_size,parameters.e_size,parameters.asset_size,parameters.s_size)
# effect_bank = zeros(parameters.a_size_neg-1,parameters.β_size,parameters.z_size,parameters.e_size,parameters.asset_size,parameters.s_size)
effect_diff = zeros(parameters.a_size_neg-1,parameters.β_size,parameters.z_size,parameters.e_size,parameters.asset_size,parameters.s_size)

test_num = 0.0
test_den = 0.0

test_2_num = 0.0
test_2_den = 0.0

for s_i in 1:parameters.s_size
    for asset_i in 1:parameters.asset_size, e_i in 1:parameters.e_size, z_i in 1:parameters.z_size, β_i in 1:parameters.β_size

        @inbounds @views a_i, b_i = parameters.asset_ind[asset_i,:]

        for tb_i in 1:(parameters.a_size_neg-1)
        # for tb_i in (parameters.a_size_neg-2):(parameters.a_size_neg-1)

            # use both loans
            a_p_i_both = tb_i + 1
            action_a_i_both = 1 + a_p_i_both
            action_i_both = 1 + parameters.a_size + (a_p_i_both-1)*parameters.b_size + 1
            @inbounds @views s_p_both = sum(parameters.s_grid.*Q_s[:,action_a_i_both,e_i,a_i,s_i])
            # effect_both[tb_i,β_i,z_i,e_i,asset_i,s_i] = s_p_both

            # use only bank loans
            a_p_i_bank = tb_i
            action_a_i_bank = 1 + a_p_i_bank
            action_i_bank = 1 + parameters.a_size + a_p_i_bank*parameters.b_size
            @inbounds @views s_p_bank = sum(parameters.s_grid.*Q_s[:,action_a_i_bank,e_i,a_i,s_i])
            # effect_bank[tb_i,β_i,z_i,e_i,asset_i,s_i] = s_p_bank

            if (σ[action_i_both,β_i,z_i,e_i,asset_i,s_i] > 0.0) && (σ[action_i_bank,β_i,z_i,e_i,asset_i,s_i] > 0.0)
                @inbounds effect_diff[tb_i,β_i,z_i,e_i,asset_i,s_i] = ((s_p_both-s_p_bank)/s_p_bank)*100

                test_num += ((s_p_both-s_p_bank)/s_p_bank)*σ[action_i_bank,β_i,z_i,e_i,asset_i,s_i]*μ[β_i,z_i,e_i,asset_i,s_i]
                test_den += σ[action_i_bank,β_i,z_i,e_i,asset_i,s_i]*μ[β_i,z_i,e_i,asset_i,s_i]

                test_2_num += ((s_p_both-s_p_bank)/s_p_bank)
                test_2_den += 1
            end
        end
    end
end

test_num/test_den*100

test_2_num/test_2_den*100

effect_tb = zeros(parameters.a_size_neg-1)

for tb_i in 1:(parameters.a_size_neg-1)
    effect_diff_tb = effect_diff[tb_i,:,:,:,:,:]
    effect_tb[tb_i] = sum(effect_diff_tb .* μ)
end

for s_i in 1:parameters.s_size
    for asset_i in 1:parameters.asset_size, e_i in 1:parameters.e_size, z_i in 1:parameters.z_size, β_i in 1:parameters.β_size

        for tb_i in 1:(parameters.a_size_neg-1)

        end
    end
end
=#
