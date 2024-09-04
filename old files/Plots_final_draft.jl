# Plots
using Plots
using LaTeXStrings
using Measures

# using Interpolations
using Dierckx

path = "C:/Users/jsun/Dropbox/Bankruptcy-Family/Figures/Final_Draft/Julia/"
# path = "C:/Users/JanSun/Dropbox/Bankruptcy-Family/Figures/Final_Draft/Julia/"
path = "C:/Users/jsun/Dropbox/Bankruptcy-Family/Results/Current/Calibration-14/Singles_have_couples_expenses_no_scale/"

a_grid_plot = round.(parameters.a_grid, digits=2)
z_grid_plot = round.(parameters.z_grid, digits=2)
η_grid_plot = round.(parameters.η_grid, digits=2)

############################
#       Event Studies      #
############################

x = [-2, -1, 0, 1, 2]

# Divorce event
# Consumption response
plot_divorce_event_consumption = plot(box = :on, size = [800, 500],
                xtickfont = font(14, "Computer Modern", :black),
                ytickfont = font(14, "Computer Modern", :black),
                legendfont = font(14, "Computer Modern", :black),
                guidefont = font(16, "Computer Modern", :black),
                titlefont = font(18, "Computer Modern", :black),
                margin = 4mm
                )

# plot_divorce_event_default = plot!(x,divorce_event_default, lw = 3, lc= :blue, xlabel = "Event Time", title=L" \textrm{Fraction\ of\ Defaulters\ Around\ Divorce\ Event}", ylims = (0.0, 0.06), xticks = (-2:1:2))
plot_divorce_event_default = plot!(x,divorce_event_default, lw = 3, lc= :blue, xlabel = "Event Time", ylims = (0.0, 0.06), xticks = (-2:1:2), legend=false)

savefig(plot_divorce_event_default, string(path,"plot_divorce_event_default.pdf"))

# Welfare

x_vals = 1.0:0.01:9.0

x = [0.1,0.3,0.5,0.7,0.9]

plot_x = collect(x_vals*0.1)

# Baseline
# Single women
y = [-13.3161404952082, -13.30453306, -13.30365684, -13.31288513, -13.32318993]

spl = Spline1D(x,y)

plot_y = spl(plot_x)

plot_welfare_base_S_women = plot(box = :on, size = [800, 500],
                xtickfont = font(14, "Computer Modern", :black),
                ytickfont = font(14, "Computer Modern", :black),
                legendfont = font(14, "Computer Modern", :black),
                guidefont = font(16, "Computer Modern", :black),
                titlefont = font(18, "Computer Modern", :black),
                margin = 4mm
                )

plot_welfare_base_S_women = plot!(plot_x,plot_y,lw = 3, lc= :blue, xlabel = L"\textrm{Garnishment rate }\phi", ylabel = "W", legend=:topright, label = "", linestyle=:dash, xticks = [0.1,0.3,0.5,0.7,0.9])

savefig(plot_welfare_base_S_women, string(path,"Welfare_S_women.pdf"))

# Single men
y = [-13.6789417, -13.67420965, -13.67786226, -13.70389961, -13.73971677]

spl = Spline1D(x,y)

plot_y = spl(plot_x)

plot_welfare_base_S_men = plot(box = :on, size = [800, 500],
                xtickfont = font(14, "Computer Modern", :black),
                ytickfont = font(14, "Computer Modern", :black),
                legendfont = font(14, "Computer Modern", :black),
                guidefont = font(16, "Computer Modern", :black),
                titlefont = font(18, "Computer Modern", :black),
                margin = 4mm
                )

plot_welfare_base_S_men = plot!(plot_x,plot_y,lw = 3, lc= :blue, xlabel = L"\textrm{Garnishment rate }\phi", ylabel = "W", legend=:topright, label = "", linestyle=:dash, xticks = [0.1,0.3,0.5,0.7,0.9])

savefig(plot_welfare_base_S_men, string(path,"Welfare_S_men.pdf"))

# Couples
y = [-22.56786519, -22.61603945, -22.6362812, -22.6397196, -22.64388494]

spl = Spline1D(x,y)

plot_y = spl(plot_x)

plot_welfare_base_C = plot(box = :on, size = [800, 500],
                xtickfont = font(14, "Computer Modern", :black),
                ytickfont = font(14, "Computer Modern", :black),
                legendfont = font(14, "Computer Modern", :black),
                guidefont = font(16, "Computer Modern", :black),
                titlefont = font(18, "Computer Modern", :black),
                margin = 4mm
                )

plot_welfare_base_C = plot!(plot_x,plot_y,lw = 3, lc= :blue, xlabel = L"\textrm{Garnishment rate }\phi", ylabel = "W", legend=:topright, label = "", linestyle=:dash, xticks = [0.1,0.3,0.5,0.7,0.9])

savefig(plot_welfare_base_C, string(path,"Welfare_C.pdf"))

# Women within couples
y = [-11.28888273, -11.34690837, -11.34956315, -11.34655405, -11.34676409]

spl = Spline1D(x,y,k=2)

plot_y = spl(plot_x)

plot_welfare_base_C_women = plot(box = :on, size = [800, 500],
                xtickfont = font(14, "Computer Modern", :black),
                ytickfont = font(14, "Computer Modern", :black),
                legendfont = font(14, "Computer Modern", :black),
                guidefont = font(16, "Computer Modern", :black),
                titlefont = font(18, "Computer Modern", :black),
                margin = 4mm
                )

plot_welfare_base_C_women = plot!(plot_x,plot_y,lw = 3, lc= :blue, xlabel = L"\textrm{Garnishment\ rate\ }\phi", ylabel = "W", legend=:topright, label = "", linestyle=:dash, xticks = [0.1,0.3,0.5,0.7,0.9])

savefig(plot_welfare_base_C_women, string(path,"Welfare_C_women.pdf"))

# Labor for women within couples
y = [0.730035112, 0.733034362, 0.734254963, 0.735199806, 0.734810001]

spl = Spline1D(x,y)

plot_y = spl(plot_x)

plot_labor_base_C_women = plot(box = :on, size = [800, 500],
                xtickfont = font(14, "Computer Modern", :black),
                ytickfont = font(14, "Computer Modern", :black),
                legendfont = font(14, "Computer Modern", :black),
                guidefont = font(16, "Computer Modern", :black),
                titlefont = font(18, "Computer Modern", :black),
                margin = 4mm
                )

plot_labor_base_C_women = plot!(plot_x,plot_y,lw = 3, lc= :blue, xlabel = L"\textrm{Garnishment\ rate\ }\phi", ylabel = "Fraction of full time", legend=:topright, label = "", linestyle=:dash, xticks = [0.1,0.3,0.5,0.7,0.9])

savefig(plot_labor_base_C_women, string(path,"Labor_C_women.pdf"))

# Men within couples
y = [-11.27898247, -11.26913107, -11.28671805, -11.29316556, -11.29712085]

spl = Spline1D(x,y,k=2)

plot_y = spl(plot_x)

plot_welfare_base_C_men = plot(box = :on, size = [800, 500],
                xtickfont = font(14, "Computer Modern", :black),
                ytickfont = font(14, "Computer Modern", :black),
                legendfont = font(14, "Computer Modern", :black),
                guidefont = font(16, "Computer Modern", :black),
                titlefont = font(18, "Computer Modern", :black),
                margin = 4mm
                )

plot_welfare_base_C_men = plot!(plot_x,plot_y,lw = 3, lc= :blue, xlabel = L"\textrm{Garnishment\ rate\ }\phi", ylabel = "W", legend=:topright, label = "", linestyle=:dash, xticks = [0.1,0.3,0.5,0.7,0.9])

savefig(plot_welfare_base_C_men, string(path,"Welfare_C_men.pdf"))

# Labor for men within couples
y = [0.741993512, 0.736637237, 0.737258406, 0.738213173, 0.737806772]

spl = Spline1D(x,y)

plot_y = spl(plot_x)

plot_labor_base_C_men = plot(box = :on, size = [800, 500],
                xtickfont = font(14, "Computer Modern", :black),
                ytickfont = font(14, "Computer Modern", :black),
                legendfont = font(14, "Computer Modern", :black),
                guidefont = font(16, "Computer Modern", :black),
                titlefont = font(18, "Computer Modern", :black),
                margin = 4mm
                )

plot_labor_base_C_men = plot!(plot_x,plot_y,lw = 3, lc= :blue, xlabel = L"\textrm{Garnishment\ rate\ }\phi", ylabel = "Fraction of full time", legend=:topright, label = "", linestyle=:dash, xticks = [0.1,0.3,0.5,0.7,0.9])

savefig(plot_labor_base_C_men, string(path,"Labor_C_men.pdf"))


# Only Income Differences
# Single women
y = [-13.24153229, -13.19225195, -13.14800826, -13.11204357, -13.10788054]

# spl = Spline1D(x,y,k=2,s=0.001)
spl = Spline1D(x,y,k=1)

plot_y = spl(plot_x)

plot_welfare_only_income_S_women = plot(box = :on, size = [800, 500],
                xtickfont = font(14, "Computer Modern", :black),
                ytickfont = font(14, "Computer Modern", :black),
                legendfont = font(14, "Computer Modern", :black),
                guidefont = font(16, "Computer Modern", :black),
                titlefont = font(18, "Computer Modern", :black),
                margin = 4mm
                )

plot_welfare_only_income_S_women = plot!(plot_x,plot_y,lw = 3, lc= :blue, xlabel = L"\textrm{Garnishment rate }\phi", ylabel = "W", legend=:topright, label = "", linestyle=:dash, xticks = [0.1,0.3,0.5,0.7,0.9])

savefig(plot_welfare_only_income_S_women, string(path,"Welfare_S_women_only_income.pdf"))

# Single men
y = [-13.60598866, -13.55579285, -13.50781712, -13.47520117, -13.47263076]

spl = Spline1D(x,y,k=1)

plot_y = spl(plot_x)

plot_welfare_only_income_S_men = plot(box = :on, size = [800, 500],
                xtickfont = font(14, "Computer Modern", :black),
                ytickfont = font(14, "Computer Modern", :black),
                legendfont = font(14, "Computer Modern", :black),
                guidefont = font(16, "Computer Modern", :black),
                titlefont = font(18, "Computer Modern", :black),
                margin = 4mm
                )

plot_welfare_only_income_S_men = plot!(plot_x,plot_y,lw = 3, lc= :blue, xlabel = L"\textrm{Garnishment rate }\phi", ylabel = "W", legend=:topright, label = "", linestyle=:dash, xticks = [0.1,0.3,0.5,0.7,0.9])

savefig(plot_welfare_only_income_S_men, string(path,"Welfare_S_men_only_income.pdf"))

# Couples
y = [-24.25559763, -24.20871147, -24.16996785, -24.14453478, -24.13994854]

spl = Spline1D(x,y,k=1)

plot_y = spl(plot_x)

plot_welfare_only_income_C = plot(box = :on, size = [800, 500],
                xtickfont = font(14, "Computer Modern", :black),
                ytickfont = font(14, "Computer Modern", :black),
                legendfont = font(14, "Computer Modern", :black),
                guidefont = font(16, "Computer Modern", :black),
                titlefont = font(18, "Computer Modern", :black),
                margin = 4mm
                )

plot_welfare_only_income_C = plot!(plot_x,plot_y,lw = 3, lc= :blue, xlabel = L"\textrm{Garnishment rate }\phi", ylabel = "W", legend=:topright, label = "", linestyle=:dash, xticks = [0.1,0.3,0.5,0.7,0.9])

savefig(plot_welfare_only_income_C, string(path,"Welfare_C_only_income.pdf"))

# Intra-HH insurance channel
# Single women
y = [-20.7966499, -20.28860826, -20.6493015, -20.54474133, -20.43897019]

# spl = Spline1D(x,y,k=2,s=0.001)
spl = Spline1D(x,y,k=2)

plot_y = spl(plot_x)

plot_welfare_intra_hh_S_women = plot(box = :on, size = [800, 500],
                xtickfont = font(14, "Computer Modern", :black),
                ytickfont = font(14, "Computer Modern", :black),
                legendfont = font(14, "Computer Modern", :black),
                guidefont = font(16, "Computer Modern", :black),
                titlefont = font(18, "Computer Modern", :black),
                margin = 4mm
                )

plot_welfare_intra_hh_S_women = plot!(plot_x,plot_y,lw = 3, lc= :blue, xlabel = L"\textrm{Garnishment rate }\phi", ylabel = "W", legend=:topright, label = "", linestyle=:dash, xticks = [0.1,0.3,0.5,0.7,0.9])

savefig(plot_welfare_intra_hh_S_women, string(path,"Welfare_S_women_fifteen.pdf"))

# Single men
y = [-29.79779506, -28.83196408, -29.49946141, -29.85327159, -29.85327159]

# spl = Spline1D(x,y,k=2,s=0.001)
spl = Spline1D(x,y,k=2)

plot_y = spl(plot_x)

plot_welfare_intra_hh_S_men = plot(box = :on, size = [800, 500],
                xtickfont = font(14, "Computer Modern", :black),
                ytickfont = font(14, "Computer Modern", :black),
                legendfont = font(14, "Computer Modern", :black),
                guidefont = font(16, "Computer Modern", :black),
                titlefont = font(18, "Computer Modern", :black),
                margin = 4mm
                )

plot_welfare_intra_hh_S_men = plot!(plot_x,plot_y,lw = 3, lc= :blue, xlabel = L"\textrm{Garnishment rate }\phi", ylabel = "W", legend=:topright, label = "", linestyle=:dash, xticks = [0.1,0.3,0.5,0.7,0.9])

savefig(plot_welfare_intra_hh_S_men, string(path,"Welfare_S_men_fifteen.pdf"))

# Couples
y = [-22.59455833, -22.40761239, -22.26928436, -22.30295795, -22.05520977]

spl = Spline1D(x,y,k=2,s=0.001)
# spl = Spline1D(x,y,k=2)

plot_y = spl(plot_x)

plot_welfare_intra_hh_C = plot(box = :on, size = [800, 500],
                xtickfont = font(14, "Computer Modern", :black),
                ytickfont = font(14, "Computer Modern", :black),
                legendfont = font(14, "Computer Modern", :black),
                guidefont = font(16, "Computer Modern", :black),
                titlefont = font(18, "Computer Modern", :black),
                margin = 4mm
                )

plot_welfare_intra_hh_C = plot!(plot_x,plot_y,lw = 3, lc= :blue, xlabel = L"\textrm{Garnishment rate }\phi", ylabel = "W", legend=:topright, label = "", linestyle=:dash, xticks = [0.1,0.3,0.5,0.7,0.9])

savefig(plot_welfare_intra_hh_C, string(path,"Welfare_C_fifteen.pdf"))

# No divorce
y = [-21.80825915, -21.84214687, -21.84969626, -21.83567812, -21.81596452]

# spl = Spline1D(x,y,k=2,s=0.001)
spl = Spline1D(x,y,k=2)

plot_y = spl(plot_x)

plot_welfare_no_divorce = plot(box = :on, size = [800, 500],
                xtickfont = font(14, "Computer Modern", :black),
                ytickfont = font(14, "Computer Modern", :black),
                legendfont = font(14, "Computer Modern", :black),
                guidefont = font(16, "Computer Modern", :black),
                titlefont = font(18, "Computer Modern", :black),
                margin = 4mm
                )

plot_welfare_no_divorce = plot!(plot_x,plot_y,lw = 3, lc= :blue, xlabel = L"\textrm{Garnishment rate }\phi", ylabel = "W", legend=:topright, label = "", linestyle=:dash, xticks = [0.1,0.3,0.5,0.7,0.9])

savefig(plot_welfare_no_divorce, string(path,"Welfare_C_no_divorce.pdf"))

# No economies of scale
y = [-24.74659711, -24.79806022, -24.81767482, -24.81757162, -24.81881133]

# spl = Spline1D(x,y,k=2,s=0.001)
spl = Spline1D(x,y,k=2)

plot_y = spl(plot_x)

plot_welfare_no_scale = plot(box = :on, size = [800, 500],
                xtickfont = font(14, "Computer Modern", :black),
                ytickfont = font(14, "Computer Modern", :black),
                legendfont = font(14, "Computer Modern", :black),
                guidefont = font(16, "Computer Modern", :black),
                titlefont = font(18, "Computer Modern", :black),
                margin = 4mm
                )

plot_welfare_no_scale = plot!(plot_x,plot_y,lw = 3, lc= :blue, xlabel = L"\textrm{Garnishment rate }\phi", ylabel = "W", legend=:topright, label = "", linestyle=:dash, xticks = [0.1,0.3,0.5,0.7,0.9])

savefig(plot_welfare_no_scale, string(path,"Welfare_C_no_scale.pdf"))

# Single expenses
y = [-22.58555663, -22.63626953, -22.66032172, -22.66569042, -22.67450311]

# spl = Spline1D(x,y,k=2,s=0.001)
spl = Spline1D(x,y,k=2)

plot_y = spl(plot_x)

plot_welfare_single_expenses = plot(box = :on, size = [800, 500],
                xtickfont = font(14, "Computer Modern", :black),
                ytickfont = font(14, "Computer Modern", :black),
                legendfont = font(14, "Computer Modern", :black),
                guidefont = font(16, "Computer Modern", :black),
                titlefont = font(18, "Computer Modern", :black),
                margin = 4mm
                )

plot_welfare_single_expenses = plot!(plot_x,plot_y,lw = 3, lc= :blue, xlabel = L"\textrm{Garnishment rate }\phi", ylabel = "W", legend=:topright, label = "", linestyle=:dash, xticks = [0.1,0.3,0.5,0.7,0.9])

savefig(plot_welfare_single_expenses, string(path,"Welfare_C_single_expenses.pdf"))

# Only One Single expense
y = [-22.52826457, -22.54454458, -22.54963716, -22.54979068, -22.55989308]

spl = Spline1D(x,y,k=3,s=0.1)
# spl = Spline1D(x,y,k=2)

plot_y = spl(plot_x)

plot_welfare_only_one_single_expense = plot(box = :on, size = [800, 500],
                xtickfont = font(14, "Computer Modern", :black),
                ytickfont = font(14, "Computer Modern", :black),
                legendfont = font(14, "Computer Modern", :black),
                guidefont = font(16, "Computer Modern", :black),
                titlefont = font(18, "Computer Modern", :black),
                margin = 4mm
                )

plot_welfare_only_one_single_expense = plot!(plot_x,plot_y,lw = 3, lc= :blue, xlabel = L"\textrm{Garnishment rate }\phi", ylabel = "W", legend=:topright, label = "", linestyle=:dash, xticks = [0.1,0.3,0.5,0.7,0.9])

savefig(plot_welfare_only_one_single_expense, string(path,"Welfare_C_only_one_singles_expense.pdf"))

# Perfectly correlated expenses
y = [-22.54643413, -22.58418325, -22.61470812, -22.63764749, -22.65790236]

spl = Spline1D(x,y,k=3,s=0.1)
# spl = Spline1D(x,y,k=2)

plot_y = spl(plot_x)

plot_welfare_perfectly_correlated_expenses = plot(box = :on, size = [800, 500],
                xtickfont = font(14, "Computer Modern", :black),
                ytickfont = font(14, "Computer Modern", :black),
                legendfont = font(14, "Computer Modern", :black),
                guidefont = font(16, "Computer Modern", :black),
                titlefont = font(18, "Computer Modern", :black),
                margin = 4mm
                )

plot_welfare_perfectly_correlated_expenses = plot!(plot_x,plot_y,lw = 3, lc= :blue, xlabel = L"\textrm{Garnishment\ rate\ }\phi", ylabel = "W", legend=:topright, label = "", linestyle=:dash, xticks = [0.1,0.3,0.5,0.7,0.9])

savefig(plot_welfare_perfectly_correlated_expenses, string(path,"Welfare_C_perfectly_correlated_expenses.pdf"))

# No divorce, no scale
y = [-24.39220233, -24.43010521, -24.4385491, -24.42287002, -24.40082066]

# spl = Spline1D(x,y,k=3,s=0.1)
spl = Spline1D(x,y,k=2)

plot_y = spl(plot_x)

plot_welfare_no_divorce_no_scale = plot(box = :on, size = [800, 500],
                xtickfont = font(14, "Computer Modern", :black),
                ytickfont = font(14, "Computer Modern", :black),
                legendfont = font(14, "Computer Modern", :black),
                guidefont = font(16, "Computer Modern", :black),
                titlefont = font(18, "Computer Modern", :black),
                margin = 4mm
                )

plot_welfare_no_divorce_no_scale = plot!(plot_x,plot_y,lw = 3, lc= :blue, xlabel = L"\textrm{Garnishment rate }\phi", ylabel = "W", legend=:topright, label = "", linestyle=:dash, xticks = [0.1,0.3,0.5,0.7,0.9])

savefig(plot_welfare_no_divorce_no_scale, string(path,"Welfare_C_no_divorce_no_scale.pdf"))

# No scale, only one singles expense
y = [-24.70289773, -24.71792916, -24.71982144, -24.71737433, -24.7259493]

spl = Spline1D(x,y,k=3,s=0.1)
# spl = Spline1D(x,y,k=2)

plot_y = spl(plot_x)

plot_welfare_no_scale_only_one_singles_expense = plot(box = :on, size = [800, 500],
                xtickfont = font(14, "Computer Modern", :black),
                ytickfont = font(14, "Computer Modern", :black),
                legendfont = font(14, "Computer Modern", :black),
                guidefont = font(16, "Computer Modern", :black),
                titlefont = font(18, "Computer Modern", :black),
                margin = 4mm
                )

plot_welfare_no_scale_only_one_singles_expense = plot!(plot_x,plot_y,lw = 3, lc= :blue, xlabel = L"\textrm{Garnishment rate }\phi", ylabel = "W", legend=:topright, label = "", linestyle=:dash, xticks = [0.1,0.3,0.5,0.7,0.9])

savefig(plot_welfare_no_scale_only_one_singles_expense, string(path,"Welfare_C_no_scale_only_one_singles_expense.pdf"))

# No divorce, only one singles expense
y = [-21.76334634, -21.76076928, -21.75188705, -21.73643283, -21.72768984]

# spl = Spline1D(x,y,k=3,s=0.1)
spl = Spline1D(x,y,k=2)

plot_y = spl(plot_x)

plot_welfare_no_divorce_only_one_singles_expense = plot(box = :on, size = [800, 500],
                xtickfont = font(14, "Computer Modern", :black),
                ytickfont = font(14, "Computer Modern", :black),
                legendfont = font(14, "Computer Modern", :black),
                guidefont = font(16, "Computer Modern", :black),
                titlefont = font(18, "Computer Modern", :black),
                margin = 4mm
                )

plot_welfare_no_divorce_only_one_singles_expense = plot!(plot_x,plot_y,lw = 3, lc= :blue, xlabel = L"\textrm{Garnishment rate }\phi", ylabel = "W", legend=:topright, label = "", linestyle=:dash, xticks = [0.1,0.3,0.5,0.7,0.9])

savefig(plot_welfare_no_divorce_only_one_singles_expense, string(path,"Welfare_C_no_divorce_only_one_singles_expense.pdf"))

# Fixed bankruptcy cost
x_vals = 0.01:0.001:0.3

x = [0.01,0.05,0.1,0.15,0.2,0.25,0.3]

plot_x = collect(x_vals)

# Single women
y = [-13.31094959, -13.3069391, -13.30596315, -13.30544416, -13.30659503, -13.31362582, -13.31980922]

# spl = Spline1D(x,y,k=3,s=0.1)
spl = Spline1D(x,y,k=2)

plot_y = spl(plot_x)

plot_welfare_fixed_cost_S_women = plot(box = :on, size = [800, 500],
                xtickfont = font(14, "Computer Modern", :black),
                ytickfont = font(14, "Computer Modern", :black),
                legendfont = font(14, "Computer Modern", :black),
                guidefont = font(16, "Computer Modern", :black),
                titlefont = font(18, "Computer Modern", :black),
                margin = 4mm
                )

plot_welfare_fixed_cost_S_women = plot!(plot_x,plot_y,lw = 3, lc= :blue, xlabel = L"\textrm{Fixed bankruptcy cost }\phi", ylabel = "W", legend=:topright, label = "", linestyle=:dash, xticks = [0.01,0.1,0.2,0.3])

savefig(plot_welfare_fixed_cost_S_women, string(path,"Welfare_S_women_fixed_cost.pdf"))

# Single men
y = [-13.66893955, -13.66699434, -13.66486833, -13.67576063, -13.68933283, -13.70968865, -13.7414624]

# spl = Spline1D(x,y,k=3,s=0.1)
spl = Spline1D(x,y,k=2)

plot_y = spl(plot_x)

plot_welfare_fixed_cost_S_men = plot(box = :on, size = [800, 500],
                xtickfont = font(14, "Computer Modern", :black),
                ytickfont = font(14, "Computer Modern", :black),
                legendfont = font(14, "Computer Modern", :black),
                guidefont = font(16, "Computer Modern", :black),
                titlefont = font(18, "Computer Modern", :black),
                margin = 4mm
                )

plot_welfare_fixed_cost_S_men = plot!(plot_x,plot_y,lw = 3, lc= :blue, xlabel = L"\textrm{Fixed bankruptcy cost }\phi", ylabel = "W", legend=:topright, label = "", linestyle=:dash, xticks = [0.01,0.1,0.2,0.3])

savefig(plot_welfare_fixed_cost_S_men, string(path,"Welfare_S_men_fixed_cost.pdf"))

# Couples
y = [-22.52957497, -22.54866526, -22.57061479, -22.58443415, -22.59968663, -22.61283783, -22.62762775]

# spl = Spline1D(x,y,k=3,s=0.1)
spl = Spline1D(x,y,k=2)

plot_y = spl(plot_x)

plot_welfare_fixed_cost_C = plot(box = :on, size = [800, 500],
                xtickfont = font(14, "Computer Modern", :black),
                ytickfont = font(14, "Computer Modern", :black),
                legendfont = font(14, "Computer Modern", :black),
                guidefont = font(16, "Computer Modern", :black),
                titlefont = font(18, "Computer Modern", :black),
                margin = 4mm
                )

plot_welfare_fixed_cost_C = plot!(plot_x,plot_y,lw = 3, lc= :blue, xlabel = L"\textrm{Fixed bankruptcy cost }\phi", ylabel = "W", legend=:topright, label = "", linestyle=:dash, xticks = [0.01,0.1,0.2,0.3])

savefig(plot_welfare_fixed_cost_C, string(path,"Welfare_C_fixed_cost.pdf"))


# Fraction of Defaulter cond. on Receiving Expense Shock
x_vals = 1.0:0.01:9.0

x = [0.1,0.2,0.3,0.5,0.7,0.9]

plot_x = collect(x_vals*0.1)
# Singles
# Any shock
S_any_shock = [44.04515579, 37.8227187, 34.21343946, 19.41249433, 11.12547818, 2.448534234]

# spl = Spline1D(x,y,k=3,s=0.1)
spl_S_any_shock = Spline1D(x,S_any_shock,k=1)

plot_S_any_shock = spl_S_any_shock(plot_x)

# Small shock
S_small_shock = [37.49896278, 30.64751662, 26.44276066, 9.364783775, 4.119148678, 0.262822732]

spl_S_small_shock = Spline1D(x,S_small_shock,k=1)

plot_S_small_shock = spl_S_small_shock(plot_x)

# Large shock
S_large_shock = [53.80588059, 48.54632195, 45.82108109, 34.4228749, 21.58107586, 5.709522675]

spl_S_large_shock = Spline1D(x,S_large_shock,k=1)

plot_S_large_shock = spl_S_large_shock(plot_x)

plot_cond_default_S = plot(box = :on, size = [800, 500],
                xtickfont = font(14, "Computer Modern", :black),
                ytickfont = font(14, "Computer Modern", :black),
                legendfont = font(14, "Computer Modern", :black),
                guidefont = font(16, "Computer Modern", :black),
                titlefont = font(18, "Computer Modern", :black),
                margin = 4mm
                )

plot_cond_default_S = plot!(plot_x,plot_S_any_shock,lw = 3, lc= :blue, xlabel = L"\textrm{Garnishment rate }\phi", ylabel = "%", legend=:topright, label = "Any expense shock", xticks = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])

plot_cond_default_S = plot!(plot_x,plot_S_small_shock, lw=3, lc= :red,label = "Small shock", linestyle=:dash)

plot_cond_default_S = plot!(plot_x,plot_S_large_shock, lw=3, lc= :green,label = "Large shock", linestyle=:dashdot)

savefig(plot_cond_default_S, string(path,"S_default_cond_expense.pdf"))

# Couples
# Any shock
C_any_shock = [34.5701891, 21.07156299, 12.78241819, 3.488799938, 0.489023114, 0.006522827]

# spl = Spline1D(x,y,k=3,s=0.1)
spl_C_any_shock = Spline1D(x,C_any_shock,k=1)

plot_C_any_shock = spl_C_any_shock(plot_x)

# One small shock
C_one_small_shock = [31.64072068, 15.77980419, 8.074034331, 1.283399626, 0.038163738, 0]

# spl = Spline1D(x,y,k=3,s=0.1)
spl_C_one_small_shock = Spline1D(x,C_one_small_shock,k=1)

plot_C_one_small_shock = spl_C_one_small_shock(plot_x)

# Two small shocks
C_two_small_shock = [36.8170316, 26.48926718, 15.77778081, 4.269805415, 0.581509828, 0]

# spl = Spline1D(x,y,k=3,s=0.1)
spl_C_two_small_shock = Spline1D(x,C_two_small_shock,k=1)

plot_C_two_small_shock = spl_C_two_small_shock(plot_x)

# One large shock
C_one_large_shock = [38.63511515, 28.31169522, 19.17295275, 6.343566552, 1.019450036, 0.009389293]

# spl = Spline1D(x,y,k=3,s=0.1)
spl_C_one_large_shock = Spline1D(x,C_one_large_shock,k=1)

plot_C_one_large_shock = spl_C_one_large_shock(plot_x)

# One large, one small shock
C_one_large_one_small_shock = [40.18747989, 34.46562108, 25.63620869, 12.01892552, 2.993322838, 0.035273369]

# spl = Spline1D(x,y,k=3,s=0.1)
spl_C_one_large_one_small_shock = Spline1D(x,C_one_large_one_small_shock,k=1)

plot_C_one_large_one_small_shock = spl_C_one_large_one_small_shock(plot_x)

# Two large shocks
C_two_large_shock = [44.05323585, 38.77787653, 33.4261792, 20.18847939, 7.076466385, 0.599565156]

# spl = Spline1D(x,y,k=3,s=0.1)
spl_C_two_large_shock = Spline1D(x,C_two_large_shock,k=1)

plot_C_two_large_shock = spl_C_two_large_shock(plot_x)

plot_cond_default_C = plot(box = :on, size = [800, 500],
                xtickfont = font(14, "Computer Modern", :black),
                ytickfont = font(14, "Computer Modern", :black),
                legendfont = font(14, "Computer Modern", :black),
                guidefont = font(16, "Computer Modern", :black),
                titlefont = font(18, "Computer Modern", :black),
                margin = 4mm
                )

plot_cond_default_C = plot!(plot_x,plot_C_any_shock,lw = 3, lc= :blue, xlabel = L"\textrm{Garnishment rate }\phi", ylabel = "%", legend=:topright, label = "Any expense shock", xticks = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])

plot_cond_default_C = plot!(plot_x,plot_C_one_small_shock, lw=3, lc= :red,label = "One small shock")

plot_cond_default_C = plot!(plot_x,plot_C_two_small_shock, lw=3, lc= :red,label = "Two small shocks", linestyle=:dash)

plot_cond_default_C = plot!(plot_x,plot_C_one_large_shock, lw=3, lc= :green,label = "One large shock")

plot_cond_default_C = plot!(plot_x,plot_C_one_large_one_small_shock, lw=3, lc= :black,label = "One large, one small shock", linestyle=:dashdot)

plot_cond_default_C = plot!(plot_x,plot_C_two_large_shock, lw=3, lc= :green,label = "Two large shocks",linestyle=:dash)

savefig(plot_cond_default_C, string(path,"C_default_cond_expense.pdf"))

# Bankruptcy filing rates across age
x_vals = 23:0.1:65

x = [23,26,29,32,35,38,41,44,47,50,53,56,59,62,65]

plot_x = collect(x_vals)

# Singles
filings_age_S = [1.278993147, 1.725751186, 1.608400824, 1.554128528, 1.306129295, 0.956594144, 0.861527771, 0.889382757, 0.740942637, 0.73932525, 0.651087842, 0.698171754, 0.734113672, 0.479644894, 0.775806297]

filings_age_S = filings_age_S./filings_age_S[1] # re-normalized to 1 at start age

# spl = Spline1D(x,y,k=3,s=0.1)
spl_filings_age_S = Spline1D(x,filings_age_S,k=1)

plot_filings_age_S = spl_filings_age_S(plot_x)

# Couples
filings_age_C = [3.326299102, 1.264641761, 1.666331437, 1.801501353, 1.782640202, 1.118119952, 0.798624876, 0.89708668, 0.640688178, 0.559662308, 0.384103517, 0.310746848, 0.222968871, 0.114424987, 0.11215993]

filings_age_C = filings_age_C ./ filings_age_C[1]

# spl = Spline1D(x,y,k=3,s=0.1)
spl_filings_age_C = Spline1D(x,filings_age_C,k=1)

plot_filings_age_C = spl_filings_age_C(plot_x)

# Divorced
filings_age_div = [2.168091238, 2.200091458, 1.742072938, 1.66799966, 1.409872293, 0.897768008, 0.679995162, 0.606437707, 0.539711931, 0.599563978, 0.476964148, 0.528216245, 0.549069323, 0.350036669, 0.584109242]

filings_age_div = filings_age_div ./ filings_age_div[1]

# spl = Spline1D(x,y,k=3,s=0.1)
spl_filings_age_div = Spline1D(x,filings_age_div,k=1)

plot_filings_age_div = spl_filings_age_div(plot_x)

# Data
x = [25,30,35,40,45,50,55,60]

x_vals_alt = 25:0.1:60

plot_x_alt = collect(x_vals_alt)

filings_age_data = [1.25, 1.53, 1.44, 1.57, 1.45, 0.84, 0.91, 0.17]

filings_age_data = filings_age_data ./ filings_age_data[1]

# spl = Spline1D(x,y,k=3,s=0.1)
spl_filings_age_data = Spline1D(x,filings_age_data,k=1)

plot_filings_age_data = spl_filings_age_data(plot_x_alt)

plot_filings_age = plot(box = :on, size = [800, 500],
                xtickfont = font(14, "Computer Modern", :black),
                ytickfont = font(14, "Computer Modern", :black),
                legendfont = font(14, "Computer Modern", :black),
                guidefont = font(16, "Computer Modern", :black),
                titlefont = font(18, "Computer Modern", :black),
                margin = 4mm
                )

plot_filings_age = plot!(plot_x,plot_filings_age_S,lw = 3, lc= :blue, xlabel = "Age", legend=:topright, label = "Singles", xticks = [23,32,41,50,59])

plot_filings_age = plot!(plot_x,plot_filings_age_C, lw=3, lc= :red,label = "Couples", linestyle=:dash)

plot_filings_age = plot!(plot_x,plot_filings_age_div, lw=3, lc= :green,label = "Divorced", linestyle=:dot)

plot_filings_age = plot!(plot_x_alt,plot_filings_age_data, lw=3, lc= :black,label = "Data", linestyle=:dashdot)

savefig(plot_filings_age, string(path,"Bankruptcy-rate.pdf"))
savefig(plot_filings_age, string(path,"Bankruptcy-rate-alt.pdf"))

# Consumption lifecycle
x_vals = 20:0.1:65

x = [20,23,26,29,32,35,38,41,44,47,50,53,56,59,62,65]

plot_x = collect(x_vals)

# Singles
consumption_age_S = [0.830018668, 0.86540014, 0.905040637, 0.959407462, 1.005553039, 1.064550042, 1.114384468, 1.164807878, 1.209757291, 1.229115992, 1.226884258, 1.233250323, 1.230978012, 1.241594017, 1.25182748, 1.257394766]

# spl = Spline1D(x,y,k=3,s=0.1)
spl_consumption_age_S = Spline1D(x,consumption_age_S,k=1)

plot_consumption_age_S = spl_consumption_age_S(plot_x)

# Couples
consumption_age_C = [0.951611962, 0.985555119, 1.043127374, 1.083216462, 1.141576298, 1.18971804, 1.239625918, 1.290414023, 1.338107367, 1.376694337, 1.415769841, 1.446597136, 1.475574625, 1.495067508, 1.502352666, 1.517726156]

# spl = Spline1D(x,y,k=3,s=0.1)
spl_consumption_age_C = Spline1D(x,consumption_age_C,k=1)

plot_consumption_age_C = spl_consumption_age_C(plot_x)

# Divorced
x_vals_alt = 23:0.1:65

x_alt = [23,26,29,32,35,38,41,44,47,50,53,56,59,62,65]

plot_x_alt = collect(x_vals_alt)

consumption_age_div = [0.863319685, 0.886633921, 0.939181799, 0.981681372, 1.037741443, 1.08250391, 1.133080336, 1.178985228, 1.205519395, 1.217335109, 1.213119292, 1.201622785, 1.203424497, 1.208328841, 1.214458904]

# spl = Spline1D(x,y,k=3,s=0.1)
spl_consumption_age_div = Spline1D(x_alt,consumption_age_div,k=1)

plot_consumption_age_div = spl_consumption_age_div(plot_x_alt)

plot_consumption_age = plot(box = :on, size = [800, 500],
                xtickfont = font(14, "Computer Modern", :black),
                ytickfont = font(14, "Computer Modern", :black),
                legendfont = font(14, "Computer Modern", :black),
                guidefont = font(16, "Computer Modern", :black),
                titlefont = font(18, "Computer Modern", :black),
                margin = 4mm
                )

plot_consumption_age = plot!(plot_x,plot_consumption_age_S,lw = 3, lc= :blue, xlabel = "Age", legend=:topleft, label = "Singles", xticks = [20,29,38,47,56,65])

plot_consumption_age = plot!(plot_x,plot_consumption_age_C, lw=3, lc= :red,label = "Couples", linestyle=:dash)

plot_consumption_age = plot!(plot_x_alt,plot_consumption_age_div, lw=3, lc= :green,label = "Divorced", linestyle=:dashdot)

savefig(plot_consumption_age, string(path,"Consumption.pdf"))

# Consumption lifecycle, Couples have singles expenses counterfactual
x_vals = 20:0.1:65

x = [20,23,26,29,32,35,38,41,44,47,50,53,56,59,62,65]

plot_x = collect(x_vals)

# Singles
consumption_age_S = consumption_S_sim_ave_age

# spl = Spline1D(x,y,k=3,s=0.1)
spl_consumption_age_S = Spline1D(x,consumption_age_S,k=1)

plot_consumption_age_S = spl_consumption_age_S(plot_x)

# Couples
consumption_age_C = consumption_C_sim

# spl = Spline1D(x,y,k=3,s=0.1)
spl_consumption_age_C = Spline1D(x,consumption_age_C,k=1)

plot_consumption_age_C = spl_consumption_age_C(plot_x)

# Divorced
x_vals_alt = 23:0.1:65

x_alt = [23,26,29,32,35,38,41,44,47,50,53,56,59,62,65]

plot_x_alt = collect(x_vals_alt)

consumption_age_div = consumption_div_sim_ave_age[2:end]

# spl = Spline1D(x,y,k=3,s=0.1)
spl_consumption_age_div = Spline1D(x_alt,consumption_age_div,k=1)

plot_consumption_age_div = spl_consumption_age_div(plot_x_alt)

plot_consumption_age = plot(box = :on, size = [800, 500],
                xtickfont = font(14, "Computer Modern", :black),
                ytickfont = font(14, "Computer Modern", :black),
                legendfont = font(14, "Computer Modern", :black),
                guidefont = font(16, "Computer Modern", :black),
                titlefont = font(18, "Computer Modern", :black),
                margin = 4mm
                )

plot_consumption_age = plot!(plot_x,plot_consumption_age_S,lw = 3, lc= :blue, xlabel = "Age", legend=:topleft, label = "Singles", xticks = [20,29,38,47,56,65])

plot_consumption_age = plot!(plot_x,plot_consumption_age_C, lw=3, lc= :red,label = "Couples", linestyle=:dash)

plot_consumption_age = plot!(plot_x_alt,plot_consumption_age_div, lw=3, lc= :green,label = "Divorced", linestyle=:dashdot)

savefig(plot_consumption_age, string(path,"Consumption.pdf"))

# Consumption lifecycle, Single Male and Female separate
x_vals = 20:0.1:65

x = [20,23,26,29,32,35,38,41,44,47,50,53,56,59,62,65]

plot_x = collect(x_vals)

# Singles, Female
consumption_age_S_female = consumption_S_sim[:,1]

# spl = Spline1D(x,y,k=3,s=0.1)
spl_consumption_age_S_female = Spline1D(x,consumption_age_S_female,k=1)

plot_consumption_age_S_female = spl_consumption_age_S_female(plot_x)

# Singles, Male
consumption_age_S_male = consumption_S_sim[:,2]

# spl = Spline1D(x,y,k=3,s=0.1)
spl_consumption_age_S_male = Spline1D(x,consumption_age_S_male,k=1)

plot_consumption_age_S_male = spl_consumption_age_S_male(plot_x)

# Couples
consumption_age_C = consumption_C_sim

# spl = Spline1D(x,y,k=3,s=0.1)
spl_consumption_age_C = Spline1D(x,consumption_age_C,k=1)

plot_consumption_age_C = spl_consumption_age_C(plot_x)

# Divorced
# x_vals_alt = 23:0.1:65
#
# x_alt = [23,26,29,32,35,38,41,44,47,50,53,56,59,62,65]
#
# plot_x_alt = collect(x_vals_alt)
#
# consumption_age_div = consumption_div_sim_ave_age[2:end]
#
# # spl = Spline1D(x,y,k=3,s=0.1)
# spl_consumption_age_div = Spline1D(x_alt,consumption_age_div,k=1)
#
# plot_consumption_age_div = spl_consumption_age_div(plot_x_alt)

plot_consumption_age = plot(box = :on, size = [800, 500],
                xtickfont = font(14, "Computer Modern", :black),
                ytickfont = font(14, "Computer Modern", :black),
                legendfont = font(14, "Computer Modern", :black),
                guidefont = font(16, "Computer Modern", :black),
                titlefont = font(18, "Computer Modern", :black),
                margin = 4mm
                )

plot_consumption_age = plot!(plot_x,plot_consumption_age_S_female,lw = 3, lc= :blue, xlabel = "Age", legend=:topleft, label = "Singles, Female", xticks = [20,29,38,47,56,65])

plot_consumption_age = plot!(plot_x,plot_consumption_age_S_male, lw=3, lc= :green,label = "Singles, Male")

plot_consumption_age = plot!(plot_x,plot_consumption_age_C, lw=3, lc= :red,label = "Couples", linestyle=:dash)

# plot_consumption_age = plot!(plot_x_alt,plot_consumption_age_div, lw=3, lc= :green,label = "Divorced", linestyle=:dashdot)

savefig(plot_consumption_age, string(path,"Consumption.pdf"))

# Income lifecycle
x_vals = 20:0.1:65

x = [20,23,26,29,32,35,38,41,44,47,50,53,56,59,62,65]

plot_x = collect(x_vals)

# Singles
income_age_S = income_S_sim_ave_age

# spl = Spline1D(x,y,k=3,s=0.1)
spl_income_age_S = Spline1D(x,income_age_S,k=3)

plot_income_age_S = spl_income_age_S(plot_x)

# Couples
income_age_C = income_C_sim

# spl = Spline1D(x,y,k=3,s=0.1)
spl_income_age_C = Spline1D(x,income_age_C,k=3)

plot_income_age_C = spl_income_age_C(plot_x)

# Divorced
x_vals_alt = 23:0.1:65

x_alt = [23,26,29,32,35,38,41,44,47,50,53,56,59,62,65]

plot_x_alt = collect(x_vals_alt)

income_age_div = income_div_sim_ave_age[2:end]

# spl = Spline1D(x,y,k=3,s=0.1)
spl_income_age_div = Spline1D(x_alt,income_age_div,k=3)

plot_income_age_div = spl_income_age_div(plot_x_alt)

plot_income_age = plot(box = :on, size = [800, 500],
                xtickfont = font(14, "Computer Modern", :black),
                ytickfont = font(14, "Computer Modern", :black),
                legendfont = font(14, "Computer Modern", :black),
                guidefont = font(16, "Computer Modern", :black),
                titlefont = font(18, "Computer Modern", :black),
                margin = 4mm
                )

plot_income_age = plot!(plot_x,plot_income_age_S,lw = 3, lc= :blue, xlabel = "Age", legend=:bottomleft, label = "Singles", xticks = [20,29,38,47,56,65])

plot_income_age = plot!(plot_x,plot_income_age_C, lw=3, lc= :red,label = "Couples", linestyle=:dash)

plot_income_age = plot!(plot_x_alt,plot_income_age_div, lw=3, lc= :green,label = "Divorced", linestyle=:dashdot)

savefig(plot_income_age, string(path,"Income.pdf"))

# Labor lifecycle
x_vals = 20:0.1:65

x = [20,23,26,29,32,35,38,41,44,47,50,53,56,59,62,65]

plot_x = collect(x_vals)

# Singles
labor_age_S = [0.799970313, 0.983585938, 0.973457813, 0.96938125, 0.9778, 0.975253125, 0.965567188, 0.953373438, 0.939985938, 0.913165625, 0.84865, 0.8060375, 0.7504375, 0.690760937, 0.633907813, 0.5727]

# spl = Spline1D(x,y,k=3,s=0.1)
spl_labor_age_S = Spline1D(x,labor_age_S,k=1)

plot_labor_age_S = spl_labor_age_S(plot_x)

# Couples
labor_age_C = [0.739826042, 0.749039086, 0.767606599, 0.778651436, 0.786549204, 0.791165705, 0.796283847, 0.792411576, 0.793172005, 0.782192463, 0.769188246, 0.740342655, 0.69973784, 0.644757033, 0.577738147, 0.505682278]

# spl = Spline1D(x,y,k=3,s=0.1)
spl_labor_age_C = Spline1D(x,labor_age_C,k=1)

plot_labor_age_C = spl_labor_age_C(plot_x)

# Divorced
x_vals_alt = 23:0.1:65

x_alt = [23,26,29,32,35,38,41,44,47,50,53,56,59,62,65]

plot_x_alt = collect(x_vals_alt)

labor_age_div = [0.91025641, 0.923896326, 0.936555275, 0.954593226, 0.957445266, 0.954005282, 0.953470524, 0.949749304, 0.927681274, 0.884663818, 0.829611514, 0.759729166, 0.690966533, 0.628772626, 0.568521686]

# spl = Spline1D(x,y,k=3,s=0.1)
spl_labor_age_div = Spline1D(x_alt,labor_age_div,k=1)

plot_labor_age_div = spl_labor_age_div(plot_x_alt)

plot_labor_age = plot(box = :on, size = [800, 500],
                xtickfont = font(14, "Computer Modern", :black),
                ytickfont = font(14, "Computer Modern", :black),
                legendfont = font(14, "Computer Modern", :black),
                guidefont = font(16, "Computer Modern", :black),
                titlefont = font(18, "Computer Modern", :black),
                margin = 4mm
                )

plot_labor_age = plot!(plot_x,plot_labor_age_S,lw = 3, lc= :blue, xlabel = "Age", legend=:topright, label = "Singles", xticks = [20,29,38,47,56,65], ylabel="Fraction of full time")

plot_labor_age = plot!(plot_x,plot_labor_age_C, lw=3, lc= :red,label = "Couples", linestyle=:dash)

plot_labor_age = plot!(plot_x_alt,plot_labor_age_div, lw=3, lc= :green,label = "Divorced", linestyle=:dashdot)

savefig(plot_labor_age, string(path,"Labor.pdf"))

# Asset lifecycle
x_vals = 20:0.1:65

x = [20,23,26,29,32,35,38,41,44,47,50,53,56,59,62,65]

plot_x = collect(x_vals)

# Singles
asset_age_S = [0, 0.078533145, 0.232625075, 0.410608948, 0.604937203, 0.829835083, 1.062887646, 1.304746468, 1.532334284, 1.731157377, 1.896010305, 1.911235629, 1.820306052, 1.6015768, 1.228425538, 0.701792164]

# spl = Spline1D(x,y,k=3,s=0.1)
spl_asset_age_S = Spline1D(x,asset_age_S,k=1)

plot_asset_age_S = spl_asset_age_S(plot_x)

# Couples
asset_age_C = [0, 0.090208398, 0.169996165, 0.267736259, 0.395995148, 0.536659026, 0.697729665, 0.879546068, 1.062354448, 1.237613138, 1.389922435, 1.485865271, 1.494717913, 1.376149067, 1.100932058, 0.652873928]

# spl = Spline1D(x,y,k=3,s=0.1)
spl_asset_age_C = Spline1D(x,asset_age_C,k=1)

plot_asset_age_C = spl_asset_age_C(plot_x)

# Divorced
x_vals_alt = 23:0.1:65

x_alt = [23,26,29,32,35,38,41,44,47,50,53,56,59,62,65]

plot_x_alt = collect(x_vals_alt)

asset_age_div = [0.091496126, 0.150217472, 0.270936665, 0.425188461, 0.610870867, 0.808287384, 1.02421254, 1.24378816, 1.446809669, 1.617564128, 1.688527409, 1.647922175, 1.470273153, 1.137038658, 0.653062215]

# spl = Spline1D(x,y,k=3,s=0.1)
spl_asset_age_div = Spline1D(x_alt,asset_age_div,k=1)

plot_asset_age_div = spl_asset_age_div(plot_x_alt)

plot_asset_age = plot(box = :on, size = [800, 500],
                xtickfont = font(14, "Computer Modern", :black),
                ytickfont = font(14, "Computer Modern", :black),
                legendfont = font(14, "Computer Modern", :black),
                guidefont = font(16, "Computer Modern", :black),
                titlefont = font(18, "Computer Modern", :black),
                margin = 4mm
                )

plot_asset_age = plot!(plot_x,plot_asset_age_S,lw = 3, lc= :blue, xlabel = "Age", legend=:topleft, label = "Singles", xticks = [20,29,38,47,56,65])

plot_asset_age = plot!(plot_x,plot_asset_age_C, lw=3, lc= :red,label = "Couples", linestyle=:dash)

plot_asset_age = plot!(plot_x_alt,plot_asset_age_div, lw=3, lc= :green,label = "Divorced", linestyle=:dashdot)

savefig(plot_asset_age, string(path,"Asset.pdf"))

# Interest rates lifecycle
# Singles
x_vals = 20:0.1:62

x = [20,23,26,29,32,35,38,41,44,47,50,53,56,59,62]

plot_x = collect(x_vals)

# Data
interest_data = [16.742057, 15.801763, 18.068077, 17.065883, 17.638265, 13.329731, 14.451917, 13.19855, 16.350672, 15.957453, 15.769524, 16.920961, 16.069703, 15.847124, 17.837056]

# spl = Spline1D(x,y,k=3,s=0.1)
spl_interest_data = Spline1D(x,interest_data,k=1)

plot_interest_data = spl_interest_data(plot_x)

# Model
interest_model = [16.64369971, 17.59341805, 17.01114221, 16.76011301, 16.48805298, 15.85065642, 15.7056943, 16.10061226, 15.79217573, 16.11795615, 16.24700721, 16.88135607, 16.96554129, 16.42386957, 16.45985565]

# spl = Spline1D(x,y,k=3,s=0.1)
spl_interest_model = Spline1D(x,interest_model,k=1)

plot_interest_model = spl_interest_model(plot_x)

plot_interest_single = plot(box = :on, size = [800, 500],
                xtickfont = font(14, "Computer Modern", :black),
                ytickfont = font(14, "Computer Modern", :black),
                legendfont = font(14, "Computer Modern", :black),
                guidefont = font(16, "Computer Modern", :black),
                titlefont = font(18, "Computer Modern", :black),
                margin = 4mm
                )

plot_interest_single = plot!(plot_x,plot_interest_data,lw = 3, lc= :blue, xlabel = "Age", legend=:bottomleft, label = "Data", xticks = [20,29,38,47,56], ylim=[0,20])

plot_interest_single = plot!(plot_x,plot_interest_model, lw=3, lc= :red,label = "Model", linestyle=:dash)

savefig(plot_interest_single, string(path,"Interest_single.pdf"))

# Couples
x_vals = 20:0.1:62

x = [20,23,26,29,32,35,38,41,44,47,50,53,56,59,62]

plot_x = collect(x_vals)

# Data
interest_data = [17.35866, 18.947325, 16.31132, 15.904852, 17.056801, 16.142919, 15.640548, 16.254436, 17.66715, 15.578185, 14.268125, 15.725748, 17.318661, 16.62182, 14.817729]

# spl = Spline1D(x,y,k=3,s=0.1)
spl_interest_data = Spline1D(x,interest_data,k=1)

plot_interest_data = spl_interest_data(plot_x)

# Model
interest_model = [16.54999645, 14.89341695, 15.1381264, 15.60559849, 15.49499036, 14.6858975, 14.25162102, 14.58627374, 14.47378519, 15.19480036, 15.62218223, 16.14421424, 15.65449474, 14.49337031, 14.29029953]

# spl = Spline1D(x,y,k=3,s=0.1)
spl_interest_model = Spline1D(x,interest_model,k=1)

plot_interest_model = spl_interest_model(plot_x)

plot_interest_couples = plot(box = :on, size = [800, 500],
                xtickfont = font(14, "Computer Modern", :black),
                ytickfont = font(14, "Computer Modern", :black),
                legendfont = font(14, "Computer Modern", :black),
                guidefont = font(16, "Computer Modern", :black),
                titlefont = font(18, "Computer Modern", :black),
                margin = 4mm
                )

plot_interest_couples = plot!(plot_x,plot_interest_data,lw = 3, lc= :blue, xlabel = "Age", legend=:bottomleft, label = "Data", xticks = [20,29,38,47,56], ylim=[0,20])

plot_interest_couples = plot!(plot_x,plot_interest_model, lw=3, lc= :red,label = "Model", linestyle=:dash)

savefig(plot_interest_couples, string(path,"Interest_couples.pdf"))

# Divorced
x_vals = 26:0.1:62

x = [26,29,32,35,38,41,44,47,50,53,56,59,62]

plot_x = collect(x_vals)

# Data
interest_data = [14.28978, 19.262968, 19.938347, 15.642569, 13.994858, 15.404031, 15.278429, 15.535402, 16.693241, 16.458317, 18.630753, 17.178202, 17.721712]

# spl = Spline1D(x,y,k=3,s=0.1)
spl_interest_data = Spline1D(x,interest_data,k=1)

plot_interest_data = spl_interest_data(plot_x)

# Model
x_vals_model = 23:0.1:62

x_model = [23,26,29,32,35,38,41,44,47,50,53,56,59,62]

plot_x_model = collect(x_vals_model)

interest_model = [16.07546265, 16.24726666, 16.44311255, 16.21731109, 15.61170349, 15.49204763, 15.90743088, 15.62520793, 16.11114702, 16.19441963, 16.79644268, 16.88880771, 16.35724242, 16.33242306]

# spl = Spline1D(x,y,k=3,s=0.1)
spl_interest_model = Spline1D(x_model,interest_model,k=1)

plot_interest_model = spl_interest_model(plot_x_model)

plot_interest_divorced = plot(box = :on, size = [800, 500],
                xtickfont = font(14, "Computer Modern", :black),
                ytickfont = font(14, "Computer Modern", :black),
                legendfont = font(14, "Computer Modern", :black),
                guidefont = font(16, "Computer Modern", :black),
                titlefont = font(18, "Computer Modern", :black),
                margin = 4mm
                )

plot_interest_divorced = plot!(plot_x,plot_interest_data,lw = 3, lc= :blue, xlabel = "Age", legend=:bottomleft, label = "Data", xticks = [20,29,38,47,56], ylim=[0,25])

plot_interest_divorced = plot!(plot_x_model,plot_interest_model, lw=3, lc= :red,label = "Model", linestyle=:dash)

savefig(plot_interest_divorced, string(path,"Interest_divorced.pdf"))

# Medical Expenses
# Married vs Single
x = [20, 26, 32, 38, 44, 50, 56, 62]

first_ser = [2086.04, 3461.91, 3650.80, 3154.31, 2643.09, 4518.65, 5442.91, 5501.35]
first_err = [609.39, 885.48, 722.81, 507.58, 400.08, 713.05, 814.55, 653.70]

second_ser = [1992.12, 2272.84, 2454.87, 2565.60, 3469.36, 3126.83, 5396.73, 4834.71]
second_err = [501.30, 468.29, 585.42, 629.76, 968.69, 557.12, 1518.32, 1016.17]

plot_exp_married_single = plot(box = :on, size = [800, 500],
                xtickfont = font(14, "Computer Modern", :black),
                ytickfont = font(14, "Computer Modern", :black),
                legendfont = font(14, "Computer Modern", :black),
                guidefont = font(16, "Computer Modern", :black),
                titlefont = font(18, "Computer Modern", :black),
                margin = 4mm
                )

plot_exp_married_single = plot!(x,first_ser,yerror=first_err,lw = 3, seriescolor= :blue, markerstrokecolor = :blue, xlabel = "Age (6-year bins)", legend=:topleft, label = "Married", xticks = [20,32,44,56], ylabel="2018 US-Dollar", ylim = [0,8000])

plot_exp_married_single = plot!(x,first_ser,lw = 0, seriescolor= :blue, label = "", markershape=:circle, markercolor = :match, markersize =5, markerstrokecolor=:blue)

plot_exp_married_single = plot!(x,second_ser,yerror=second_err,lw = 3, seriescolor= :red, markerstrokecolor = :red, label = "Single",linestyle=:dash)

plot_exp_married_single = plot!(x,second_ser,lw = 0, seriescolor= :red, label = "",linestyle=:dash, markershape=:circle, markercolor = :red, markersize =5, markerstrokecolor=:red, linealpha=0)

savefig(plot_exp_married_single, string(path,"expenses_married_single.png"))

# Married female vs Single female
x = [20, 26, 32, 38, 44, 50, 56, 62]

first_ser = [2303.01, 4057.91, 4818.67, 3859.87, 2875.25, 4423.96, 6178.05, 5384.42]
first_err = [842.31, 1240.00, 1253.87, 817.73, 569.00, 697.08, 1276.14, 810.45]

second_ser = [1927.50, 2539.61, 2817.86, 2478.80, 3292.56, 3178.23, 6255.31, 5364.26]
second_err = [345.60, 787.03, 907.63, 791.12, 949.39, 666.01, 2543.88, 1362.21]

plot_exp_married_single_female = plot(box = :on, size = [800, 500],
                xtickfont = font(14, "Computer Modern", :black),
                ytickfont = font(14, "Computer Modern", :black),
                legendfont = font(14, "Computer Modern", :black),
                guidefont = font(16, "Computer Modern", :black),
                titlefont = font(18, "Computer Modern", :black),
                margin = 4mm
                )

plot_exp_married_single_female = plot!(x,first_ser,yerror=first_err,lw = 3, seriescolor= :blue, markerstrokecolor = :blue, xlabel = "Age (6-year bins)", legend=:topleft, label = "Married female", xticks = [20,32,44,56], ylabel="2018 US-Dollar", ylim = [0,10000])

plot_exp_married_single_female = plot!(x,first_ser,lw = 0, seriescolor= :blue, label = "", markershape=:circle, markercolor = :match, markersize =5, markerstrokecolor=:blue)

plot_exp_married_single_female = plot!(x,second_ser,yerror=second_err,lw = 3, seriescolor= :red, markerstrokecolor = :red, label = "Single female",linestyle=:dash)

plot_exp_married_single_female = plot!(x,second_ser,lw = 0, seriescolor= :red, label = "",linestyle=:dash, markershape=:circle, markercolor = :red, markersize =5, markerstrokecolor=:red, linealpha=0)

savefig(plot_exp_married_single_female, string(path,"expenses_married_female_single_female.png"))

# Married male vs Married female
x = [20, 26, 32, 38, 44, 50, 56, 62]

first_ser = [1676.15, 2684.04, 2282.06, 2490.12, 2394.33, 4612.79, 4647.82, 5601.22]
first_err = [749.78, 1119.58, 616.72, 531.75, 525.51, 1238.06, 848.32, 955.60]

second_ser = [2303.01, 4057.91, 4818.67, 3859.87, 2875.25, 4423.96, 6178.05, 5384.42]
second_err = [842.31, 1240.00, 1253.87, 817.73, 569.00, 697.08, 1276.14, 810.45]

plot_exp_married_male_female = plot(box = :on, size = [800, 500],
                xtickfont = font(14, "Computer Modern", :black),
                ytickfont = font(14, "Computer Modern", :black),
                legendfont = font(14, "Computer Modern", :black),
                guidefont = font(16, "Computer Modern", :black),
                titlefont = font(18, "Computer Modern", :black),
                margin = 4mm
                )

plot_exp_married_male_female = plot!(x,first_ser,yerror=first_err,lw = 3, seriescolor= :blue, markerstrokecolor = :blue, xlabel = "Age (6-year bins)", legend=:topleft, label = "Married male", xticks = [20,32,44,56], ylabel="2018 US-Dollar", ylim = [0,10000])

plot_exp_married_male_female = plot!(x,first_ser,lw = 0, seriescolor= :blue, label = "", markershape=:circle, markercolor = :match, markersize =5, markerstrokecolor=:blue)

plot_exp_married_male_female = plot!(x,second_ser,yerror=second_err,lw = 3, seriescolor= :red, markerstrokecolor = :red, label = "Married female",linestyle=:dash)

plot_exp_married_male_female = plot!(x,second_ser,lw = 0, seriescolor= :red, label = "",linestyle=:dash, markershape=:circle, markercolor = :red, markersize =5, markerstrokecolor=:red, linealpha=0)

savefig(plot_exp_married_male_female, string(path,"expenses_married_male_married_female.png"))

# Married male vs Single male
x = [20, 26, 32, 38, 44, 50, 56, 62]

first_ser = [1676.15, 2684.04, 2282.06, 2490.12, 2394.33, 4612.79, 4647.82, 5601.22]
first_err = [749.78, 1119.58, 616.72, 531.75, 525.51, 1238.06, 848.32, 955.60]

second_ser = [2048.25, 2056.33, 2080.99, 2657.39, 3660.05, 3076.57, 4463.71, 4219.71]
second_err = [876.88, 540.70, 632.17, 976.06, 1788.32, 807.83, 1380.90, 1148.59]

plot_exp_married_single_male = plot(box = :on, size = [800, 500],
                xtickfont = font(14, "Computer Modern", :black),
                ytickfont = font(14, "Computer Modern", :black),
                legendfont = font(14, "Computer Modern", :black),
                guidefont = font(16, "Computer Modern", :black),
                titlefont = font(18, "Computer Modern", :black),
                margin = 4mm
                )

plot_exp_married_single_male = plot!(x,first_ser,yerror=first_err,lw = 3, seriescolor= :blue, markerstrokecolor = :blue, xlabel = "Age (6-year bins)", legend=:topleft, label = "Married male", xticks = [20,32,44,56], ylabel="2018 US-Dollar", ylim = [0,10000])

plot_exp_married_single_male = plot!(x,first_ser,lw = 0, seriescolor= :blue, label = "", markershape=:circle, markercolor = :match, markersize =5, markerstrokecolor=:blue)

plot_exp_married_single_male = plot!(x,second_ser,yerror=second_err,lw = 3, seriescolor= :red, markerstrokecolor = :red, label = "Single male",linestyle=:dash)

plot_exp_married_single_male = plot!(x,second_ser,lw = 0, seriescolor= :red, label = "",linestyle=:dash, markershape=:circle, markercolor = :red, markersize =5, markerstrokecolor=:red, linealpha=0)

savefig(plot_exp_married_single_male, string(path,"expenses_married_male_single_male.png"))

# Single male vs Single female
x = [20, 26, 32, 38, 44, 50, 56, 62]

first_ser = [2048.25, 2056.33, 2080.99, 2657.39, 3660.05, 3076.57, 4463.71, 4219.71]
first_err = [876.88, 540.70, 632.17, 976.06, 1788.32, 807.83, 1380.90, 1148.59]

second_ser = [1927.50, 2539.61, 2817.86, 2478.80, 3292.56, 3178.23, 6255.31, 5364.26]
second_err = [345.60, 787.03, 907.63, 791.12, 949.39, 666.01, 2543.88, 1362.21]

plot_exp_single_male_female = plot(box = :on, size = [800, 500],
                xtickfont = font(14, "Computer Modern", :black),
                ytickfont = font(14, "Computer Modern", :black),
                legendfont = font(14, "Computer Modern", :black),
                guidefont = font(16, "Computer Modern", :black),
                titlefont = font(18, "Computer Modern", :black),
                margin = 4mm
                )

plot_exp_single_male_female = plot!(x,first_ser,yerror=first_err,lw = 3, seriescolor= :blue, markerstrokecolor = :blue, xlabel = "Age (6-year bins)", legend=:topleft, label = "Single male", xticks = [20,32,44,56], ylabel="2018 US-Dollar", ylim = [0,10000])

plot_exp_single_male_female = plot!(x,first_ser,lw = 0, seriescolor= :blue, label = "", markershape=:circle, markercolor = :match, markersize =5, markerstrokecolor=:blue)

plot_exp_single_male_female = plot!(x,second_ser,yerror=second_err,lw = 3, seriescolor= :red, markerstrokecolor = :red, label = "Single female",linestyle=:dash)

plot_exp_single_male_female = plot!(x,second_ser,lw = 0, seriescolor= :red, label = "",linestyle=:dash, markershape=:circle, markercolor = :red, markersize =5, markerstrokecolor=:red, linealpha=0)

savefig(plot_exp_single_male_female, string(path,"expenses_single_male_single_female.png"))

# Calibration
# Expense shocks
x = [20, 26, 32, 38, 44, 50, 56, 62]

perc_95 = [37867.73 20472.00;
            25860.11 22615.92;
            24499.11 25094.35;
            20832.17 18999.93;
            14738.40 31552.32;
            26437.99 25790.97;
            27973.34 52265.67;
            25360.92 26763.57]

perc_98 = [142708.72  62241.04;
            47873.49  79580.04;
            76452.26 105396.74;
            82348.90  48495.42;
            56478.85 104122.64;
            69384.15 153571.01;
            66666.57 210308.47;
            49228.16 104261.12]

plot_exp_calibration = plot(box = :on, size = [800, 500],
                xtickfont = font(14, "Computer Modern", :black),
                ytickfont = font(14, "Computer Modern", :black),
                legendfont = font(14, "Computer Modern", :black),
                guidefont = font(16, "Computer Modern", :black),
                titlefont = font(18, "Computer Modern", :black),
                margin = 4mm
                )

plot_exp_calibration = plot!(x,perc_95[:,1]/1000,lw = 3, seriescolor= :blue, xlabel = "Age (6-year bins)", legend=:topleft, label = L"\kappa_{1,\textrm{Married}}", xticks = [20,32,44,56], ylabel="2018 US-Dollar (in thousands)", ylim=[-100,350], yticks=[0,100,200,300])

plot_exp_calibration = plot!(x,perc_95[:,1]/1000,lw = 0, seriescolor= :blue, label = "", markershape=:circle, markercolor = :match, markersize =6, markerstrokecolor=:blue)

plot_exp_calibration = plot!(x,perc_95[:,2]/1000,lw = 3, seriescolor= :blue, label = L"\kappa_{1,\textrm{Single/Divorced}}", linestyle=:dash)

plot_exp_calibration = plot!(x,perc_95[:,2]/1000,lw = 0, seriescolor= :blue, label = "", markershape=:circle, markercolor = :match, markersize =6, markerstrokecolor=:blue)

plot_exp_calibration = plot!(x,perc_98[:,1]/1000,lw = 3, seriescolor= :red, label = L"\kappa_{2,\textrm{Married}}")

plot_exp_calibration = plot!(x,perc_98[:,1]/1000,lw = 0, seriescolor= :red, label = "", markershape=:circle, markercolor = :match, markersize =6, markerstrokecolor=:red)

plot_exp_calibration = plot!(x,perc_98[:,2]/1000,lw = 3, seriescolor= :red, label = L"\kappa_{2,\textrm{Single/Divorced}}", linestyle=:dash)

plot_exp_calibration = plot!(x,perc_98[:,2]/1000,lw = 0, seriescolor= :red, label = "", markershape=:circle, markercolor = :match, markersize =6, markerstrokecolor=:red)

savefig(plot_exp_calibration, string(path,"expenses_calibration.png"))
