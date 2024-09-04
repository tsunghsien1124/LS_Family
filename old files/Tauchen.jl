import QuantEcon: tauchen, rouwenhorst

N = 5

# Persistent process for women
# ρ = 0.886
# σ_sq = 0.05128

ρ = 0.963
σ_sq = 0.014
# Double the standard deviation
# σ_sq = σ_sq*2
# Triple the standard deviation
# σ_sq = σ_sq*3
# Quintuple the standard deviation
# σ_sq = σ_sq*5
# Six times the standard deviation
# σ_sq = σ_sq*6
# Seven times the standard deviation
# σ_sq = σ_sq*7
# Ten times the standard deviation
# σ_sq = σ_sq*10
# Thirteen times the standard deviation
σ_sq = σ_sq*13
# Fifteen times the standard deviation
# σ_sq = σ_sq*15

# ρ = 0.99
# σ_sq = 0.007

ρ_3 = ρ^3
σ_sq_3 = (1+ρ^2+ρ^4)*σ_sq
σ_3 = sqrt(σ_sq_3)

mc = tauchen(N,ρ_3,σ_3)
mc = rouwenhorst(N,ρ_3,σ_3)

mc.p
collect(mc.state_values)

# Persistent process for men
# ρ = 0.9999
# σ_sq = 0.0032

ρ = 0.973
σ_sq = 0.016
# Double the standard deviation
# σ_sq = σ_sq*2
# Triple the standard deviation
# σ_sq = σ_sq*3
# Quintuple the standard deviation
# σ_sq = σ_sq*5
# Six times the standard deviation
# σ_sq = σ_sq*6
# Seven times the standard deviation
# σ_sq = σ_sq*7
# Ten times the standard deviation
# σ_sq = σ_sq*10
# Thirteen times the standard deviation
σ_sq = σ_sq*13
# Fifteen times the standard deviation
# σ_sq = σ_sq*15

ρ_3 = ρ^3
σ_sq_3 = (1+ρ^2+ρ^4)*σ_sq
σ_3 = sqrt(σ_sq_3)

mc = tauchen(N,ρ_3,σ_3)
mc = rouwenhorst(N,ρ_3,σ_3)

mc.p
collect(mc.state_values)

# Transitory for women

# Transitory for men






using Distributions
using QuadGK

function adda_cooper(N::Integer, ρ::Real, σ::Real; μ::Real = 0.0)
    """
    Approximation of an autoregression process with a Markov chain proposed by Adda and Cooper (2003)
    """

    σ_ϵ = σ / sqrt(1.0 - ρ^2.0)
    ϵ = σ_ϵ .* quantile.(Normal(), [i/N for i = 0:N]) .+ μ
    z = zeros(N)
    for i = 1:N
        if i != (N+1)/2
            z[i] = N * σ_ϵ * (pdf(Normal(), (ϵ[i]-μ)/σ_ϵ) - pdf(Normal(), (ϵ[i+1]-μ)/σ_ϵ)) + μ
        end
    end
    Π = zeros(N,N)
    if ρ == 0.0
        Π .= 1.0/N
    else
        for i = 1:N, j = 1:N
            f(u) = exp(-(u-μ)^2.0/(2.0*σ_ϵ^2.0)) * (cdf(Normal(), (ϵ[j+1]-μ*(1.0-ρ)-ρ*u)/σ) - cdf(Normal(), (ϵ[j]-μ*(1.0-ρ)-ρ*u)/σ))
            integral, err = quadgk(u -> f(u), ϵ[i], ϵ[i+1])
            Π[i,j] = (N/sqrt(2.0*π*σ_ϵ^2.0)) * integral
        end
    end
    return z, Π
end



a = 0.0009
b = 0.0233
# b = 0.0236
c = 1.0297
# c = 1.0708

x_plot = collect(range(25,65,step=1))
x = collect(range(0,40,step=1))

y = (exp.(a.*(x.^2).-b.*x.-c))./(1 .+ exp.(a.*(x.^2).-b.*x.-c))

using Plots

plot(x_plot,y)
