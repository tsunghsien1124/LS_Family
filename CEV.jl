W_benchmark = similar(W_S)
copy!(W_benchmark, W_S)
μ_benchmark = similar(μ_S)
copy!(μ_benchmark, μ_S)

W_policy = similar(W_S)
copy!(W_policy, W_S)
μ_policy = similar(μ_S)
copy!(μ_policy, μ_S)

CEV = ((W_policy ./ W_benchmark).^(1.0/(1.0-parameters.γ_c)) .- 1.0)
CEV_weighted = sum(μ_policy.*CEV)*100

CEV_weighted_alt = sum(μ_benchmark.*CEV)*100
