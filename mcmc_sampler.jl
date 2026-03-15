# dande | kagome-sampler
# adaptive mcmc for bayesian inference. 

using Random, LinearAlgebra, Statistics

function target_log_pdf(x::Vector{Float64})
    # dummy multimodal target distribution (replace with actual log-likelihood)
    return log(exp(-0.5 * norm(x .- 2.0)^2) + exp(-0.5 * norm(x .+ 2.0)^2))
end

function adaptive_mcmc(iters::Int, dim::Int)
    samples = zeros(Float64, dim, iters)
    current_x = randn(dim)
    cov_mat = Matrix{Float64}(I, dim, dim)
    accepts = 0

    for i in 2:iters
        # adapt covariance every 100 steps
        if i > 100 && i % 100 == 0
            cov_mat = cov(samples[:, 1:i-1]') .+ 1e-6 * I
        end
        
        proposal = current_x .+ cholesky(cov_mat).L * randn(dim)
        
        log_alpha = target_log_pdf(proposal) - target_log_pdf(current_x)
        if log(rand()) < log_alpha
            current_x = proposal
            accepts += 1
        end
        samples[:, i] = current_x
    end
    println("acceptance rate: ", round(accepts/iters, digits=2))
    return samples
end

@time chain = adaptive_mcmc(100_000, 5)

