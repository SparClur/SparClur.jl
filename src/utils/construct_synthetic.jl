using Random

function construct_synthetic(
    num_features::Int,
    cluster_sizes::Vector{Int},
    num_relevant::Int;
    snr::Float64 = -1.0,
    )
    # randomly selected relevant features
    supp = sort(sample(1:num_features, num_relevant, replace = false))
    weights = [zeros(num_relevant) for _ in eachindex(cluster_sizes)]
    Xs = [zeros(cs, num_features) for cs in cluster_sizes]
    Ys = [zeros(cs) for cs in cluster_sizes]

    # get X~norm(0,1) and Y for each cluster
    for (i, cs) in enumerate(cluster_sizes)
        Xs[i] = randn(cs, num_features)
        # generate random weights and cache them
        weights[i] = rand([-1.0, 1.0], num_relevant)
        Ys[i] = Xs[i][:, supp] * weights[i]
        # add noise
        if snr > 0
            E = randn(cs)
            E .*= norm(Ys[i]) / (sqrt(snr) * norm(E))
            Y .+= E
        end
    end
    return (Xs, Ys, supp, weights)
end
