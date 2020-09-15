import StatsBase: sample

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
            Ys[i] .+= E
        end
    end
    return (Xs, Ys, supp, weights)
end

function construct_synthetic_nocoord(
    num_features::Int,
    cluster_sizes::Vector{Int},
    relevant_per_cluster::Int,
    incommon::Int;
    snr::Float64 = -1.0,
    )
    num_clusters = length(cluster_sizes)
    Xs = [zeros(cs, num_features) for cs in cluster_sizes]
    Ys = [zeros(cs) for cs in cluster_sizes]
    weights = [Float64[] for _ in 1:num_clusters]
    support = [Int[] for _ in 1:num_clusters]
    not_in_common = relevant_per_cluster - incommon

    inds_table = zeros(Int, num_clusters, relevant_per_cluster)
    all_inds = sample(1:num_features, incommon + num_clusters * not_in_common, replace = false)
    common_inds = all_inds[1:incommon]
    idx = incommon + 1
    for (k, num_obs) in enumerate(cluster_sizes)
        inds = idx:(idx + not_in_common - 1)
        idx += not_in_common
        support[k] = vcat(common_inds, all_inds[inds])

        Xs[k] = randn(num_obs, num_features)
        weights[k] = rand([-1.0, 1.0], relevant_per_cluster)
        Ys[k] = Xs[k][:, support[k]] * weights[k]
        if snr > 0
            E = randn(num_obs)
            E .*= norm(Ys[k]) / (sqrt(snr) * norm(E))
            Ys[k] .+= E
        end
    end

    return (Xs, Ys, support, weights, common_inds)
end
