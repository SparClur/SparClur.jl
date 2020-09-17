# https://github.com/lkapelevich/ClusterRegression.jl/blob/master/datasets/synthetic/experiments/support_recovery.jl
using SparClur2
using CPLEX
using Random
using DataFrames
import StatsBase: mean, std
import CSV

signal_ratio = 400.0
num_features = 2000
cluster_range = [1, 2, 5, 10, 20]
num_relevant = 10
optimizer = CPLEX.Optimizer
# coordinated_gamma_factors = [0.001, 0.01, 0.02, 0.04, 0.08, 0.1, 1.0] # 0
coordinated_gamma_factors = [1e-4, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10_000.0, 100_000.0] # 2
num_obs_range = 60:20:600
seeds = 1:5
optimizer_params = ("CPX_PARAM_TILIM" => 30, "CPXPARAM_MIP_Tolerances_MIPGap" => 1e-2)

# Since we don't do an out-of-sample test with synthetic data, we shouldn't
# cross validate on the data we use in training. Instead, we use a held out
# dataset.
function get_gamma(Xs, Ys, true_supp, true_weights, num_relevant, num_obs)
    best_gamma = 0.0
    best_acc = -Inf # TODO mse instead
    best_mse = Inf
    best_time = Inf
    for gamma in coordinated_gamma_factors * num_relevant / num_obs
        gamma_time = @elapsed (supp, weights) = SparClur2.solve_MIOP(Xs, Ys, num_relevant, gamma, optimizer, optimizer_params = optimizer_params)
        acc = SparClur2.accuracy(supp, true_supp)
        Y_preds = [Xs[k][:, supp] * weights[k] for k in eachindex(Ys)]
        mse = sum(sum(abs2, Y_preds[k] - Ys[k]) for k in eachindex(Ys))
        if mse < best_mse - 1e-3
        # if (acc > best_acc + 1e-3) # || (acc ≈ 1 && gamma_time < best_time)
            # best_acc = acc
            best_mse = mse
            best_gamma = gamma
            best_time = gamma_time
        end
        num_features = size(Xs[1], 2)
        open("output/validexp1_num_clusters_$(length(Xs))_num_features_$(num_features)_num_obs_$(num_obs).txt", "a") do io
            println(io, "gamma = ", gamma, " acc = ", acc, " mse = ", mse, " time = ", gamma_time)
            flush(io)
        end
    end
    return best_gamma
end

# function runall()
    num_rows = length(num_obs_range) * length(seeds) * length(cluster_range)
    results = zeros(num_rows, 7)
    global idx = 0
    # Loop over number of observations
    for num_obs in num_obs_range, num_clusters in cluster_range
        cluster_sizes = fill(div(num_obs, num_clusters), num_clusters)
        @show cluster_sizes
        # do a dry run to find a sensible gamma, seed won't be used in future
        Random.seed!(seeds[end] + 1)
        # data, truth = construct_synthetic(nfeatures, cluster_sizes, nrelevant, snr=SNR, same_weights=false, zero_one=true)
        (Xs, Ys, true_supp, true_weights) = SparClur2.construct_synthetic(num_features, cluster_sizes, num_relevant, snr = signal_ratio)
        gamma = get_gamma(Xs, Ys, true_supp, true_weights, num_relevant, num_obs)

        # TODO delete
        Random.seed!(seeds[end] + 2)
        # data, truth = construct_synthetic(nfeatures, cluster_sizes, nrelevant, snr=SNR, same_weights=false, zero_one=true)
        (Xs, Ys, true_supp, true_weights) = SparClur2.construct_synthetic(num_features, cluster_sizes, num_relevant, snr = signal_ratio)
        gamma = get_gamma(Xs, Ys, true_supp, true_weights, num_relevant, num_obs)

        @show gamma
        # Now test model on five different sets of data
        for seed in seeds
            @show seed
            Random.seed!(seed)
            (Xs, Ys, true_supp, true_weights) = SparClur2.construct_synthetic(num_features, cluster_sizes, num_relevant, snr = signal_ratio)
            tm = @elapsed (supp, weights) = SparClur2.solve_MIOP(Xs, Ys, num_relevant, gamma, optimizer, optimizer_params = optimizer_params)
            acc = SparClur2.accuracy(supp, true_supp)
            fp = SparClur2.falsepositive(supp, true_supp)
            global idx += 1
            results[idx, :] = [seed, num_obs, num_clusters, acc, fp, tm, gamma * num_obs / num_relevant]
        end # seeds
    end
    return results
# end
# results = runall()
CSV.write("output/exp1_2.csv", DataFrame(results))

function analyze(results)
    df = DataFrame(results[:, 2:(end - 1)], [:observations, :nclusters, :accuracy, :false_detection, :time])

    se(x) = std(x, corrected = false)
    df_agg = aggregate(df, [:observations, :nclusters], [mean, se])


    for nclusters in cluster_range
        subdf = filter(row -> row[:nclusters] == nclusters, df_agg)
        # delete!(subdf, :nclusters)
        isdir("output") || mkdir("output")
        CSV.write("output/nclusters_$(nclusters)_cplex.csv", subdf)
    end
    return
end
# analyze(results)
