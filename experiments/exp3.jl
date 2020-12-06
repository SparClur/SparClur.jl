# https://github.com/lkapelevich/ClusterRegression.jl/blob/master/datasets/synthetic/experiments/uncoordinated.jl
using SparClur2
using CPLEX
using GLPK
using Random
using DataFrames
import StatsBase: mean, std
import CSV

relevant_per_cluster = 10
num_common_range = [2, 5]
model_q_range = [2, 5, 8, 10, 12, 15, 18, 20]
gamma_factor = 1.0
num_features = 2000
num_obs = 1000
seeds = 1:5
num_clusters = 2
signal_ratio = 400.0
n_rows = length(num_common_range) * length(seeds) * length(model_q_range)
optimizer = CPLEX.Optimizer
optimizer_params = ("CPX_PARAM_TILIM" => 30, "CPXPARAM_MIP_Tolerances_MIPGap" => 1e-2)

function acc_limit(num_common::Int, relevant_per_cluster::Int, num_clusters::Int, model_q::Int)
    support_superset = num_common + num_clusters * (relevant_per_cluster - num_common)
    return min(support_superset, model_q) / support_superset
end

results = zeros(n_rows, 9)
global idx = 0
cluster_sizes = fill(div(num_obs, num_clusters), num_clusters)
for (i, incommon) in enumerate(num_common_range), seed in seeds
    Random.seed!(seed)
    (big_Xs, big_Ys, true_supp, true_weights, true_common_inds) = SparClur2.construct_synthetic_nocoord(
        num_features, cluster_sizes .* 2, relevant_per_cluster, incommon, snr = signal_ratio)
    (Xs, test_Xs) = ([big_Xs[k][1:cluster_sizes[k], :] for k in 1:num_clusters], [big_Xs[k][(cluster_sizes[k] + 1):end, :] for k in 1:num_clusters])
    (Ys, test_Ys) = ([big_Ys[k][1:cluster_sizes[k]] for k in 1:num_clusters], [big_Ys[k][(cluster_sizes[k] + 1):end] for k in 1:num_clusters])
    all_inds = unique(vcat(true_supp...))
    @show all_inds
    gamma = gamma_factor / num_obs
    for (k, model_q) in enumerate(model_q_range)
        tm = @elapsed (supp, weights) = SparClur2.solve_MIOP(Xs, Ys, model_q, gamma, optimizer, optimizer_params = optimizer_params, silent = true, regularize_weights = false)
        acc = SparClur2.accuracy(supp, all_inds)
        acc_common = SparClur2.accuracy(supp, true_common_inds)
        fp = SparClur2.falsepositive(supp, all_inds)
        pred_Ys = [test_Xs[k][:, supp] * weights[k] for k in 1:num_clusters]
        test_err = sum(sum(abs2, test_Ys[k] - pred_Ys[k]) for k in 1:num_clusters)
        base_err = sum(sum(abs2, test_Ys[k] .- mean(test_Ys[k])) for k in 1:num_clusters)
        @show test_err, base_err
        rsqr = 1 - test_err / base_err
        global idx += 1
        acc_lim = acc_limit(incommon, relevant_per_cluster, num_clusters, model_q)
        results[idx, :] = [seed, incommon, model_q, acc, acc_common, fp, tm, acc_lim, rsqr]
    end # k
end

CSV.write("output/exp3.csv", DataFrame(results))

function analyze(results)
    df = DataFrame(results[:, 2:end], [:shared, :k, :accuracy, :a_common, :false_detection, :time, :a_limit, :r2])
    se(x) = std(x, corrected = false)
    # df_agg = aggregate(df, [:shared, :k], [mean, se]) # TODO deprecation
    cols = [:shared, :k]
    df_agg = combine(groupby(df, cols), [names(df, Not(cols)) .=> f for f in [mean, se]]...)

    for incommon in num_common_range
        subdf = filter(row -> row[:shared] == incommon, df_agg)
        isdir("output") || mkdir("output")
        CSV.write("output/unco_$(incommon)_cplexr3.csv", subdf)
    end
    return
end
analyze(results)
