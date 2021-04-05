import SparClur2
import CSV
import CPLEX
import MLDataUtils: kfolds
import StatsBase: mean
using DataFrames
using GLMNet
using Random

include(joinpath(@__DIR__(), "real.jl"))

seeds = 1:5
depth = 5

method = "sparse"
# method = "sparclur"
if method == "sparse"
    features_range = 1:5
    ignore_coordination = true
else
    features_range = 7:17
    ignore_coordination = false
end

for seed in seeds
    io = open(joinpath("output", method * "_s$(seed).csv"), "w")
    sparse_io = open("output/sparse_s$(seed).csv", "w")
    println(sparse_io, "totalsupp,r2")
    for num_features in features_range
        res = zeros(length(depths))
        (clusters, best_gamma, (supp, weights)) = train_sparclur(depth, seed, relaxation = false, ignore_coordination = ignore_coordination, q_range = [num_features])
        (X_list, Y_list, memberships_list) = read_data(depth, seed, false)
        (Xs_test, Ys_test) = make_clusters(X_list, Y_list, memberships_list, clusters)
        Ys_pred = [Xs_test[c][:, supp[c]] * weights[c] for c in eachindex(clusters)]
        mse = sum(sum(abs2, Ys_pred[c] - Ys_test[c]) for c in eachindex(clusters))
        mean_all = mean(Y_list)
        baseline_mse = sum(sum(abs2, mean_all .- Ys_test[c]) for c in eachindex(clusters))
        res = 1 - mse / baseline_mse
        total_supp = length(unique(vcat(supp...)))
        @show res, num_features, total_supp
        println(sparse_io, "$(total_supp),$(res)")
    end
    close(sparse_io)
end
