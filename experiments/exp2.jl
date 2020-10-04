# ref https://github.com/lkapelevich/ClusterRegression.jl/blob/master/datasets/synthetic/experiments/big_synthetic_2.jl
using SparClur2
using CPLEX
using Random
using DataFrames
import CSV

signal_ratio = 400.0
num_clusters = 10
num_relevant = 10
optimizer = CPLEX.Optimizer
silent = false

gamma_range = [0.005, 0.01, 0.02]
cluster_sizes = div.([20_000, 50_000, 100_000], num_clusters)
num_features_range = [20_000, 50_000]
timings = zeros(length(cluster_sizes), length(num_features_range), length(gamma_range))
accuracies = zeros(length(cluster_sizes), length(num_features_range), length(gamma_range))
falses = zeros(length(cluster_sizes), length(num_features_range), length(gamma_range))

for seed in 1:5
    Random.seed!(11111 * seed)
    for (icsize, csize) in enumerate(cluster_sizes)
        cluster_sizes = fill(csize, num_clusters)
        memberships = Int[]
        for i in 1:num_clusters
            memberships = vcat(memberships, fill(i, csize))
        end
        memberships = shuffle(memberships)
        for (inum_features, num_features) in enumerate(num_features_range)
            (Xs, Ys, true_supp, true_weights) = SparClur2.construct_synthetic(num_features, cluster_sizes, num_relevant, snr = signal_ratio)
            for (gidx, gamma) = enumerate(gamma_range)
                tm = @elapsed (supp, weights) = SparClur2.solve_MIOP(Xs, Ys, num_relevant, gamma, optimizer, silent = false)
                acc = SparClur2.accuracy(supp, true_supp)
                fp = SparClur2.falsepositive(supp, true_supp)
                accuracies[icsize, inum_features, gidx] += acc
                falses[icsize, inum_features, gidx] += fp
                timings[icsize, inum_features, gidx] += tm
                @show acc, fp, num_features, gamma
            end # gamma
        end # p
    end # nobs
end # seeds

@show round.(timings ./ 5.0, digits = 2)
# round.(timings ./ 5.0, digits = 2) =
# [2.52 4.83;
# 4.51 10.65;
# 8.38 19.88]
# [1.7 4.67;
# 4.08 10.39;
# 7.73 19.4]
# [1.69 4.71;
# 4.08 10.36;
# 7.73 19.46]

CSV.write("output/accuracies.csv", DataFrame(reshape(accuracies, length(cluster_sizes) * length(num_features_range), length(gamma_range))))
CSV.write("output/falsepositives.csv", DataFrame(reshape(falses, length(cluster_sizes) * length(num_features_range), length(gamma_range))))
CSV.write("output/timings.csv", DataFrame(reshape(timings, length(cluster_sizes) * length(num_features_range), length(gamma_range))))
