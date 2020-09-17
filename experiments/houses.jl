# sparclur, sparclur relaxation, sparse relaxation, lasso, ORT point
# bonus: sparse no relaxation, ORT linear

import SparClur2
import CSV
import CPLEX
import MLDataUtils: kfolds, rescale!
import StatsBase
using DataFrames # TODO remove

data_dir = "experiments/data"
normalized_dir = joinpath(data_dir, "normalized")

# TODO normalize data

optimizer = CPLEX.Optimizer
optimizer_params = ("CPX_PARAM_TILIM" => 30, "CPXPARAM_MIP_Tolerances_MIPGap" => 1e-2)

gamma_range = [1e-6, 0.001, 0.1, 10.0, 100.0]
q_range = 5:10
silent = true
depths = 3:3

function unscale_pred(pred, y_mean, y_scal)
    return (pred .* y_scal) .+ y_mean
end

# TODO add a bias term
function normalize_data(depth)
    train_data = CSV.read(joinpath(data_dir, "const_depth$(depth)_train.csv"), DataFrame)
    num_train = size(train_data, 1)
    test_data = CSV.read(joinpath(data_dir, "const_depth$(depth)_test.csv"), DataFrame)
    num_test = size(test_data, 1)

    train_X = train_data[:, 1:(end - 3)]
    test_X = test_data[:, 1:(end - 3)]
    mean_X = StatsBase.mean.(eachcol(train_X))
    scal_X = StatsBase.std.(eachcol(train_X))
    rescaled_train_X = (train_X .- mean_X') ./ scal_X'
    rescaled_test_X = (test_X .- mean_X') ./ scal_X'

    train_Y = train_data[:, end - 2]
    test_Y = test_data[:, end - 2]
    mean_Y = StatsBase.mean(train_Y)
    scal_Y = StatsBase.std(train_Y)
    rescaled_train_Y = (train_Y .- mean_Y) ./ scal_Y

    # TODO delete, just debugging
    num_cols = size(train_data, 2)
    check = hcat(train_data[:, 1:(end - 3)], train_data[:, end - 2])
    rescale!(check)
    @assert check == hcat(rescaled_train_X, rescaled_train_Y)

    # NOTE first column is bias, last column is memberships
    rescaled_train = hcat(ones(num_train), Array(rescaled_train_X), rescaled_train_Y, train_data[:, end]) # TODO why Array?
    rescaled_test = hcat(ones(num_test), Array(rescaled_test_X), Array(test_Y), test_data[:, end])

    isdir(normalized_dir) || mkdir(normalized_dir)
    CSV.write(joinpath(normalized_dir, "const_depth$(depth)_train.csv"), DataFrame(rescaled_train)) # TODO shouldn't need DataFrame()
    CSV.write(joinpath(normalized_dir, "const_depth$(depth)_test.csv"), DataFrame(rescaled_test))
    return (mean_Y, scal_Y)
end

# allow clusters to be an input in case of empty clusters
function make_clusters(X_list, Y_list, memberships_list, clusters)
    Xs = AbstractMatrix{Float64}[view(X_list, 1:1, :) for _ in clusters] # TODO
    Ys = AbstractVector{Float64}[[] for _ in clusters] # TODO view
    for (c, cluster) in enumerate(clusters)
        idxs = findall(x -> x .== cluster, memberships_list)
        Xs[c] = view(X_list, idxs, :)
        Ys[c] = view(Y_list, idxs)
    end
    return (Xs, Ys)
end

# heuristic for warm starts. uses relaxation algorithm, only trying a fixed set of gammas.
function get_warm_start(Xs, Ys, q::Int)
    best_mse = Inf
    best_supp = []
    for gamma in 10.0 .^ (-1:2)
        (supp, _, weights) = SparClur2.solve_relaxation(Xs, Ys, q, gamma = gamma)
        Y_pred = [Xs[i][:, supp] * weights[i] for i in eachindex(Ys)]
        mse = sum(sum(abs2, Y_pred[i] - Ys[i]) for i in eachindex(Ys))
        if mse < best_mse
            best_mse = mse
            best_supp = supp
        end
    end
    warm_start = zeros(size(Xs[1], 2))
    warm_start[best_supp] .= 1
    return warm_start
end

function train_sparclur(depth; relaxation = true, ignore_coordination = false)
    data_train = Array(CSV.read(joinpath(normalized_dir, "const_depth$(depth)_train.csv"), DataFrame)) # TODO
    num_features = size(data_train, 2)
    X_big_list = data_train[:, 1:(end - 2)]
    Y_big_list = data_train[:, end - 1]
    num_obs = size(Y_big_list, 1)
    memberships_list = Int.(data_train[:, end])
    clusters = unique(memberships_list) # not contiguous
    folds = kfolds(collect(1:num_obs), k = 5)
    mse_scores = zeros(length(q_range), length(gamma_range))
    valid_io = open("validation_depth$(depth).csv", "w")
    println(valid_io, "fold,q,gamma,mse")
    for (fold_idx, (train_idxs, valid_idxs)) in enumerate(folds)
        # split data
        X_train_list = view(X_big_list, train_idxs, :)
        X_valid_list = view(X_big_list, valid_idxs, :)
        Y_train_list = view(Y_big_list, train_idxs)
        Y_valid_list = view(Y_big_list, valid_idxs)
        memberships_train = view(memberships_list, train_idxs)
        memberships_valid = view(memberships_list, valid_idxs)
        # group by leaves
        (Xs_train, Ys_train) = make_clusters(X_train_list, Y_train_list, memberships_train, clusters)
        (Xs_valid, Ys_valid) = make_clusters(X_valid_list, Y_valid_list, memberships_valid, clusters)
        # grid
        for (q_idx, q) in enumerate(q_range), (gamma_idx, gamma) in enumerate(gamma_range)
            if relaxation
                if ignore_coordination
                    supp = Vector{Int}[]
                    weights = Vector{Float64}[]
                    # one leaf = single cluster at a time
                    for c in eachindex(clusters)
                        (supp_c, num_supp, weights_c) = SparClur2.solve_relaxation([Xs_train[c]], [Ys_train[c]], q, gamma = gamma)
                        push!(supp, supp_c[1:num_supp])
                        push!(weights, weights_c[1])
                    end
                else
                    (supp_c, _, weights) = SparClur2.solve_relaxation(Xs_train, Ys_train, q, gamma = gamma) # TODO
                    # repeat the same support for all clusters (tidy this line)
                    supp = fill(supp_c, length(clusters))
                end
            else
                if ignore_coordination
                    supp = Vector{Int}[]
                    weights = Vector{Float64}[]
                    # one leaf = single cluster at a time
                    for c in eachindex(clusters)
                        warm_start = get_warm_start([Xs_train[c]], [Ys_train[c]], q)
                        (supp_c, weights_c) = SparClur2.solve_MIOP([Xs_train[c]], [Ys_train[c]], q, gamma, optimizer, silent = silent, optimizer_params = optimizer_params, bin_init = warm_start)
                        push!(supp, supp_c)
                        push!(weights, weights_c[1])
                    end
                else
                    warm_start = get_warm_start(Xs_train, Ys_train, q)
                    (supp_c, weights) = SparClur2.solve_MIOP(Xs_train, Ys_train, q, gamma, optimizer, silent = silent, optimizer_params = optimizer_params, bin_init = warm_start)
                    # repeat the same support for all clusters (tidy this line)
                    supp = fill(supp_c, length(clusters))
                end
            end
            Ys_pred = [Xs_valid[c][:, supp[c]] * weights[c] for c in eachindex(clusters)]
            mse_scores[q_idx, gamma_idx] += sum(sum(abs2, Ys_pred[c] - Ys_valid[c]) for c in eachindex(clusters))
            println(valid_io, "$(fold_idx),$(q),$(gamma),$(mse_scores[q_idx, gamma_idx])")
            flush(valid_io)
        end
    end
    close(valid_io)

    (_, best_idx) = findmin(mse_scores)
    (best_q, best_gamma) = (q_range[best_idx[1]], gamma_range[best_idx[2]])
    # retrain
    (Xs_big, Ys_big) = make_clusters(X_big_list, Y_big_list, memberships_list, clusters)
    if relaxation
        if ignore_coordination
            supp = Vector{Int}[]
            weights = Vector{Float64}[]
            # one leaf = single cluster at a time
            for c in eachindex(clusters)
                (supp_c, num_supp, weights_c) = SparClur2.solve_relaxation([Xs_big[c]], [Ys_big[c]], best_q, gamma = best_gamma)
                @show supp_c
                push!(supp, supp_c[1:num_supp])
                push!(weights, weights_c[1])
            end
        else
            (supp_c, num_supp, weights) = SparClur2.solve_relaxation(Xs_big, Ys_big, best_q, gamma = best_gamma)
            supp = fill(supp_c[1:num_supp], length(clusters))
        end
    else
        if ignore_coordination
            supp = Vector{Int}[]
            weights = Vector{Float64}[]
            # one leaf = single cluster at a time
            for c in eachindex(clusters)
                warm_start = get_warm_start([Xs_big[c]], [Ys_big[c]], best_q)
                (supp_c, weights_c) = SparClur2.solve_MIOP([Xs_big[c]], [Ys_big[c]], best_q, best_gamma, optimizer, silent = silent, optimizer_params = optimizer_params, bin_init = warm_start)
                push!(supp, supp_c)
                push!(weights, weights_c[1])
            end
        else
            warm_start = get_warm_start(Xs_big, Ys_big, best_q)
            (supp_c, weights) = SparClur2.solve_MIOP(Xs_big, Ys_big, best_q, best_gamma, optimizer, silent = silent, optimizer_params = optimizer_params, bin_init = warm_start)
            supp = fill(supp_c, length(clusters))
        end
    end
    @show supp
    return (best_gamma, (supp, weights))
end

function train_lasso()
end

function test_sparclur()
    ignore_coord = false
    use_relaxation = true
    res = zeros(length(depths))
    for (depth_idx, depth) in enumerate(depths)
        (mean_Y, scal_Y) = normalize_data(depth)
        (best_gamma, (supp, weights)) = train_sparclur(depth, relaxation = use_relaxation, ignore_coordination = ignore_coord)
        data_test = Array(CSV.read(joinpath(normalized_dir, "const_depth$(depth)_test.csv"), DataFrame)) # TODO
        X_list = data_test[:, 1:(end - 2)]
        Y_list = data_test[:, end - 1]
        memberships_list = Int.(data_test[:, end])
        clusters = unique(memberships_list)
        (Xs_test, Ys_test) = make_clusters(X_list, Y_list, memberships_list, clusters)
        Ys_pred = [Xs_test[c][:, supp[c]] * weights[c] for c in eachindex(clusters)]
        Ys_pred = unscale_pred.(Ys_pred, mean_Y, scal_Y)
        mse = sum(sum(abs2, Ys_pred[c] - Ys_test[c]) for c in eachindex(clusters))
        # baseline_mse = sum(sum(abs2, mean(Ys_test[c]) .- Ys_test[c]) for c in eachindex(clusters))
        mean_all = mean(Y_list)
        baseline_mse = sum(sum(abs2, mean_all .- Ys_test[c]) for c in eachindex(clusters))
        res[depth_idx] = 1 - mse / baseline_mse
        open("output/housing_depth_$(depth)_ignore_coord_$(ignore_coord)_relaxation_$(use_relaxation).txt", "w") do io
            println(io, supp)
            println(io, weights)
            println(io, res[depth_idx])
            println(io, best_gamma)
        end
    end
    return res
end
res = test_sparclur()
# @show res


function test_ort_point()
    res = zeros(length(depths))
    for (depth_idx, depth) in enumerate(depths)
        data_test = Array(CSV.read(joinpath(data_dir, "const_depth$(depth)_test.csv"), DataFrame))
        Y_pred = data_test[:, end - 1]
        Y_test = data_test[:, end - 2]
        mse = sum(abs2, Y_pred - Y_test)
        baseline_mse = sum(abs2, Y_test .- mean(Y_test))
        res[depth_idx] = 1 - mse / baseline_mse
    end
    return res
end
# @show test_ort_point()
#
function test_ort_lasso()
    res = zeros(length(depths))
    for (depth_idx, depth) in enumerate(depths)
        data_test = Array(CSV.read("experiments/data/linear_depth$(depth)_test.csv", DataFrame))
        Y_pred = data_test[:, end - 1]
        Y_test = data_test[:, end - 2]
        mse = sum(abs2, Y_pred - Y_test)
        baseline_mse = sum(abs2, Y_test .- mean(Y_test))
        res[depth_idx] = 1 - mse / baseline_mse
    end
    return res
end
@show test_ort_lasso()
