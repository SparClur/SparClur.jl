using SparClur2
using Test

@testset "calc_dual!" begin
    n = p = 5
    Y = rand(n)
    X = rand(n, p)
    γ = 10.0
    dual_var = similar(Y)
    @test SparClur2.calc_dual!(dual_var, γ, X, Y) ≈ (I + γ * X * X') \ Y
end

# include(joinpath(dirname("@__DIR__()"), "construct_synthetic.jl"))
#
# cluster_sizes = [
#     [100, 200],
#     [100, 400],
#     [250, 250],
#     [250, 500],
#     [250, 750],
#     [250, 1000],
#     [500, 500],
#     [500, 1000],
#     [500, 1500],
#     ]
#
# γ = 100.0
# num_relevant = 10
#
# @testset "($i, $j)" for (i, j) in cluster_sizes
#     (Xs, Ys, supp, weights) = construct_synthetic(nfeatures, [i, j], num_relevant)
#
#     single_test(data, truth, sparsity, γ, solver)
# end
