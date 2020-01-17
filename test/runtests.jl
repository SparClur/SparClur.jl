using SparClur2
using Test
using LinearAlgebra
# using CPLEX
using Random
# using GLPK
using Gurobi
include(joinpath(dirname(@__DIR__()), "src", "utils", "construct_synthetic.jl"))

@testset "calc_dual!" begin
    n = p = 5
    Y = rand(n)
    X = rand(n, p)
    γ = 10.0
    dual_var = similar(Y)
    @test SparClur2.calc_dual!(dual_var, γ, X, Y) ≈ (I + γ * X * X') \ Y

    # @test regression_objective!([X], [Y]
    #     Xs::Vector{Matrix{Float64}},
    #     Ys::Vector{Vector{Float64}},
    #     bin_grad::Vector{Float64},
    #     bin_val::Vector{Float64},
    #     γ::Float64,
    #     num_samples::Int,
    #     dual_vars::Vector{Vector{Float64}},
    #     work::Vector{Float64},
    #     )
end

cluster_sizes = [
    # [100, 200],
    # [100, 400],
    # [250, 250],
    # [250, 500],
    [250, 750],
    # [250, 1000],
    # [500, 500],
    # [500, 1000],
    # [500, 1500],
    ]

γ = 100.0
num_features = 50
num_relevant = 10

@testset "($i, $j)" for (i, j) in cluster_sizes
    Random.seed!(32)
    (Xs, Ys, supp, weights) = construct_synthetic(num_features, [i, j], num_relevant)
    (supp, weights) = SparClur2.solve_MIOP(Xs, Ys, num_relevant, γ, Gurobi.Optimizer)
end
