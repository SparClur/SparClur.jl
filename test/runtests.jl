using SparClur2
using Test
using LinearAlgebra
using CPLEX
using Random
# using GLPK

@testset "accuracy falsepositive" begin
    pred = [1; 2; 5]
    truth = [1; 2; 3]
    a = SparClur2.accuracy(pred, truth)
    f = SparClur2.falsepositive(pred, truth)
    @test a ≈ 2 / 3.0
    @test f ≈ 1 / 3.0
end

@testset "calc_dual!" begin
    n = p = 5
    Y = rand(n)
    X = rand(n, p)
    γ = 10.0
    dual_var = similar(Y)
    @test SparClur2.calc_dual!(dual_var, γ, X, Y) ≈ (I + γ * X * X') \ Y
end

@testset "algorithm" begin
    cluster_sizes = [
        [100, 200],
        [100, 400],
        [250, 250],
        [250, 500],
        [250, 750],
        [250, 1000],
        [500, 500],
        [500, 1000],
        [500, 1500],
        ]
    gamma_MIOP = 100.0
    num_features = 50
    num_relevant = 10

    @testset "MIOP" begin
        @testset "$c" for c in cluster_sizes
            Random.seed!(32)
            (Xs, Ys, true_supp, true_weights) = SparClur2.construct_synthetic(num_features, c, num_relevant)
            (supp, weights) = SparClur2.solve_MIOP(Xs, Ys, num_relevant, gamma_MIOP, CPLEX.Optimizer)
            @test supp ≈ true_supp
            @test weights ≈ weights
        end
    end

    @testset "relaxation" begin
        @testset "$c" for c in cluster_sizes
            gamma_relaxation = 1.0
            @testset "two clusters" begin
                Random.seed!(32)
                (Xs, Ys, true_supp, true_weights) = SparClur2.construct_synthetic(num_features, c, num_relevant)
                (supp, num_indices, weights) = SparClur2.solve_relaxation(Xs, Ys, num_relevant, gamma = gamma_relaxation)
                @test supp ≈ true_supp
                @test weights ≈ weights
            end
            @testset "one cluster" begin
                Random.seed!(64)
                (Xs, Ys, true_supp, true_weights) = SparClur2.construct_synthetic(num_features, [c[1]], num_relevant)
                (supp, num_indices, weights) = SparClur2.solve_relaxation(Xs, Ys, num_relevant, gamma = gamma_relaxation)
                @test supp ≈ true_supp
                @test weights ≈ weights
            end
        end
    end
end
