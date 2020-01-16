"""
    solve_MIOP
"""
function solve_MIOP(
    Xs::Matrix{Float64},
    Ys::Vector{Float64},
    sparsity::Int,
    γ::Float64,
    optimizer;
    optimizer_params = NamedTuple(),
    )
    num_clusters = length(clusters)
    num_features = size(Xs[1], 2)
    num_samples  = sum(length(Y) for Y in Ys)
    dual_vars = [similar(Y) for Y in Ys]
    work = [similar(Y) for Y in Ys]
    bin_grad = zeros(num_features)
    bin_init = collect(1:num_relevant)
    initial_bound = regression_objective!(bin_grad, clusters, s0, γ, num_samples)

    model = Model()
    set_optimizer(model, () -> optimizer(; optimizer_params...))
    @variable(model, bin_var[i in 1:num_features], Bin, start = s0[i])
    @variable(model, approx_obj >= 0)
    @objective(model, Min, approx_obj)
    @constraint(model, sum(s) <= sparsity)
    @constraint(model, approx_obj >= initial_bound + dot(∇s, s - s0)) # turn on for a warm start

    function outer_approximation(cb_data)
        bin_val = callback_value(cb_data, bin_var)
        obj = regression_objective!(Xs, Ys, bin_grad, bin_val, γ, num_samples, dual_vars, work)
        con = @build_constraint(approx_obj >= obj + dot(bin_grad, bin_var - bin_val))
        MOI.submit(model, MOI.LazyConstraint(cb_data), con)
    end
    MOI.set(model, MOI.LazyConstraintCallback(), outer_approximation)

    optimize!(model)

    # recover optimal weights
    supp = getsupport(value.(bin_var))
    for i in 1:num_clusters
        Z = Xs[i][:, supp]
        # if not underdetermined
        if length(supp) <= size(Z, 1)
            # just do least squares
            solution.weights[i] = (Z' * Z) \ (Z' * Ys[i])
        else
            dual_var = calc_dual!(dual_var, γ, Z, Ys[i])
            solution.weights[i] = γ * Z' * dual_var
        end
    end

    return (supp, weights)
end

getsupport(s::Vector{Float64}) = find(s .> 0.5)

function calc_dual!(dual_var::Vector{Float64}, γ::Float64, Z::Matrix{Float64}, Y::Vector{Float64})
    k = size(Z, 2)
    # compute (Iₙ + γZZᵀ)⁻¹ Y  via Y - Z (Iₚ / γ + γZᵀZ)⁻¹ Zᵀ Y
    # unfortunately the size of Z matrices is unknown each iteration
    cap_matrix = Matrix(I / γ, k, k)
    ZtY = Z' * Y
    mul!(cap_matrix, Z', Z, true, true)
    ldiv!(cholesky!(Symmetric(cap_matrix)), ZtY)
    mul!(dual_var, Z, ZtY, -1, false)
    @. dual_var += Y
    return dual_var
end

# computes the objective of the loss function and updates bin_grad
function regression_objective!(
    Xs::Vector{Matrix{Float64}},
    Ys::Vector{Vector{Float64}},
    bin_grad::Vector{Float64},
    bin_val::Vector{Float64},
    γ::Float64,
    num_samples::Int,
    dual_vars::Vector{Float64},
    work::Vector{Float64},
    )
    bin_grad .= 0
    obj = 0.0
    supp = getsupport(s)
    k = length(supp)
    # compute optimal dual parameter for each cluster
    for i in eachindex(Xs)
        Z = view(Xs[i], :, supp)
        calc_dual!(dual_vars[i], γ, Z, Ys[i])

        mul!(work[i], Z', dual_vars[i])
        for i in eachindex(work[i])
            work[i] = abs2(work[i])
        end
        axpby!(-γ, work[i], true, bin_grad)

        obj += dot(Ys[i], dual_vars[i])
    end
    @. bin_grad /= 2num_samples
    obj /= 2num_samples

    return obj
end
