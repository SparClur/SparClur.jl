"""
    solve_MIOP
"""
function solve_MIOP(
    Xs::Vector{<:AbstractMatrix{Float64}},
    Ys::Vector{<:AbstractVector{Float64}},
    num_relevant::Int,
    γ::Float64,
    optimizer;
    optimizer_params = Dict(),
    silent::Bool = false,
    bin_init = vcat(ones(num_relevant), zeros(size(Xs[1], 2) - num_relevant)),
    )
    num_clusters = length(Ys)
    num_features = size(Xs[1], 2)
    num_samples  = sum(length(Y) for Y in Ys)
    dual_vars = [zeros(length(Y)) for Y in Ys]
    bin_grad = zeros(num_features)
    work = similar(bin_grad)
    initial_bound = regression_objective!(Xs, Ys, bin_grad, bin_init, γ, num_samples, dual_vars, work)

    # model = direct_model(optimizer())
    model = Model(optimizer)
    set_optimizer_attribute(model, MOI.Silent(), silent)
    for (attr, val) in optimizer_params
        set_optimizer_attribute(model, attr, val)
    end
    @variable(model, bin_var[i in 1:num_features], Bin, start = bin_init[i])
    @variable(model, approx_obj >= 0)
    @objective(model, Min, approx_obj)
    @constraint(model, sum(bin_var) <= num_relevant)
    @constraint(model, approx_obj >= initial_bound + dot(bin_grad, bin_var - bin_init)) # turn on for a warm start

    function outer_approximation(cb_data)
        bin_val = map(x -> callback_value(cb_data, x), bin_var)
        approx_obj_val = callback_value(cb_data, approx_obj)
        obj = regression_objective!(Xs, Ys, bin_grad, bin_val, γ, num_samples, dual_vars, work)
        if approx_obj_val < obj - 1e-3
            con = @build_constraint(approx_obj >= obj + dot(bin_grad, bin_var - bin_val))
            MOI.submit(model, MOI.LazyConstraint(cb_data), con)
        end
    end
    MOI.set(model, MOI.LazyConstraintCallback(), outer_approximation)

    optimize!(model)

    # recover optimal weights
    supp = getsupport(value.(bin_var))
    weights = [Float64[] for _ in eachindex(Ys)]
    for i in 1:num_clusters
        Z = Xs[i][:, supp]
        # if overdetermined
        # if length(supp) <= size(Z, 1)
        #     # just do least squares
        #     # Z_rank = rank(Z)
        #     fact = qr(Z)
        #     Z_rank = count(abs.(diag(fact.R)) .> 1e-8)
        #     @show Z_rank
        #     fact = qr(Z[:, 1:Z_rank])
        #     try
        #         @show "good"
        #         weights[i][1:Z_rank] .= fact \ Ys[i]
        #         @show sum(abs2, Ys[i] - Z * weights[i])
        #     catch e
        #         # my_rank = count(abs.(diag(fact.R)) .> 1e-13)
        #         # fact = qr(Z[:, 1:my_rank])
        #         # weights[i][1:my_rank] .= fact \ Ys[i]
        #         println(e)
        #         println("TODO debug")
        #         fact = bunchkaufman(Symmetric(Z' * Z + I * γ))
        #         weights[i] = fact \ (Z' * Ys[i])
        #         # @show sum(abs2, Ys[i] - Z * weights[i])
        #     end
        #
        #     # fact = bunchkaufman(Symmetric(Z' * Z), check = false)
        #     # if !issuccess(fact)
        #     #     @show eigvals(Z' * Z), size(Z)
        #     #     fact = bunchkaufman(Symmetric(Z' * Z) + I)
        #     # end
        #     # weights[i] = fact \ (Z' * Ys[i])
        # else
            dual_var = calc_dual!(dual_vars[i], γ, Z, Ys[i])
            weights[i] = γ * Z' * dual_var
        # end
    end

    return (supp, weights)
end

getsupport(s::Vector{Float64}) = findall(s .> 0.5)

function calc_dual!(
    dual_var::Vector{Float64},
    γ::Float64,
    Z::AbstractMatrix{Float64},
    Y::AbstractVector{Float64},
    )
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
    Xs::Vector{<:AbstractMatrix{Float64}},
    Ys::Vector{<:AbstractVector{Float64}},
    bin_grad::Vector{Float64},
    bin_val::Vector{Float64},
    γ::Float64,
    num_samples::Int,
    dual_vars::Vector{Vector{Float64}},
    work::Vector{Float64},
    )
    bin_grad .= 0
    obj = 0.0
    supp = getsupport(bin_val)
    k = length(supp)
    # compute optimal dual parameter for each cluster
    for i in eachindex(Xs)
        Z = view(Xs[i], :, supp)
        calc_dual!(dual_vars[i], γ, Z, Ys[i])

        mul!(work, Xs[i]', dual_vars[i])
        for j in eachindex(work)
            work[j] = abs2(work[j])
        end
        axpby!(-γ, work, true, bin_grad)
        obj += dot(Ys[i], dual_vars[i])
    end
    @. bin_grad /= 2num_samples
    obj /= 2num_samples

    return obj
end
