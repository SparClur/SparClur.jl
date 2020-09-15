# modified from https://github.com/jeanpauphilet/SubsetSelection.jl, credit to the original contributors

const sgtol = 1e-6

mutable struct PolyakCache
    best_upper::Float64
    best_lower::Float64
end

struct Cache
    g::Vector{Float64}
    sortperm::Vector{Int}
    function Cache(n::Int, p::Int)
        new(
          Vector{Float64}(undef,n),
          Vector{Int}(undef,p),
        )
    end
end

# heuristic
function getdelta!(stepping_factor::Float64, X::Matrix{Float64}, Y::Vector{Float64}, a::Vector{Float64},
    nabla::Vector{Float64}, indices::Vector{Int}, n_indices::Int, gamma::Float64, pc::PolyakCache, astar::Vector{Float64})

    lower_bound = dual_bound(X, Y, a, indices, n_indices, gamma)
    upper_bound = primal_bound(X, Y, gamma, indices, n_indices, astar)
    if upper_bound < pc.best_upper
        pc.best_upper = upper_bound
    end
    (lower_bound > pc.best_lower) && (pc.best_lower = lower_bound)
    if lower_bound - sgtol > upper_bound + sgtol
          error("Bounds overlap. LB = $lowerbound and UB = $upper_bound.")
    end
    return stepping_factor * (pc.best_upper - lower_bound) / sum(abs2, nabla)
end

# heuristic
function updatefactor(pc::PolyakCache, old_ub::Float64, stepping_factor::Float64, times_unchanged::Int)
    if pc.best_upper >= old_ub - sgtol
        times_unchanged += 1
    else
        times_unchanged = 0
        old_ub = pc.best_upper
    end
    if times_unchanged > 30
        stepping_factor /= 2
        times_unchanged = 0
    end
    return (old_ub, times_unchanged, stepping_factor)
end

function solve_relaxation(
    Xs::Vector{Matrix{Float64}},
    Ys::Vector{Vector{Float64}},
    num_relevant::Int;
    stepping_factor = 0.5,
    max_iter::Int = 100,
    averaging::Bool = true,
    gamma = 1 / sqrt(size(Xs[1], 1))
    )

    # heuristic scaling (ub - lb) / norm^2
    @assert 0.5 <= stepping_factor <= 2
    num_clusters = length(Xs)
    n = size(Xs[1], 1)
    p = size(Xs[1], 2)
    cluster_sizes = zeros(Int, num_clusters)
    a = fill(Float64[], num_clusters)
    nabla = fill(Float64[], num_clusters)
    for k in 1:num_clusters
        cluster_size = length(Ys[k])
        cluster_sizes[k] = cluster_size
        a[k] = -Ys[k]
        nabla[k] = zeros(cluster_size)
    end
    avg_a = deepcopy(a)
    astar = deepcopy(a)
    cache = Cache(n, p)
    indices = collect(1:num_relevant)
    n_indices = n_indices_max = num_relevant

    # we will compute the dual bound incorrectly unless indicies match initial a
    n_indices = partial_min!(indices, num_relevant, Xs, a, gamma, cache)

    pc = PolyakCache(Inf, -Inf)

    # dual sub-gradient algorithm
    iter = 2
    # best upper bound so far and the number of times it hasn't improved
    old_ub = pc.best_upper
    times_unchanged = 0
    while iter < max_iter
        # gradient ascent on a
        for _ in 1:min(1.0, div(p, n_indices)), k in 1:num_clusters
            grad_dual!(nabla[k], Ys[k], Xs[k], a[k], indices, n_indices, gamma)
            δ = getdelta!(stepping_factor, Xs[k], Ys[k], a[k], nabla[k], indices, n_indices, gamma, pc, astar[k])
            @. a[k] += δ * nabla[k]
        end
        # if the upper bound did not go down
        (old_ub, times_unchanged, stepping_factor) = updatefactor(pc, old_ub, stepping_factor, times_unchanged)
        for k in 1:num_clusters
            @. avg_a[k] = ((iter - 1) * avg_a[k] + a[k]) / iter
        end
        # minimization w.r.t. s
        n_indices = partial_min!(indices, num_relevant, Xs, a, gamma, cache)
        iter += 1
    end

    # sparse estimator
    n_indices = partial_min!(indices, num_relevant, Xs, averaging ? avg_a : a, gamma, cache)

    w = Vector{Float64}[]
    for k in 1:num_clusters
        @views A = Xs[k][:, indices]
        if size(A, 1) >= size(A, 2)
            push!(w, (A' * A) \ (A' * Ys[k]))
        else
            warn("You should have minbucket ≥ k for good estimates.")
            a0 = similar(Ys[k])
            calc_dual!(a0, gamma, A, Ys[k])
            push!(w, gamma * A' * a0)
        end
    end

    return (indices, indices, w)
  end


function grad_dual!(g::Vector{Float64}, Y::Vector{Float64}, X::Matrix{Float64}, a::Vector{Float64}, indices::Vector{Int}, n_indices::Int, gamma::Float64)
    @. g -= a + Y
    for j in 1:n_indices
        @views x = X[:, indices[j]]
        d = dot(x, a)
        @. g -= gamma * d * x
    end
    return
end

function partial_min!(indices, n_indices::Int, Xs::Vector{Matrix{Float64}}, a::Vector{Vector{Float64}}, gamma::Float64, cache::Cache)
    perm = cache.sortperm
    p = size(Xs[1], 2)
    # cluster case is maxⱼ: ∑ₖ αₖ'XⱼXⱼ'αₖ
    s = zeros(p)
    for j in 1:p, k in 1:length(Xs)
        @views s[j] += dot(Xs[k][:, j], a[k])^2
    end
    sortperm!(perm, s, rev = true)
    @views indices[1:n_indices] .= perm[1:n_indices]
    @views sort!(indices[1:n_indices])
    return n_indices
end

function ax_squared(X::Matrix{Float64}, a::Vector{Float64}, indices::Vector{Int}, n_indices::Int)
    @views return sum(dot(a, X[:, indices[j]]) ^ 2 for j in 1:n_indices)
end

function primal_bound(X::Matrix{Float64}, Y::Vector{Float64}, gamma::Float64, indices::Vector{Int}, n_indices::Int, astar::Vector{Float64})
    calc_dual!(astar, gamma, X[:, indices], Y)
    axsum = ax_squared(X, astar, indices, n_indices)
    return dot(Y, astar) - (sum(abs2, astar) + gamma * axsum) / 2
end

function dual_bound(X::Matrix{Float64}, Y::Vector{Float64}, a::Vector{Float64}, indices::Vector{Int}, n_indices::Int, gamma::Float64)
    axsum = ax_squared(X, a, indices, n_indices)
    return dot(Y, a) - (sum(abs2, a) + gamma * axsum) / 2
end
