import StatsBase: mean

"""
    predict(X::Array{Float64,2}, memberships::Vector{Int}, support::Vector{Int}, weights::Vector{Vector{Float64}})
Returns a vector `y` using predictions from `weights`, assuming each row i in `X`
belongs to the cluster memberships[i].
"""
function predict(X::Array{Float64,2}, memberships::Vector{Int}, support::Vector{Int}, weights::Vector{Vector{Float64}})
    y = zeros(size(X, 1))
    for i in 1:size(X, 1)
        y[i] = dot(X[i, support], weights[memberships[i]])
    end
    return y
end

mse(y_pred, y_true) = sum(abs2, (y_pred .- y_true))
"""
    rsquared(X::Array{Float64,2}, ytest::Vector{Float64}, memberships::Vector{Int}, support::Vector{Int}, weights::Vector{Vector{Float64}})
"""
function rsquared(X::Array{Float64,2}, ytest::Vector{Float64}, memberships::Vector{Int}, support::Vector{Int}, weights::Vector{Vector{Float64}})
    ypred = predict(X, memberships, support, weights)
    err = mse(ypred, ytest)
    baseline = mse(ypred, mean(ytest))
    return 1 - err / baseline
end

"""
    accuracy(pred::Vector{Int}, truth::Vector{Int})
Returns proportion of indices in `truth` that are also in `pred`.
"""
function accuracy(pred::Vector{Int}, truth::Vector{Int})
    detected = 0
    for t in truth
        (t in pred) && (detected += 1)
    end
    return detected / length(truth)
end

"""
    falsepositive(pred::Vector{Int}, truth::Vector{Int})
Returns the proportion of indices in `pred` that are not in `truth`.
"""
function falsepositive(pred::Vector{Int}, truth::Vector{Int})
    detected = 0
    for p in pred
        (p in truth) || (detected += 1)
    end
    return detected / length(pred)
end
