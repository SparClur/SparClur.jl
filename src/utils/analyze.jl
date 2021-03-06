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
