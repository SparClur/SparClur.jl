# sparclur, sparclur relaxation, sparse relaxation, lasso, ORT point
# bonus: sparse no relaxation, ORT linear

using CSV
using DataFrames

# five trees not one
# validation tree

function train_sparclur()
    for relaxation in [true, false]
        for depth in 0:5
            data_train = DataFrame(CSV.File("const_depth$(depth)_train.csv"))
            data_test = DataFrame(CSV.File("const_depth$(depth)_train.csv"))
            X_big_list = data_train[:, 1:(end - 3)]
            Y_big_list = data_train[:, end - 3]
            memberships_list = data_train[:, end]
            folds = kfolds(X_big_list, Y_big_list, k = 5)
            for ((X_train_list, Y_train_list), (X_valid_list, Y_valid_list)) in folds
                for q in 1:10, gamma in gamma_range
                    # use memberships to get clusters
                    # train
                    # valid_score = apply to validation data
                    # update MSE data
                end
            end
        end
    end
end

# validation X and Y are picked a priori (we don't know what leaves they'll end up on)
# and they are the same validation data used by optimal trees in their grid search.
