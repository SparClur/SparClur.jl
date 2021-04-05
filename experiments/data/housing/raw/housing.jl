using CSV, DataFrames
import StatsBase

# Dataset from https://www.kaggle.com/harlfoxem/housesalesprediction
data = DataFrame(CSV.File("kc_house_data.csv"))
X = select(data, Not([:id, :date, :price, :zipcode]))

X_scale = Matrix(X)
dt = StatsBase.fit(StatsBase.ZScoreTransform, X_scale, dims=1)
X = DataFrame(StatsBase.transform(dt, X_scale), names(X))

y = data[:, :price]
y .-= StatsBase.mean(y)
y ./= StatsBase.std(y)

function output_df(lnr, X, y, filename)
  out = copy(X)
  out.y_true = y
  out.y_pred = IAI.predict(lnr, X)
  out.leaf = IAI.apply(lnr, X)
  CSV.write(filename, out)
end

for seed = 1:5

  (X_train, y_train), (X_test, y_test) =
      IAI.split_data(:regression, X, y, seed=seed, train_proportion=0.5)


  isdir("results") || mkpath("results")
  for depth = 0:5
    grid = IAI.GridSearch(
        IAI.OptimalTreeRegressor(random_seed=seed, minbucket=50),
        max_depth=depth:depth,
    )
    IAI.fit!(grid, X_train, y_train)
    @show depth
    @show IAI.score(grid, X_train, y_train)
    @show IAI.score(grid, X_test, y_test)
    @show IAI.get_learner(grid)

    output_df(grid, X_train, y_train,
              "results/const_depth$(depth)_train_s$(seed).csv")
    output_df(grid, X_test, y_test,
              "results/const_depth$(depth)_test_s$(seed).csv")
    IAI.write_html("results/const_depth$(depth)_tree_s$(seed).html",
                   IAI.get_learner(grid))
  end


  for depth = 0:5
    grid = IAI.GridSearch(
        IAI.OptimalTreeRegressor(
            random_seed=seed,
            minbucket=50,
            regression_sparsity=:all,
            regression_features=:sqft_living,
        ),
        max_depth=depth:depth,
    )
    IAI.fit!(grid, X_train, y_train)
    @show depth
    @show IAI.score(grid, X_train, y_train)
    @show IAI.score(grid, X_test, y_test)
    @show IAI.get_learner(grid)

    output_df(grid, X_train, y_train,
              "results/sqft_depth$(depth)_train_s$(seed).csv")
    output_df(grid, X_test, y_test,
              "results/sqft_depth$(depth)_test_s$(seed).csv")
    IAI.write_html("results/sqft_depth$(depth)_tree_s$(seed).html",
                   IAI.get_learner(grid))
  end


  for depth = 0:5
    grid = IAI.GridSearch(
        IAI.OptimalTreeRegressor(
            random_seed=seed,
            minbucket=50,
            regression_sparsity=:all,
        ),
        max_depth=depth:depth,
    )
    IAI.fit!(grid, X_train, y_train)
    @show depth
    @show IAI.score(grid, X_train, y_train)
    @show IAI.score(grid, X_test, y_test)
    @show IAI.get_learner(grid)

    output_df(grid, X_train, y_train,
              "results/linear_depth$(depth)_train_s$(seed).csv")
    output_df(grid, X_test, y_test,
              "results/linear_depth$(depth)_test_s$(seed).csv")
    IAI.write_html("results/linear_depth$(depth)_tree_s$(seed).html",
              IAI.get_learner(grid))
  end
end
