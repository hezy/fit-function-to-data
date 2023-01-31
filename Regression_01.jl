using CSV, DataFrames, Plots, Measurements, Polynomials

df = DataFrame(CSV.File("sample01.csv"))

p = scatter(df."x" .± df."dx", df."y" .± df."dy",
marker = 0, label = "experiment", legend = :topright)

f = fit(x, y, 2)

p = plot!(f, label = "fit")

display(p)

savefig("Regression_01")
