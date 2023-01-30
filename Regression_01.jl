using CSV, DataFrames, Plots, Measurements, Polynomials

df = DataFrame(CSV.File("sample01.csv"))

x = df[!, 1]
Δx = df[!, 2]
y = df[!, 3]
Δy = df[!, 4]

scatter(x .± Δx, y .± Δy,
marker = 0, label = "experiment", legend = :topright)

f = fit(x, y, 2)

plot!(f, label = "fit")

savefig("Regression_01")
