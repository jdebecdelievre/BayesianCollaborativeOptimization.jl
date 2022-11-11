using Plots
using CSV

function plotlearningcurves(folder)
    for f=readdir(joinpath(folder,"training"))
        ite = joinpath(folder,"training",f)
        for dis=readdir(ite)
            p = plot()
            for th=readdir(joinpath(ite,dis))
                np = 1
                if th.endswith(".csv")
                    hist = CSV.read(joinpath(ite,dis,th), NamedTuple)
                    plot!(p,hist.loss .+ eps(), yscale=:log10, label="$np")
                    np += 1
                end
            end
            savefig(p, joinpath(folder,"training",ite,dis,"trainingcurves.pdf"))
        end
end