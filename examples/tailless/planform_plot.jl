using Plots
using LinearAlgebra
HOME = pwd()
include("$(pwd())/examples/tailless/tailless.jl")
T = Tailless()
##
xle_tip = root_chord/4 + span / 2 * tand(sweep) - tip_chord / 4

x = [
    0., xle_tip, xle_tip + tip_chord, root_chord, 0.
]
y = [0., half_span, half_span, 0., 0.]

ye = half_span * 16/20
y_elevon =  [ye, half_span, half_span, ye, ye]

xte_tip = xle_tip + tip_chord
elev_chord = root_chord + (tip_chord-root_chord)*16/20
x_e = xle_tip * 16/20 + elev_chord * 3/4
x_elevon  = [
    x_e, xle_tip+tip_chord*3/4, xle_tip + tip_chord, xle_tip * 16/20 + elev_chord, x_e
]

p = plot(y, -x, aspectratio = :equal, label="", axis=false, grid=false, ticks=nothing)
plot!(p, y_elevon, -x_elevon, aspectratio = :equal, label="", axis=false, grid=false, ticks=nothing)
savefig(p,"$HOME/examples/tailless/planform.pdf")