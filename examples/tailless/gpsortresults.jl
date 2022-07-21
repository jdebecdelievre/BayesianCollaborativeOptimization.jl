using DataFrames
using CSV


gdir = "/home/adgboost/bco/tailless_mar8"

D   = (Array(CSV.read("$gdir/plotdat_D.csv", DataFrame))[:,2:end].+1)/2
xcg = (Array(CSV.read("$gdir/plotdat_xcg.csv", DataFrame))[:,2:end].+1)/2
a1  = (Array(CSV.read("$gdir/plotdat_a1.csv", DataFrame))[:,2:end].+1)/2
a3  = (Array(CSV.read("$gdir/plotdat_a3.csv", DataFrame))[:,2:end].+1)/2
R   = (Array(CSV.read("$gdir/plotdat_R.csv", DataFrame))[:,2:end].+1)/2

gZ = [
[[R[i,j], D[i,j], a1[i,j], a3[i,j], xcg[i,j]] for i=1:size(D,1)]
for j=1:size(D,2)];

D   = (Array(CSV.read("$gdir/plotdat_aero_D.csv", DataFrame))[:,2:end].+1)/2
xcg = (Array(CSV.read("$gdir/plotdat_aero_xcg.csv", DataFrame))[:,2:end].+1)/2
a1  = (Array(CSV.read("$gdir/plotdat_aero_a1.csv", DataFrame))[:,2:end].+1)/2
a3  = (Array(CSV.read("$gdir/plotdat_aero_a3.csv", DataFrame))[:,2:end].+1)/2
R   = (Array(CSV.read("$gdir/plotdat_aero_R.csv", DataFrame))[:,2:end].+1)/2

gZaero = [
[[R[i,j], D[i,j], a1[i,j], a3[i,j], xcg[i,j]] for i=1:size(D,1)]
for j=1:size(D,2)];

D   = (Array(CSV.read("$gdir/plotdat_struc_D.csv", DataFrame))[:,2:end].+1)/2
xcg = (Array(CSV.read("$gdir/plotdat_struc_xcg.csv", DataFrame))[:,2:end].+1)/2
a1  = (Array(CSV.read("$gdir/plotdat_struc_a1.csv", DataFrame))[:,2:end].+1)/2
a3  = (Array(CSV.read("$gdir/plotdat_struc_a3.csv", DataFrame))[:,2:end].+1)/2
R   = (Array(CSV.read("$gdir/plotdat_struc_R.csv", DataFrame))[:,2:end].+1)/2

gZstruc = [
[[R[i,j], D[i,j], a1[i,j], a3[i,j], xcg[i,j]] for i=1:size(D,1)]
for j=1:size(D,2)];
##
sqJaero = map((z,zs)->norm.(z-zs), gZ,gZaero)
sqJstruc = map((z,zs)->norm.(z-zs), gZ,gZstruc)
gmetric = map(z->abs.(first.(z).+(-zopt[1][1])),gZ) + sqJaero + sqJstruc
for m=gmetric
    for i=2:length(m)
        m[i] = min(m[i],m[i-1])
    end
end

##

first.(Z) 
##
first.(gZ)[1:7]

##
hdir = "examples/tailless"
Z = [load_data("$hdir/xp$i/struc.jld2").Z for i=1:7];
Zaero = [load_data("$hdir/xp$i/aero.jld2").Zs for i=1:7];

#
zs = [struc_subspace(z[1],"tmp.txt") for z=gZ];
gzs = first.(gZstruc)
sum(v->abs.(v),(zs-gzs))/30

##
za = [[aero_subspace(z[2:end],"tmp.txt") for z=Z] for Z=gZ];
gza = [[z[2:end] for z=Z] for Z=gZaero]
sum(v->abs.(v),(za-gza))/30