# Adding imports used in the ACE+ACEbase tutorial
using ExtXYZ, Unitful, AtomsCalculators, Distributed, ACEpotentials
using LinearAlgebra
using AtomsCalculators: potential_energy
using Random
using Plots
using Statistics
using AtomsBuilder, GeomOpt, AtomsCalculators, AtomsBase,
      AtomsCalculatorsUtilities

sym            = :Cu
model          = ace1_model(elements = [sym,], order = 3, totaldegree = [20,16,12])
train, test, _ = ACEpotentials.example_dataset("Zuo20_$sym")
solver         = ACEfit.BLR(; factorization = :svd)

train = train[1:29:end]

function POPS(A, Γ, coeffs, Y)
    local H = (transpose(Γ) * Γ + transpose(A) * A)
    local dθ = zeros(size(A))
    for i = 1:size(A[:, 1])[1]
        local V  = H \ A[i, :]
        leverage = transpose(A[i, :]) * V
        E        = transpose(A[i, :]) * coeffs
        dy       = Y[i] - E
        dθ[i, :] = (dy / leverage) .* V 
    end
    return dθ
end

function hypercube(dθ, percentile_clipping)
    U, S, V    = svd(dθ)

    projected = dθ * V
    
    # This is the naive implementation where we just use a percentile clipping to get 
    # rid of insane parameter bounds
    lower  = [quantile(projected[:, i], percentile_clipping / 100.0) for i in 1:size(projected[1,:])[1]]
    upper  = [quantile(projected[:, i], 1.0 - percentile_clipping / 100.0) for i in 1:size(projected[1,:])[1]] 
    bounds = hcat(lower, upper)
    
    return bounds, V
end
function sample_hypercube(bounds, number_of_committee_members, V)
    # Setting up for sampling for committee
    local δθ = zeros((number_of_committee_members, size(bounds)[1]))

    for j = 1:number_of_committee_members
        U  = rand(Float64, size(bounds)[1])
	δθ[j, :] = (transpose(V) * (bounds[1] .+ bounds[2] .* U))
    end
    return δθ
end

function POPS!(model, train, solver, number_of_committee_members)
    # Creates Γ - the ridge regression/Tikhonov regulariser
    Γ      = ACEpotentials._make_prior(model, 4, nothing)
    
    # Constructs all the stuff we need to construct our problem
    A, Y, W= ACEfit.assemble(train, model)
    
    Ap     = Diagonal(W) * (A / Γ)
    Y_    = W .* Y
    result = ACEfit.solve(solver, Ap, Y_)
    local coeffs = Γ \ result["C"]
    
    # The "H" matrix
#     H = (transpose(Γ)*Γ + transpose(A)*A)
#     
#     H_inv = H^-1
#     
#     # For storing the pointwise corrections to the parameters
#     dθ = ones(size(A))
#     
#     # This is the basic loop of getting updated parameters to build the 
#     # hypercube from - this is algorithm 1 in appendix E of the paper.
#     for i = 1:size(A[:, 1])[1]
#         leverage = transpose(A[i, :]) * H_inv * A[i, :]
#         E        = transpose(A[i, :]) * coeffs
#         dy       = Y[i] - E
#         dθ[i, :] = (dy / leverage) .* H_inv * A[i, :]
##     end
    percentile_clipping = 45.0
    dθ = POPS(A, Γ, coeffs, Y)
    bounds, V = hypercube(dθ, percentile_clipping)
    δθ = sample_hypercube(bounds, number_of_committee_members, V)
    # This is the hypercube part of the process - algorithm 2 in appendix E.
#    U, S, V = svd(dθ)
#    
#    projected = dθ * V
#    
#    # This is the naive implementation where we just use a percentile clipping to get 
#    # rid of insane parameter bounds
#    percentile_clipping = 25.0
#    lower  = [quantile(projected[:, i], percentile_clipping / 100.0) for i in 1:size(projected[1,:])[1]]
#    upper  = [quantile(projected[:, i], 1.0 - percentile_clipping / 100.0) for i in 1:size(projected[1,:])[1]] 
#    bounds = hcat(lower, upper)
#    
#    # Setting up for sampling for committee
#    num_in_committee = 50
#    δθ = zeros((num_in_committee, size(bounds)[1]))
#    E_mean = sum(A * coeffs)
#    E_POPS = zeros(num_in_committee)
#    
#    for j = 1:num_in_committee
#        U  = rand(Float64, size(bounds)[1])
#        δθ[j, :] = transpose(V) * (lower .+ upper .* U)
#    end
    

    # I am not entirely sure, but I think that this is the correct handling for 
    # setting committee coefficients. The co_effs = [...] is because there are no
    # methods of set_committee!() that match a matrix input as the second argument, so
    # we need the co_coeffs as a vector of vectors
    ACEpotentials.Models.set_linear_parameters!(model, coeffs)
    co_coeffs = [δθ[i, :] for i = 1:size(δθ[:,1])[1]]
    set_committee!(model, co_coeffs);
end


function plot!(; E_POPS = E_POPS, E_mean = E_mean, name = "POPS")
    y_ = ones(length(E_POPS))
    x_ = ones(length(E_POPS))
    y_[1:end] = E_POPS
    scatter(x_, y_, label = "pops committee")
    scatter!([1,], [E_mean,], markercolor = 2, label = "mean")
    xlabel!("")
    savefig("$name.png")
end
function plot!(; E_POPS, E_mean, name = "POPS", rattles)
    scatter(rattles, E_mean, label = "global coeffs")
    for i =1:length(rattles)
        scatter!(ones(length(E_POPS)) .* rattles[i], E_POPS[i, :], markercolor = 2, label = "POPS committee")
    end
    xlabel!("Rattle")
    ylabel!("E (eV)")
    savefig("$name.png")
end

percentile_clipping= 25.0
number_of_committee_members = 3
number_of_features = 5
x = collect(range(- 3, 3, 10))
true_func = sin.(x) .+ exp.(4.0 .* x) ./ (3 .* x .^ 2)
design_matrix = zeros((length(x), number_of_features))
for i = 1:number_of_features
   design_matrix[:, i] = x .^ i
end
m, n = size(design_matrix)
Γ = Matrix{Float64}(I, m, n)
coeffs = design_matrix \ true_func
dθ = POPS(design_matrix, Γ, coeffs, true_func)
bounds, V = hypercube(dθ, percentile_clipping)
println(size(V), size(bounds))
println(size(design_matrix))
δθ = sample_hypercube(bounds, number_of_committee_members, V)
y_predict = design_matrix * coeffs
plot(x, true_func, label = "True")
scatter!(x, design_matrix * coeffs, label = "Global")
for j = 1:number_of_committee_members
    x_ = collect(range(-3, 3, 20))
    temp_coeffs = δθ[j, :]
    design_matrix = zeros((length(x_), number_of_features))
    for i = 1:number_of_features
        design_matrix[:, i] = x_ .^ i
    end
    Y_pred = design_matrix * temp_coeffs
    scatter!(x_, Y_pred)
end
savefig("sin_fit.png")


num_in_committee = 1000
POPS!(model, train, solver, num_in_committee)



# I am not entirely sure, but I think that this is the correct handling for 
# setting committee coefficients. The co_effs = [...] is because there are no
# methods of set_committee!() that match a matrix input as the second argument, so
# we need the co_coeffs as a vector of vectors


 
rattle_levels = [0.1, 0.2, 0.3, 0.4, 0.5]
E_POPS = zeros((length(rattle_levels), num_in_committee))
E_mean = zeros(length(rattle_levels))	
 
for i = 1:length(rattle_levels)
    ucell   = bulk(sym, cubic = true)
    rattle!(ucell, rattle_levels[i])
    E, co_E = @committee potential_energy(ucell, model)
    E = ustrip(E); co_E = ustrip(co_E);
    E_mean[i]   = E
    E_POPS[i,:] = co_E.+E 
end
 
plot!(E_POPS=E_POPS, E_mean=E_mean, name = "+functions_$(percentile_clipping)_p_clip_rattles_pop", rattles = rattle_levels)


# Here onwards is stuff ripped from the tutorial on ACEpotentials+AtomsBase
# with an attempt to tackle committees too. The committee aspect for geometry
# optimisation is definitely not considered here, so this part is just a work in
# progress at the moment 
using GeomOpt
using AtomsBuilder, GeomOpt, AtomsCalculators, AtomsBase
using AtomsBase: FlexibleSystem, FastSystemi

function _flexiblesystem(sys)
    c3ll = cell(sys)
    particles = [ AtomsBase.Atom(species(sys, i), position(sys, i)) for i = 1:length(sys) ]
    return FlexibleSystem(particles, c3ll)
end;

ucell    = bulk(sym, cubic = true)
ucell, _ = GeomOpt.minimise(ucell, model; variablecell=true)

Eparat, co_Eparat = (@committee potential_energy(ucell, model)) ./ length(ucell)

sys = _flexiblesystem(ucell) * (2, 2, 2)

deleteat!(sys,1)

vacancy_equil, result = GeomOpt.minimise(sys,model;variablecell=false)

E_vac, co_E_vac =  @committee potential_energy(vacancy_equil, model)
E_def = E_vac - length(sys) * Eparat
co_E_def = co_E_vac .- (length(sys) .* co_Eparat)

@show E_def
@show co_E_def

E_def = ustrip(E_def); co_E_def = ustrip(co_E_def)

scatter(ones(length(co_E_def)), co_E_def, label = "pops committee")
scatter!([1,], [E_def,], markercolor = 2, label = "Mean")
savefig("functions_Vacancy_formation_energy_POPS.png")
