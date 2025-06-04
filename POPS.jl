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

train = train[1:20:end]

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
function sample_hypercube(bounds, number_of_committee_members, V, coeffs)
    # Setting up for sampling for committee
    local δθ = zeros((number_of_committee_members, size(bounds)[1]))

    for j = 1:number_of_committee_members
        U  = rand(Float64, size(bounds)[1])
	δθ[j, :] = (transpose(V) * (bounds[1] .+ bounds[2] .* U)) + coeffs
    end
    return δθ
end

function POPS!(model, train, solver, number_of_committee_members)
    # Creates Γ - the ridge regression/Tikhonov regulariser/ prior covariance matrix
    Γ      = ACEpotentials._make_prior(model, 4, nothing)
    
    # Constructs all the stuff we need to construct our problem
    A, Y, W= ACEfit.assemble(train, model)
    
    Ap     = Diagonal(W) * (A / Γ)
    Y_    = W .* Y
    result = ACEfit.solve(solver, Ap, Y_)
    local coeffs = Γ \ result["C"]
    
    percentile_clipping = 25.0
    dθ = POPS(A, Γ, coeffs, Y)
    bounds, V = hypercube(dθ, percentile_clipping)
    δθ = sample_hypercube(bounds, number_of_committee_members, V, coeffs)
    

    # The co_effs = [...] is because there are no
    # methods of set_committee!() that match a matrix input as the second argument, so
    # we need the co_coeffs as a vector of vectors
    ACEpotentials.Models.set_linear_parameters!(model, coeffs)
    co_coeffs = [δθ[i, :] for i = 1:size(δθ[:,1])[1]]
    set_committee!(model, co_coeffs);
end

function curve_fit()
    r_max = 1
    percentile_clipping=0.0
    number_of_committee_members = 100
    number_of_features = 7
    num = 2000
    x = collect(range(- r_max, r_max, num))
    true_func = cos.(2.0 * π.*(x.-0.5)) ./ (10.0 .- 5.0 .* x)
    design_matrix = zeros((length(x), number_of_features))
    for i = 1:number_of_features
    	design_matrix[:, i] = x .^ (i-1)
    end
    m, n = size(design_matrix)
    Γ = Matrix{Float64}(I, m, n)
    coeffs = design_matrix \ true_func
    dθ = POPS(design_matrix, Γ, coeffs, true_func)
    bounds, V = hypercube(dθ,percentile_clipping)
    println(size(V), size(bounds))
    println(size(design_matrix))
    δθ = sample_hypercube(bounds, number_of_committee_members, V, coeffs)
    println(bounds)
    plot(x, true_func, label = "True")
    y_predict = design_matrix * coeffs
    design_matrix = zeros((40, number_of_features))
    x_ = ones(40)
    for i = 1:number_of_features
        design_matrix[:, i] = x_ .^ (i-1)
    end
    scatter!(x_, design_matrix * coeffs, label = "Global")
    num = 40
    sin_pred = zeros(number_of_committee_members, num)
    x        = collect(range(-r_max, r_max, num))
    for j = 1:number_of_committee_members
        x_ = collect(range(-r_max, r_max, num))
        temp_coeffs = δθ[j, :]
        design_matrix = zeros((length(x_), number_of_features))
        for i = 1:number_of_features
    	    design_matrix[:, i] = x_ .^ (i-1)
        end
        sin_pred[j, :] =design_matrix * temp_coeffs
    end
    for i in 1:number_of_committee_members
        scatter!(x, sin_pred[i,:], primary=false, markercolor=i+1)
    
    end
    savefig("sin_fit.png")
end

num_in_committee = 20
POPS!(model, train, solver, num_in_committee)




function rattle_committee()
    rattle_levels = [0.1, 0.2, 0.3, 0.4, 0.5]
    E_POPS = zeros((length(rattle_levels), num_in_committee))
    E_mean = zeros(length(rattle_levels))
    
    for i = 1:length(rattle_levels)
        ucell   = bulk(sym, cubic = true)
        rattle!(ucell, rattle_levels[i])
        E, co_E = @committee potential_energy(ucell, model)
        E = ustrip(E); co_E = ustrip(co_E);
        E_mean[i]   = E
        E_POPS[i,:] = co_E
    end
end     

# Here onwards is stuff ripped from the tutorial on ACEpotentials+AtomsBase
# with an attempt to tackle committees too. The committee aspect for geometry
# optimisation is definitely not considered here, so this part is just a work in
# progress at the moment 
using GeomOpt
using AtomsBuilder, GeomOpt, AtomsCalculators, AtomsBase
using AtomsBase: FlexibleSystem, FastSystemi

n = 100
lattice_consts = collect(range(3.5, 5.5, n))
volumes        = zeros(n)
energies       = zeros(n)
co_energies    = zeros((n, num_in_committee))
using StaticArrays

for i = 1:length(lattice_consts)
    a = lattice_consts[i]u"Å"
    frac_positions = [[0.0,0.0,0.0],[0.5,0.5,0.0],[0.5,0.0,0.5],[0.5,0.5,0.5]]
    cartesian_positions = [SVector{3}(pos .* a.val) *u"Å" for pos in frac_positions]
    atoms = [Atom(sym, pos) for pos in cartesian_positions]
    cell_vectors = (SVector(a, 0.0u"Å", 0.0u"Å"),
		    SVector(0.0u"Å", a, 0.0u"Å"),
		    SVector(0.0u"Å", 0.0u"Å", a))
    fcc_cu = periodic_system(atoms, cell_vectors)
    vec_1, vec_2, vec_3  = fcc_cu.cell.cell_vectors
    volume = transpose(vec_1) * cross(vec_2, vec_3)
    E, co_E = @committee potential_energy(fcc_cu, model)
    energies[i] = ustrip(E)
    co_energies[i, :] = ustrip(co_E)
    @show E
    volumes[i] = ustrip(volume)
end
function OLS(x::Vector, y::Vector; degree::Int=3)
    n = length(x)
    X = zeros(n, degree + 1)
        for i in 0:degree
            X[:, i+1] .= x .^ i
        end
    coeffs = X \ y
    return coeffs, X
end
function QoI(coeffs)
    c0, c1, c2, c3 = coeffs
    a = 3.0*c3
    b = 2.0*c2
    c = c1
    if (b ^ 2 - 4 * a * c < 0)
	return (0, 0, 0)
    end
    v0= (- b + sqrt(b ^ 2 - 4 * a * c)) / (2 * a)
    e0= c0 + c1 * v0 + c2 * v0^2 + c3 * v0^3
    B = (2 * c2 + 6 * c3 * v0) * v0
    return v0, e0, B
end
i, j, k = QoI([-1,-2,-3,-4])
coeffs, design_matrix = OLS(volumes, energies)
v0, e0, B = QoI(coeffs)

function OLS(x, y)
    number_of_features = 4
    design_matrix = zeros((length(x), number_of_features))
    for i = 1:number_of_features
        design_matrix[:, i] = x .^ (i-1)
    end
    coeffs = design_matrix \ y
    return coeffs, design_matrix
end
plot(volumes, design_matrix * coeffs)

scatter!(volumes, energies)
savefig("Curve_fit.png")



scatter(volumes, energies, xlabel="Volume (Å)", ylabel="Energy (eV)")
for i = 1:num_in_committee
    scatter!(volumes, co_energies[:,i], primary=false, markercolor = i+1, markersize = 2.0)
end
savefig("eos_with_pops$(num_in_committee).png")

scatter([1,], [B,], xlabel = "Volume (Å)", ylabel = "Bulk Modulus (ev Å^-3)", label = "Global", markercolor = 1)
for i = 1:num_in_committee
    coeffs, _ = OLS(volumes, co_energies[:, i])
    v0, e0, B = QoI(coeffs)
    if (B == 0)
	continue
    end
    scatter!([1,], [B,], primary = false, markercolor = 2)
end
scatter!([1,], [B,], primary = false, markercolor = 1)
savefig("bogus_bulk.png")
#ucell    = bulk(sym, cubic = true)
#ucell, _ = GeomOpt.minimise(ucell, model; variablecell=true)

#Eparat, co_Eparat = (@committee potential_energy(ucell, model)) ./ length(ucell)

function _flexiblesystem(sys) 
    c3ll = cell(sys)
    particles = [ AtomsBase.Atom(species(sys, i), position(sys, i)) 
                  for i = 1:length(sys) ] 
    return FlexibleSystem(particles, c3ll)
end; 

# sys = _flexiblesystem(ucell) * (2, 2, 2)

# deleteat!(sys,1)

# vacancy_equil, result = GeomOpt.minimise(sys,model;variablecell=false)
# 
# E_vac, co_E_vac =  @committee potential_energy(vacancy_equil, model)
# E_def = E_vac - length(sys) * Eparat
# co_E_def = co_E_vac .- (length(sys) .* co_Eparat)
# @show E_def
# @show co_E_def
# 
# E_def = ustrip(E_def); co_E_def = ustrip(co_E_def)
# 
# scatter(ones(length(co_E_def)), co_E_def, label = "pops committee")
# scatter!([1,], [E_def,], markercolor = 2, label = "Mean")
# savefig("functions_Vacancy_formation_energy_POPS.png")
