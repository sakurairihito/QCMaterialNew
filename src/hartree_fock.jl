module HartreeFock

export extract_tij_Uijlk, solve_scf

using LinearAlgebra
import PyCall: pyimport


"""
Extract tij and Uijlk from a FermionOperator object, representing a Hamiltonian.
The Hamiltonian must be number conserving, and is allowed to contain upto two-body operators.
"""
function extract_tij_Uijlk(ham)
    ofermion = pyimport("openfermion")
    FermionOperator = ofermion.ops.operators.FermionOperator
    count_qubits = ofermion.utils.operator_utils.count_qubits
    normal_ordered = ofermion.transforms.opconversions.normal_ordered

    ham = normal_ordered(ham)
    if ham.many_body_order() > 4
        throw(DomainError("Allowed only upto two-body operators!"))
    end
    if !ham.is_two_body_number_conserving()
        throw(DomainError("ham must be number conserving!"))
    end
    nflavors = count_qubits(ham)
    T = ham.zero().constant isa Real ? Float64 : Complex{Float64}
    tij = zeros(T, nflavors, nflavors)
    Uijlk = zeros(T, nflavors, nflavors, nflavors, nflavors)
    for (ops, constant) in ham.terms
        flavors = collect((o[1]+1 for o in ops))
        if length(ops) == 2
            tij[flavors[1], flavors[2]] += constant
        elseif length(ops) == 4
            Uijlk[flavors[1], flavors[2], flavors[3], flavors[4]] += 2*constant
        else
            throw(DomainError("Invalid many_body_order!"))
        end
    end
    tij, Uijlk
end

"""
Compute Tij
"""
function compute_Tij(tij, barUijlk, rhoij)
    Tij = copy(tij)
    N = size(tij)[1]
    for i in 1:N, j in 1:N
        for k in 1:N, l in 1:N
            Tij[i,j] += barUijlk[i,k,j,l] * rhoij[k,l]
        end
    end
    Tij
end

function compute_rhoij!(rhoij, evecs, nelec)
    rhoij .= 0.0
    N = size(rhoij)[1]
    for e in 1:nelec
        for i in 1:N, j in 1:N
           rhoij[i, j] += evecs[i,e] * evecs[j,e]
        end
    end
end


"""
Solve the Hartree-Fock equation self-consistently:
   hat{H} = sum_{ij=1}^N t_{ij} hat{c}^dagger_i hat{c}_j 
       + frac{1}{2} sum_{ijkl=1}^N U_{ijlk}hat{c}^dagger_i hat{c}^dagger_j hat{c}_k hat{c}_l
"""
function solve_scf(tij::Array{T,2}, Uijlk::Array{T,4}, rhoij0::Array{T,2}, nelec::Integer, niter::Integer, mixing::Float64) where T
    # Antisymmetric Coulomb tensor
    barUijlk = Uijlk - permutedims(Uijlk, [1, 2, 4, 3])
    rhoij = copy(rhoij0)
    rhoij_out = copy(rhoij)
    for iter in 1:niter
        Tij = compute_Tij(tij, Uijlk, rhoij)
        e = eigen(Hermitian(Tij))
        compute_rhoij!(rhoij_out, e.vectors, nelec)
        @. rhoij = (1-mixing) * rhoij + mixing * rhoij_out
    end
    rhoij
end

end