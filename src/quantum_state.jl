# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     main_language: julia
#     text_representation:
#       extension: .jl
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
################################################################################
############################# QUANTUM STATE  ###################################
################################################################################
abstract type QuantumState end

# %%
# Wrap qulacs.QuantumState
struct QulacsQuantumState <: QuantumState
    pyobj
end

# %%
function QulacsQuantumState(n_qubit::Int, int_state=0b0)
    state = QulacsQuantumState(qulacs.QuantumState(n_qubit))
    set_computational_basis!(state, int_state)
    state
end

# %%
function Base.copy(state::QulacsQuantumState)
    QulacsQuantumState(state.pyobj.copy())
end

# %%
function set_computational_basis!(state::QulacsQuantumState, int_state)
    state.pyobj.set_computational_basis(int_state)
end

# %%
function create_hf_state(n_qubit, n_electron)
    int_state = parse(Int, repeat("0", n_qubit-n_electron) * repeat("1", n_electron), base=2)
    state = QulacsQuantumState(n_qubit) 
    set_computational_basis!(state, int_state)
    state
end

# %%
function get_n_qubit(state::QulacsQuantumState)
    state.pyobj.get_qubit_count()
end

# %%
function get_vector(state::QulacsQuantumState)
    state.pyobj.get_vector()
end

# %%
function state_load!(state::QulacsQuantumState, vec_load)
    state.pyobj.load(vec_load)
end


# %%
function state_sampling(state::QulacsQuantumState, nshots)
    state.pyobj.sampling(nshots)
end

