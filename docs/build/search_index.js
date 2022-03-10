var documenterSearchIndex = {"docs":
[{"location":"index.html","page":"Home","title":"Home","text":"CurrentModule = QCMaterial","category":"page"},{"location":"index.html#QCMaterial","page":"Home","title":"QCMaterial","text":"","category":"section"},{"location":"index.html","page":"Home","title":"Home","text":"","category":"page"},{"location":"index.html","page":"Home","title":"Home","text":"Modules = [QCMaterial]","category":"page"},{"location":"index.html#QCMaterial.QulacsVariationalQuantumCircuit","page":"Home","title":"QCMaterial.QulacsVariationalQuantumCircuit","text":"Wrap a QulacsParametricQuantumCircuit object, which will not be copied.\n\n\n\n\n\n","category":"type"},{"location":"index.html#Base.copy-Tuple{QulacsVariationalQuantumCircuit}","page":"Home","title":"Base.copy","text":"Copy a QulacsVariationalQuantumCircuit object. This makes a copy of the underlying QulacsParametricQuantumCircuit object. \n\n\n\n\n\n","category":"method"},{"location":"index.html#QCMaterial.apply_qubit_op!-Tuple{QubitOperator, QuantumState, VariationalQuantumCircuit, QuantumState}","page":"Home","title":"QCMaterial.apply_qubit_op!","text":"Apply a qubit operator op to |stateket> and fit the result with circuit * |statebra>. The circuit object will be updated on exit. The squared norm of op * |stateket>  will be returned. state0bra will not be modified.\n\n\n\n\n\n","category":"method"},{"location":"index.html#QCMaterial.check_py_type-Tuple{PyCall.PyObject, String}","page":"Home","title":"QCMaterial.check_py_type","text":"Check if the Python type of a given PyObject matches the expected one\n\n\n\n\n\n","category":"method"},{"location":"index.html#QCMaterial.convert_openfermion_op-Tuple{Any, Any}","page":"Home","title":"QCMaterial.convert_openfermion_op","text":"Convertopenfermionop\n\nArgs:     nqubit (:class:int)     openfermionop (:class:openfermion.ops.QubitOperator) Returns:     :class:qulacs.Observable\n\n\n\n\n\n","category":"method"},{"location":"index.html#QCMaterial.divide_real_imag-Tuple{QubitOperator}","page":"Home","title":"QCMaterial.divide_real_imag","text":"Divide a qubit operator into the hermite and antihermite parts.\n\n\n\n\n\n","category":"method"},{"location":"index.html#QCMaterial.gen_p_t2-NTuple{4, Any}","page":"Home","title":"QCMaterial.gen_p_t2","text":"Generate pair dobule excitations\n\n\n\n\n\n","category":"method"},{"location":"index.html#QCMaterial.gen_t1-Tuple{Any, Any}","page":"Home","title":"QCMaterial.gen_t1","text":"Generate single excitations\n\n\n\n\n\n","category":"method"},{"location":"index.html#QCMaterial.generate_numerical_grad-Tuple{Any}","page":"Home","title":"QCMaterial.generate_numerical_grad","text":"Generates parallelized numerical grad_cost\n\n\n\n\n\n","category":"method"},{"location":"index.html#QCMaterial.get_thetas-Tuple{VariationalQuantumCircuit}","page":"Home","title":"QCMaterial.get_thetas","text":"Return a copy of variational parameters\n\n\n\n\n\n","category":"method"},{"location":"index.html#QCMaterial.get_transition_amplitude_with_obs-Tuple{VariationalQuantumCircuit, QuantumState, QubitOperator, QuantumState}","page":"Home","title":"QCMaterial.get_transition_amplitude_with_obs","text":"Compute <statebra| circuit^+ obs |stateket>, where obs is a hermite observable.\n\n\n\n\n\n","category":"method"},{"location":"index.html#QCMaterial.imag_time_evolve","page":"Home","title":"QCMaterial.imag_time_evolve","text":"Perform imaginary-time evolution.\n\nham_op:     Hamiltonian vc:     Variational circuit. The current value of variational parameters are     used as the initial value of the imaginary-time evolution. state0:     The initial state to which the Variational circuit is applied to taus:     list of imaginary times in ascending order     The first element must be 0.0.  return:     list of variational parameters at the given imaginary times.\n\n\n\n\n\n","category":"function"},{"location":"index.html#QCMaterial.mk_scipy_minimize","page":"Home","title":"QCMaterial.mk_scipy_minimize","text":"Make a wrapped scipy minimizer\n\n\n\n\n\n","category":"function"},{"location":"index.html#QCMaterial.num_theta-Tuple{VariationalQuantumCircuit}","page":"Home","title":"QCMaterial.num_theta","text":"Return the number of independent variational parameters\n\n\n\n\n\n","category":"method"},{"location":"index.html#QCMaterial.numerical_grad-Tuple{Any, Vector{Float64}}","page":"Home","title":"QCMaterial.numerical_grad","text":"Compute partial derivative of a given function at a point x\n\n\n\n\n\n","category":"method"},{"location":"index.html#QCMaterial.overlap-Tuple{VariationalQuantumCircuit, QulacsQuantumState, Vector{Float64}, Vector{Float64}}","page":"Home","title":"QCMaterial.overlap","text":"Compute <phi (thetabra) | phi(thetaket)>\n\n\n\n\n\n","category":"method"},{"location":"index.html#QCMaterial.parse_pauli_str-Tuple{Any}","page":"Home","title":"QCMaterial.parse_pauli_str","text":"Parse a tuple representing a Pauli string   When x is ((1, \"X\"), (5, \"Y\")), returns [1, 5], [PauliID.X, PauliID.Y]\n\n\n\n\n\n","category":"method"},{"location":"index.html#QCMaterial.solve_gs-Tuple{QubitOperator, VariationalQuantumCircuit, QuantumState}","page":"Home","title":"QCMaterial.solve_gs","text":"Compute ground state\n\n\n\n\n\n","category":"method"},{"location":"index.html#QCMaterial.tikhonov","page":"Home","title":"QCMaterial.tikhonov","text":"y = Ax tikhov regularization: minimize ||Ax-b||^2 + λ||x^2|| \n\n\n\n\n\n","category":"function"},{"location":"index.html#QCMaterial.transition-Tuple{OFQubitOperator, VariationalQuantumCircuit, QulacsQuantumState, Vector{Float64}, Vector{Float64}}","page":"Home","title":"QCMaterial.transition","text":"Compute <phi (thetabra) |H| phi(thetaket)>\n\n\n\n\n\n","category":"method"},{"location":"index.html#QCMaterial.uccgsd-Tuple{Any}","page":"Home","title":"QCMaterial.uccgsd","text":"Returns UCCGSD circuit.\n\n\n\n\n\n","category":"method"},{"location":"index.html#QCMaterial.update_circuit_param!-Tuple{UCCQuantumCircuit, Vector{Float64}}","page":"Home","title":"QCMaterial.update_circuit_param!","text":"Update circuit parameters\n\nthetas wil be copied.\n\n\n\n\n\n","category":"method"},{"location":"index.html#QCMaterial.update_circuit_param!-Tuple{VariationalQuantumCircuit, Vector{Float64}}","page":"Home","title":"QCMaterial.update_circuit_param!","text":"Update the values of the independent variational parameters\n\n\n\n\n\n","category":"method"},{"location":"index.html#QCMaterial.update_quantum_state!-Tuple{QulacsParametricQuantumCircuit, QulacsQuantumState}","page":"Home","title":"QCMaterial.update_quantum_state!","text":"Update a state using a circuit\n\n\n\n\n\n","category":"method"},{"location":"index.html#QCMaterial.update_quantum_state!-Tuple{UCCQuantumCircuit, QulacsQuantumState}","page":"Home","title":"QCMaterial.update_quantum_state!","text":"Update a state using a circuit\n\n\n\n\n\n","category":"method"},{"location":"index.html#QCMaterial.update_quantum_state!-Tuple{VariationalQuantumCircuit, QuantumState}","page":"Home","title":"QCMaterial.update_quantum_state!","text":"Update a state using a circuit\n\n\n\n\n\n","category":"method"},{"location":"index.html","page":"Home","title":"Home","text":"```@contents     Pages = [           \"chapter1/intro.md\",           \"chapter1/goma.md\",           \"chapter2/azarashi.md\",     ]     Depth = 3","category":"page"}]
}
