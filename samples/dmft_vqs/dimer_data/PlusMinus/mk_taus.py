import numpy as np
import irbasis

# Load fermionic IR basis
beta = 1000.0
wmax = 100.0
Lambda = beta * wmax

basis = irbasis.load("F", Lambda)
sl = basis.sl()
dim = np.sum(sl / sl[0] > 1e-5)

sp_x = basis.sampling_points_x(dim - 1)
sp_x_half = sp_x[0 : sp_x.size // 2]
sp_x_half = np.hstack((-1, sp_x_half, 0))

taus = 0.5 * beta * (sp_x_half + 1)
with open(f"sp_tau_p_{taus.size}.txt", "w") as f:
    print(taus.size, file=f)
    for t in taus:
        print(t, file=f)
