import numpy
import irbasis
import scipy
import scipy.integrate as integrate
import matplotlib
import matplotlib.pyplot as plt

# Load fermionic IR basis
beta = 1000.0
wmax = 100.0
Lambda = beta * wmax
print(wmax, beta, Lambda)

basis = irbasis.load("F", Lambda)
dim = basis.dim()
print(dim)

sp_x = basis.sampling_points_x(dim - 1)
# sp_x

sp_tau = beta * (sp_x + 1) / 2
# sp_tau

# Add tau=0 and beta
sp_tau = numpy.hstack((0.0, sp_tau, beta))

# If tau = beta/2 is not included, add this
if sp_tau.size % 2 == 0:
    sp_tau = numpy.sort(numpy.hstack((0.5 * beta, sp_tau)))

half = (len(sp_tau) + 1) / 2
# half = 70
tau_plus_dimer = sp_tau[:70]


def write_to_txt(file_name, x):
    with open(file_name, mode="w") as f:
        print(len(x), file=f)
        for i in range(0, len(x)):
            print(x[i], file=f)


write_to_txt("sp_tau.txt", tau_plus_dimer)
print("a")
