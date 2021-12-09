
"Hamiltonian dimer"

from triqs.operators import c, c_dag
up, down = 0, 1
n_0up = c_dag(up, 0) * c(up, 0)
n_0down = c_dag(down, 0) * c(down, 0)
n_1up = c_dag(up, 1) * c(up, 1)
n_1down = c_dag(down, 1) * c(down, 1)

#coulomb
#U (c_dag(up, 0) * c(up, 0) * c_dag(down, 0) * c(down, 0))

#chemical potential
#-μ (c_dag(up, 0) * c(up, 0) + c_dag(down, 0) * c(down, 0))

#hybritization term

#V (c_dag(up, 0) * c(up, 1) + c_dag(up, 1) * c(up, 0)
#+ c_dag(down, 0) * c(down, 1) + c_dag(down, 1) * c(down, 0))

#bath energy level
#ε_b (c_dag(up, 1) * c(up, 1) + c_dag(down, 1) * c(down, 1))

U = 1.0
mu = 1.0
V = 1.0
ε_b = 1.0

H = U * n_0up * n_0down - mu * (n_0up + n_0down)  
H += -V * (c_dag(up, 0) * c(up, 1) + c_dag(up, 1) * c(up, 0)+ c_dag(down, 0) * c(down, 1) + c_dag(down, 1) * c(down, 0))
H += ε_b * (n_1up + n_1down)

print('H =', H)


beta = 1000.0 # inverse temperature
fundamental_operators = [c(up,0), c(down,0),c(up,1), c(down,1)]

from pyed.TriqsExactDiagonalization import TriqsExactDiagonalization
ed = TriqsExactDiagonalization(H, fundamental_operators, beta)

print(r'Z =', ed.get_partition_function())
print(r'\Omega =', ed.get_free_energy())
print(r'\rho =')
print(ed.ed.get_density_matrix())

"Thermal expectation value"
print('<n_0up>   =', ed.get_expectation_value(n_0up))
print('<n_0down> =', ed.get_expectation_value(n_0down))
print('<n_0up * n_0down> =', ed.get_expectation_value(n_0up * n_0down))

print('<n_1up>   =', ed.get_expectation_value(n_1up))
print('<n_1down> =', ed.get_expectation_value(n_1down))
print('<n_1up * n_1down> =', ed.get_expectation_value(n_1up * n_1down))

"Imaginary time single-particle Green's function"
import itertools
from triqs.gf import GfImTime
from triqs.gf import transpose
import matplotlib.pyplot as plt
from triqs.plot.mpl_interface import oplot

plt.figure(figsize=(8, 6))

subp = [2, 2, 1]
n_points = 50000
g_of_tau = []
for i1, i2 in itertools.product(range(2), repeat=2):
    g_tau = GfImTime(name=r'$g_{%i%i}$' % (i1, i2), beta=beta, 
                     statistic='Fermion', n_points=n_points, target_shape=(1,1))
    ed.set_g2_tau(g_tau[0,0], c(up, i1), c_dag(up, i2))
    g_of_tau.append(g_tau) 
    plt.subplot(*subp); subp[-1] += 1
    oplot(g_tau)
    plt.xlim(0,10)
    plt.show()



import numpy as np

n_points = 30000
beta = 1000
taus = np.linspace(0, beta, n_points)
print(taus)

def write_to_txt(file_name, x, y):
    with open(file_name,mode="w") as f:
        for i in range(0, len(x)):
            print(x[i], " ", -y[i].real[0][0], file=f)

write_to_txt("g_of_tau_dimer_tau_minus.txt", taus, g_of_tau[0].data[::-1])


#with open('sample.txt', 'w') as f:
#    print(g_of_tau[0].data, file=f)
#with open('sample.txt') as f:
#  print(f.readlines())