from triqs.operators import c, c_dag
up, down = 0, 1
n_0up = c_dag(up, 0) * c(up, 0)
n_0down = c_dag(down, 0) * c(down, 0)

n_1up = c_dag(up, 1) * c(up, 1)
n_1down = c_dag(down, 1) * c(down, 1)

n_2up = c_dag(up, 2) * c(up, 2)
n_2down = c_dag(down, 2) * c(down, 2)

n_3up = c_dag(up, 3) * c(up, 3)
n_3down = c_dag(down, 3) * c(down, 3)


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
H += -V * (c_dag(up, 0) * c(up, 1) + c_dag(up, 1) * c(up, 0)+ c_dag(down, 0) * c(down, 1) + c_dag(down, 1) * c(down, 0)\
          + c_dag(up, 0) * c(up, 2) + c_dag(up, 2) * c(up, 0)+ c_dag(down, 0) * c(down, 2) + c_dag(down, 2) * c(down, 0)\
          + c_dag(up, 0) * c(up, 3) + c_dag(up, 3) * c(up, 0)+ c_dag(down, 0) * c(down, 3) + c_dag(down, 3) * c(down, 0))
H += ε_b * (n_1up + n_1down + n_2up + n_2down + n_3up + n_3down )

print('H =', H)

beta = 1000.0 # inverse temperature

fundamental_operators = [c(up,0), c(down,0),c(up,1), c(down,1),c(up,2), c(down,2),c(up,3), c(down,3)]

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

print('<n_2up>   =', ed.get_expectation_value(n_2up))
print('<n_2down> =', ed.get_expectation_value(n_2down))
print('<n_2up * n_2down> =', ed.get_expectation_value(n_2up * n_2down))

print('<n_3up>   =', ed.get_expectation_value(n_3up))
print('<n_3down> =', ed.get_expectation_value(n_3down))
print('<n_3up * n_3down> =', ed.get_expectation_value(n_3up * n_3down))

"Imaginary time single-particle Green's function"
import itertools
from triqs.gf import GfImTime
import matplotlib.pyplot as plt
from triqs.plot.mpl_interface import oplot

plt.figure(figsize=(8, 6))

subp = [2, 2, 1]
n_points = 30000
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
 

import csv
with open('dimer_tau_plus.csv', 'wt') as f:
    writer = csv.writer(f)
    #writer.writerows(beta)
    #writer.writerows(n_points)
    writer.writerows(g_of_tau[0].data)
    
#print(g_of_tau[0].data) 

#with open('sample.txt', 'w') as f:
#    print(g_of_tau[0].data, file=f)
#with open('sample.txt') as f:
#  print(f.readlines())