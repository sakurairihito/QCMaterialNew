import numpy as np
import irbasis
from matplotlib import pyplot as plt

Lambda = 1e5
data = np.loadtxt("gf_dimer_vqs_plus_minus.txt")
taus = data[:, 0]
gtau = data[:, 1]
beta = taus[-1]
wmax = Lambda / beta
print(f"beta = {beta}")
print(f"wmax = {wmax}")
basis_f = irbasis.load("F", Lambda)

# Use all basis functions
dim = basis_f.dim()
alll = np.arange(dim)
xs = 2 * taus / beta - 1
Ftau = np.sqrt(2 / beta) * basis_f.ulx(alll[:, None], xs[None, :]).T
print(Ftau.shape, gtau.shape)

gl = np.linalg.pinv(Ftau) @ gtau

# sp_n = np.arange(0,10000)
sp_n = np.arange(0, 1000000, 100)
print(sp_n)
hatF = np.sqrt(beta) * basis_f.compute_unl(sp_n)

giwn = hatF @ gl

wn = (2 * sp_n + 1) * np.pi / beta
plt.plot(wn, np.abs(giwn.real), label="Re")
plt.plot(wn, np.abs(giwn.imag), label="Im")
plt.plot(wn, 1 / wn, label="1/wn")
# plt.plot(wn, giwn.imag, label="Im")
plt.legend()
plt.xscale("log")
plt.yscale("log")
plt.savefig("matsubara.pdf")
plt.show()
