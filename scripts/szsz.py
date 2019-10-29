#!/usr/bin/env python

import numpy as np
from numpy import linalg as LA
import curie_weiss_periodic as cp
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib import rc
rc('text', usetex=True)

N = 10
omega = 5.0
hz = 25.0

t = np.linspace(0.0,300.0,1000)

J = np.diagflat(np.ones(N), 1) + np.diagflat(np.ones(N), -1)
J[0,N] = 1
J[N,0] = 1

p = cp.ParamData(hopmat = J, lattice_size=N, omega=omega, times=t, hz=hz, jx=1.0, jz=0.0)
h = cp.Hamiltonian(p)
E, U = LA.eigh(h.hamiltmat + h.trans_hamilt)
out = cp.run_dyn(p, initstate=U[0])
sz = out["sz"]
dd = out["defect_density"]
szcorr = out["szsz"]

np.savetxt("sz.txt", sz)
np.savetxt("defect_density.txt", dd)
np.savetxt("sz_corr.txt", szcorr)

plt.switch_backend('agg')
plt.figure()
plt.suptitle(r'$N = {}$, '.format(N) + r'$h_z = {}$, '.format(hz) + r'$\omega = {}$'.format(omega), fontsize=25)
plt.subplot(121)
plt.title(r'$\langle\sigma^z_i\sigma^z_j\rangle(t)$')
plot = plt.imshow(szcorr, interpolation='nearest', aspect='auto')

nx = np.arange(N)
nx = nx[nx != N/2]
no_of_labels = nx.shape[0]
x_positions = np.arange(N)
x_labels = nx
plt.xticks(x_positions, x_labels)
plt.xlabel(r'$j$', fontsize=15)

nt = t.shape[0]
no_of_labels = 7
step_t = int(nt / (no_of_labels - 1))
t_positions = np.arange(0,nt,step_t)
t_labels = np.round(t[::step_t], decimals=2)
plt.yticks(t_positions, t_labels)
plt.ylabel(r't', fontsize=15)

plt.colorbar(plot)

plt.subplot(122)
plt.plot(t,sz)
plt.xlabel(r't', fontsize=15)
plt.ylabel(r'$\overline{\sigma_z}$', fontsize=15)

plt.savefig("corr.svg", format='svg')
plt.show()
