#!/usr/bin/env python

import numpy as np
from numpy import linalg as LA
import curie_weiss_periodic as cp
import matplotlib.pyplot as plt

N = 10
t = np.linspace(0.0,200.0,1000)

J = np.diagflat(np.ones(N), 1) + np.diagflat(np.ones(N), -1)
J[0,N] = 1
J[N,0] = 1


p = cp.ParamData(hopmat = J, lattice_size=N, omega=5.0, times=t, hz=2.0, jx=-1.0, jz=0.0)
h = cp.Hamiltonian(p)
E, U = LA.eigh(h.hamiltmat + h.trans_hamilt)
out = cp.run_dyn(p, initstate=U[0])
szcorr = out["szsz"]

np.savetxt("sz_corr.txt", szcorr)



fig = plt.figure(figsize=(10,20))
ax = fig.add_subplot(111)
cax = ax.matshow(data, interpolation='nearest', aspect='auto')
fig.colorbar(cax)

ax.set_yticklabels(['']+np.array2string(t))

plt.show()
