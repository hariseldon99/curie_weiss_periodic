#!/usr/bin/env python

import numpy as np
import curie_weiss_periodic as cp
import matplotlib.pyplot as plt

N = 4
t = np.linspace(0.0,200.0,1000)

J = np.diagflat(np.ones(N), 1) + np.diagflat(np.ones(N), -1)
J[0,N] = 1
J[N,0] = 1


p = cp.ParamData(hopmat = J, lattice_size=N, omega=5.0, times=t, hz=0.5, jx=-0.5)
out = cp.run_dyn(p)
time = out["t"]
rho = out["defect_density"]
sz = out["sz"]
plt.subplot(2, 1, 1)
plt.plot(time,rho)
plt.subplot(2, 1, 2)
plt.plot(time, sz)
plt.show()
