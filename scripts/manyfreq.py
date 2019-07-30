#!/usr/bin/env python

import numpy as np
import curie_weiss_periodic as cp
import matplotlib.pyplot as plt

N = 4
t = np.linspace(0.0,20.0,1000)

J = np.diagflat(np.ones(N), 1) + np.diagflat(np.ones(N), -1)
J[0,N] = 1
J[N,0] = 1


freqs = np.linspace(1.0,14.0,5)


for f in freqs:
    p = cp.ParamData(hopmat = -J, lattice_size=N, ampl=10.0, omega=f, times=t, hz=1.0, jx=1.0)
    out = cp.run_dyn(p)
    time = out["t"]
    sz = out["sz"]
    filename = "sz_frequency" + str(f) + ".dat"
    np.savetxt(filename, np.vstack((time,sz)).T)
