#!/usr/bin/env python

import numpy as np
import curie_weiss_periodic as cp
from numpy.linalg import eigh
from scipy.sparse import dia_matrix

def get_jmat_pbc(lsize, beta):
    N = lsize
    J = dia_matrix((N, N))
    mid_diag = np.floor(N/2).astype(int)
    for i in xrange(1,mid_diag+1):
        elem = pow(i, -beta)
        J.setdiag(elem, k=i)
        J.setdiag(elem, k=-i)
    for i in xrange(mid_diag+1, N):
        elem = pow(N-i, -beta)
        J.setdiag(elem, k=i)
        J.setdiag(elem, k=-i)
    return J.toarray()


N = 4
t = np.linspace(0.0,100.0,10000)
b = 15.0

J = get_jmat_pbc(N, b)

freqs = np.linspace(1.0,14.0,5)


for f in freqs:
    p = cp.ParamData(hopmat = -J, lattice_size=N, ampl=10.0, omega=f, times=t, hz=1.0, jx=1.0)
    h = cp.Hamiltonian(p)
    E, U = eigh(h.hamiltmat + h.trans_hamilt)
    out = cp.run_dyn(p, initstate=U[0])
    time = out["t"]
    sz = out["sz"]
    filename = "sz_frequency" + str(f) + ".dat"
    np.savetxt(filename, np.vstack((time,sz)).T)
