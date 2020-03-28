#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from quspin.basis import spinless_fermion_basis_1d
from quspin.operators import hamiltonian
import general_functions as f
import plot_functions as p

t = 1 # intracell hopping,
tp = 1.1 # intercell hopping,
Delta = 0 # onsite potential
L = 20 # atomic sites in a chain
bc = 'obc' # boundary conditions 'pbc' or 'obc'

def ssh_model_real_space(t, tp, Delta, L, bc = 'pbc'):
    stagg_pot = [[Delta*(-1)**i, i] for i in range(L)]
    if bc == 'pbc':
        hop_intracell_pm = [[t, i, (i+1)%L] for i in range(0, L, 2)]
        hop_intracell_mp = [[-t, i, (i+1)%L] for i in range(0, L, 2)]
        hop_intercell_pm = [[tp, i, (i+1)%L] for i in range(1, L, 2)]
        hop_intercell_mp = [[-tp, i, (i+1)%L] for i in range(1, L, 2)]
    elif bc == 'obc':
        hop_intracell_pm = [[t, i, (i+1)] for i in range(0, L-1, 2)]
        hop_intracell_mp = [[-t, i, (i+1)] for i in range(0, L-1, 2)]
        hop_intercell_pm = [[tp, i, (i+1)] for i in range(1, L-1, 2)]
        hop_intercell_mp = [[-tp, i, (i+1)] for i in range(1, L-1, 2)]

    basis = spinless_fermion_basis_1d(L, Nf = 1) # single-body if Nf = 1
    static = [["+-", hop_intracell_pm], ["-+", hop_intracell_mp], ["+-", hop_intercell_pm], ["-+", hop_intercell_mp], ["n", stagg_pot]]
    checks = dict(check_pcon = True, check_symm = True, check_herm = True)
    H = hamiltonian(static, [], basis = basis, dtype = np.complex_, **checks)
    E, V = H.eigh()
    return E, V

def ssh_model_bloch(t, tp, Delta, kpoints = 50):
    kvec = np.linspace(0, 2*np.pi, kpoints)
    H = np.zeros((2, 2), dtype = np.complex_)
    np.fill_diagonal(H, [Delta, -Delta])
    evals = np.zeros((kpoints, H.shape[0]))
    evecs = np.zeros((kpoints, H.shape[0], H.shape[0]), dtype = np.complex_)
    for i, k in enumerate(kvec):
        H[0, 1] = (t + tp*np.exp(1j*k))
        H[1, 0] = np.conj(H[0, 1])
        E, V = f.eig_(H)
        evals[i, ] = E
        evecs[i, ] = V
    return kvec, evals, evecs

# E, V = ssh_model_real_space(t, tp, Delta, L, bc = bc)
# p.plot_localization1D(V[:, L // 2], 'ssh-localiz-{}-{}-{}'.format(t, tp, Delta))
# p.plot_energy_states(E, name = 'ssh-{}-{}-{}'.format(t, tp, Delta), mode = 'normal')

# kvec, evals, evecs = ssh_model_bloch(1, 2, 3, kpoints = 50)
# p.plot_bands1D(kvec, evals, name = 'ssh-bloch-{}-{}-{}'.format(t, tp, Delta))
