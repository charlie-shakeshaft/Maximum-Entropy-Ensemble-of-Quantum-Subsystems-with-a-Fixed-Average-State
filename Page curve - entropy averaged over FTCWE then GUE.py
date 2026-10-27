# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 15:20:11 2026

@author: Charlie Shakeshaft
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

# random GUE Hamiltonian:
def random_gue(d):
    """
    Generates a d x d GUE random matrix with O(1) spectral support.

    With this normalization, the large-d spectral density approaches
    the Wigner semicircle law on approximately [-2, 2].
    """
    A = (
        np.random.normal(0, 1, (d, d))
        + 1j * np.random.normal(0, 1, (d, d))
    ) / np.sqrt(2)

    H = (A + A.conj().T) / 2.0

    # scaling s.t. spectral width remains O(1) as d increases
    H /= np.sqrt(d)

    return H

# constructs Gibbs state diag(p1, ... , pd) by diagonalising H
def gibbs_sqrt_eigendecomposition(H, beta):

    energies, vecs = np.linalg.eigh(H)

    # Numerical stabilisation of the Gibbs factors.
    shifted_energies = energies - np.min(energies)

    weights = np.exp(-beta * shifted_energies)
    weights /= np.sum(weights)

    sqrt_weights = np.sqrt(weights)

    return vecs, sqrt_weights

#Samples a state "rho = rho0^{1/2} AA^dagger rho0^{1/2} / trace" and computes its vN entropy.
def weighted_induced_entropy_from_gibbs(d, k, vecs, sqrt_weights):


    # At k = 1 the resulting state is rank 1 (pure).
    if k == 1:
        return 0.0

    # Complex Ginibre matrix A, d x k.
    A = (
        np.random.normal(0, 1, (d, k))
        + 1j * np.random.normal(0, 1, (d, k))
    ) / np.sqrt(2)

    # B = rho_0^(1/2) A in the eigenbasis of rho_0.
    B = sqrt_weights[:, None] * A

    # The nonzero eigenvalues of BB^\dagger are the eigenvalues of B^\dagger B. Use the smaller k x k matrix.
    gram = B.conj().T @ B

    trace = np.trace(gram).real

    if trace <= 0.0:
        return 0.0

    eigenvalues = np.linalg.eigvalsh(gram / trace)

    # Remove eigenvalues < 10^{-14}.
    eigenvalues = eigenvalues[eigenvalues > 1e-14]

    return -np.sum(eigenvalues * np.log(eigenvalues))

# for each GUE Hamiltonian, we: 
    #(1) compute the thermal rho0
    #(2) generate FTCWE density matrices from Ginibre matrices
    #(3) assign a vN entropy to each rho ~ FTCWE
    #(4) average over the FTCWE.
def average_entropy_for_hamiltonian(d, k, beta, H, state_samples):

    # Diagonalise H and obtain the Gibbs eigenbasis/weights once.
    vecs, sqrt_weights = gibbs_sqrt_eigendecomposition(H, beta)

    # For k = 1 every constrained state is pure.
    if k == 1:
        return 0.0

    entropies = np.empty(state_samples)

    for j in range(state_samples):
        entropies[j] = weighted_induced_entropy_from_gibbs(
            d,
            k,
            vecs,
            sqrt_weights
        )

    return np.mean(entropies)


# parameters:

Kd = 2**10

beta_values = [1.0, 1.5, 2.0]

# Number of independent GUE realisations:
ham_samples = 30

# Number of FTCWE samples for each H ~ GUE
state_samples = 15

divs = np.array([
    1,
    2,
    4,
    8,
    16,
    32,
    64,
    128,
    256,
    512,
    1024
])

# X-axis values:
# log2(d) / log2(Kd)
x_vals = np.log2(divs) / np.log2(Kd)


# Unconstrained analytics given by Page's Formula:

theory_y = []

for d in divs:

    k = Kd // d

    if d <= k:
        val = np.log(d) - d / (2 * k)
    else:
        val = np.log(k) - k / (2 * d)

    theory_y.append(val)

theory_y = np.array(theory_y)

# Calculate constrained data for each temperature:

multi_beta_y = []

for beta in beta_values:

    avg_entropies = []

    print(f"\n======================================")
    print(f"beta = {beta}")
    print(f"======================================")

    for d in divs:

        k = Kd // d

        print(
            f"d = {d:4d}, k = {k:4d}, "
            f"generating {ham_samples} GUE Hamiltonians..."
        )

        # Entropy averaged over the Ginibre ensemble for each H.
        entropy_per_hamiltonian = np.empty(ham_samples)

        for h in range(ham_samples):

            H = random_gue(d)

            entropy_per_hamiltonian[h] = average_entropy_for_hamiltonian(
                d=d,
                k=k,
                beta=beta,
                H=H,
                state_samples=state_samples
            )

        avg_entropy = np.mean(entropy_per_hamiltonian)

        avg_entropies.append(avg_entropy)

        print(
            f"    <S> = {avg_entropy:.6f}"
        )

    multi_beta_y.append(np.array(avg_entropies))


# PLOT

fig, ax = plt.subplots(figsize=(10, 6), dpi=120)

# Dense grid for quadratic interpolation.
x_smooth = np.linspace(
    x_vals.min(),
    x_vals.max(),
    300
)


n_betas = len(beta_values)

for i, (beta, y_vals) in enumerate(
    zip(beta_values, multi_beta_y),
    start=1
):

    opacity = 0.3 + 0.7 * (i / n_betas)**2

    spline = make_interp_spline(
        x_vals,
        y_vals,
        k=2
    )

    ax.plot(
        x_smooth,
        spline(x_smooth),
        color=(1.0, 0.0, 0.0, opacity),
        linewidth=2,
        label=rf'$\beta={beta}$'
    )



spline_theory = make_interp_spline(
    x_vals,
    theory_y,
    k=2
)

ax.plot(
    x_smooth,
    spline_theory(x_smooth),
    color='black',
    linestyle='--',
    linewidth=2,
    label='unconstrained'
)



ax.set_xlabel(
    r'$\log_2(d)/\log_2(Kd)$',
    fontsize=16
)

ax.set_ylabel(
    r'$\langle S(\rho) \rangle$',
    fontsize=16
)

ax.grid(
    True,
    linestyle=':',
    color='gray',
    alpha=0.5
)

ax.legend(
    loc='upper right',
    frameon=True,
    facecolor='white',
    edgecolor='0.8',
    fontsize=13
)

plt.tight_layout()
plt.show()