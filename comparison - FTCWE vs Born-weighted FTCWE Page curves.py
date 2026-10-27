# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 12:05:12 2026

@author: j64461cs
"""

import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

# Pauli matrices and Identity
sigmaX = np.array([[0, 1], [1, 0]], dtype=complex)
sigmaZ = np.array([[1, 0], [0, -1]], dtype=complex)
Id = np.eye(2, dtype=complex)

def TFIMHamiltonian(nSpins, J, g):
    """Constructs the Transverse Field Ising Model (TFIM) Hamiltonian."""
    if nSpins == 0:
        return np.array([[0.0]], dtype=complex)
    if nSpins == 1:
        return -g * sigmaX
    
    def mk_operator(ops):
        res = ops[0]
        for op in ops[1:]:
            res = np.kron(res, op)
        return res

    dim = 2**nSpins
    hZZ = np.zeros((dim, dim), dtype=complex)
    for i in range(nSpins - 1):
        ops = [Id] * i + [sigmaZ, sigmaZ] + [Id] * (nSpins - i - 2)
        hZZ += J * mk_operator(ops)
        
    hX = np.zeros((dim, dim), dtype=complex)
    for i in range(nSpins):
        ops = [Id] * i + [sigmaX] + [Id] * (nSpins - i - 1)
        hX += g * mk_operator(ops)
        
    return -(hZZ + hX)

def GibbsState(H, beta):
    """Calculates the Gibbs state rho = exp(-beta*H) / Tr(exp(-beta*H))."""
    evals, evecs = la.eigh(H)
    e_vals = np.exp(-beta * evals)
    rhoUn = evecs @ np.diag(e_vals) @ evecs.conj().T
    return rhoUn / np.trace(rhoUn)

def sqrt_hermitian(M):
    """Safely computes the square root of a positive semi-definite Hermitian matrix."""
    evals, evecs = la.eigh(M)
    evals = np.clip(evals, 0, None)
    return evecs @ np.diag(np.sqrt(evals)) @ evecs.conj().T

def WeightedInducedDensity(d, k, rho0):
    """Generates a constrained random density matrix and its unnormalized trace weight."""
    A = (np.random.randn(d, k) + 1j * np.random.randn(d, k)) / np.sqrt(2)
    sqrtRho = sqrt_hermitian(rho0)
    X = A @ A.conj().T
    rhoTilde = sqrtRho @ X @ sqrtRho
    
    # Extract the trace for Born rule weighting (p_X = Tr W_X)
    trace_W = np.real(np.trace(rhoTilde))
    
    return rhoTilde / trace_W, trace_W

def VonNeumannEntropy(rho):
    """Computes the Von Neumann entropy of a density matrix."""
    vals = np.real(la.eigvals(rho))
    vals = vals[vals > 1e-12]
    return -np.sum(vals * np.log(vals))

# PARAMETERS
nSpinsTotal = 8
Kd = 2**nSpinsTotal
J = 1.0
g = 1.0
betaValues = [0.4, 0.6, 1.0]
stateSamples = 100

# Subsystem dimensions
subSpinsRange = np.arange(0, nSpinsTotal + 1)
divs = 2**subSpinsRange

# Analytical Page curve (beta -> 0)
theory_x = []
theory_y = []
for d in divs:
    if d == 1:
        theory_x.append(0.0)
        theory_y.append(0.0)
    else:
        k = Kd // d
        if d <= k:
            val = np.log(d) - d / (2 * k)
        else:
            val = np.log(k) - k / (2 * d)
        theory_x.append(np.log2(d) / nSpinsTotal)
        theory_y.append(val)

# Constrained entropies for multiple betas
multi_beta_data_FTCWE = []
multi_beta_data_Born = []
thermal_entropy_data = []

print("=== Thermal Entropy Gradient & Peak Position ===")
for b in betaValues:
    b_x, b_y_FTCWE, b_y_Born, therm_y = [], [], [], []
    for d in divs:
        nSub = int(np.log2(d))
        k = Kd // d
        if d == 1:
            avgS_FTCWE = 0.0
            avgS_Born = 0.0
            thermS = 0.0
        else:
            Hsub = TFIMHamiltonian(nSub, J, g)
            rho0Fixed = GibbsState(Hsub, b)
            thermS = VonNeumannEntropy(rho0Fixed)
            
            entropies = []
            weights = []
            
            for _ in range(stateSamples):
                rho, w = WeightedInducedDensity(d, k, rho0Fixed)
                entropies.append(VonNeumannEntropy(rho))
                weights.append(w)
                
            # FTCWE average (uniform over Haar measure)
            avgS_FTCWE = np.mean(entropies)
            # Born rule average (weighted by p_X = Tr W_X)
            avgS_Born = np.average(entropies, weights=weights)
        
        b_x.append(nSub / nSpinsTotal)
        b_y_FTCWE.append(avgS_FTCWE)
        b_y_Born.append(avgS_Born)
        therm_y.append(thermS)
        
    multi_beta_data_FTCWE.append((b_x, b_y_FTCWE))
    multi_beta_data_Born.append((b_x, b_y_Born))
    thermal_entropy_data.append((b_x, therm_y))

    # (i) Compute gradient s(beta) = dS(rho_0)/dN_d (entropy per spin)
    s_beta = np.polyfit(subSpinsRange[1:], therm_y[1:], 1)[0]
    
    # (ii) Compute predicted peak position c_peak
    c_peak = 1.0 / (1.0 + s_beta / np.log(2))
    
    print(f"Beta = {b:0.1f} | s(beta) = {s_beta:.4f} nats/spin | Predicted c_peak = {c_peak:.4f}")

# PLOTTING FUNCTION
def plot_page_curves(data, title):
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, b in enumerate(betaValues):
        x, y = data[i]
        _, y_therm = thermal_entropy_data[i]
        
        x = np.array(x)
        y = np.array(y)
        y_therm = np.array(y_therm)
        
        x_smooth = np.linspace(x.min(), x.max(), 300)
        
        spline_page = make_interp_spline(x, y, k=2)
        y_smooth = spline_page(x_smooth)
        
        spline_therm = make_interp_spline(x, y_therm, k=2)
        y_therm_smooth = spline_therm(x_smooth)
        
        opacity = 0.3 + 0.7 * ((i + 1) / len(betaValues))**2
        color = (1.0, 0.0, 0.0, opacity)
        
        ax.plot(x_smooth, y_smooth, color=color, linewidth=2, label=rf'$\beta = {b}$')
        ax.plot(x, y, 'o', color=color, markersize=4)
        ax.plot(x_smooth, y_therm_smooth, color=color, linestyle=':', linewidth=2)

    x_th = np.array(theory_x)
    y_th = np.array(theory_y)
    x_th_sm = np.linspace(x_th.min(), x_th.max(), 300)
    spl_th = make_interp_spline(x_th, y_th, k=2)
    y_th_sm = spl_th(x_th_sm)

    ax.plot(x_th_sm, y_th_sm, color='black', linestyle='--', linewidth=2, label='Page')

    #ax.set_title(title, fontsize=18)
    ax.set_xlabel(r'$N_d / N$', fontsize=16)
    ax.set_ylabel(r'$\langle S(\rho) \rangle$', fontsize=16)
    ax.grid(True, linestyle='-', alpha=0.5)

    ax.legend(loc='upper right',
              fancybox=True, shadow=False, edgecolor='0.8',
              facecolor='white', framealpha=1.0, fontsize=13)

    plt.tight_layout()
    plt.ylim(0, 3)
    plt.show()

# Generate the FTCWE Page Curves
plot_page_curves(multi_beta_data_FTCWE, "Fixed-Trace Correlated Wishart Ensemble (FTCWE)")

# Generate the Born-Weighted Ensemble Page Curves
plot_page_curves(multi_beta_data_Born, r"Born-Rule Weighted Ensemble ($\mathcal{E}$)")


