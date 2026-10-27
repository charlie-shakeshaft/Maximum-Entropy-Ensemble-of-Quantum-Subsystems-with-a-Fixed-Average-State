# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 17:08:51 2026

@author: j64461cs
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 10:22:56 2026

@author: j64461cs
"""

import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.interpolate import make_interp_spline
from scipy.optimize import brentq

# =====================================================================
# 1. HAMILTONIAN AND STATE DEFINITIONS
# =====================================================================
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
    """Generates a constrained random density matrix (FTCWE)."""
    A = (np.random.randn(d, k) + 1j * np.random.randn(d, k)) / np.sqrt(2)
    sqrtRho = sqrt_hermitian(rho0)
    X = A @ A.conj().T
    rhoTilde = sqrtRho @ X @ sqrtRho
    return rhoTilde / np.trace(rhoTilde)

def VonNeumannEntropy(rho):
    """Computes the Von Neumann entropy of a density matrix."""
    vals = np.real(la.eigvals(rho))
    vals = vals[vals > 1e-12]
    return -np.sum(vals * np.log(vals))

# =====================================================================
# 2. FREE PROBABILITY: FULL (NON-APPROXIMATED) INVERSE S-TRANSFORM
# =====================================================================
def free_prob_entropy(d, k, lam, ngrid=4000, x_lo=1e-8, a0=0.5, a_end=1e-3, nsteps=30):
    """
    Free-probability prediction for <S(rho)> of the FTCWE, WITHOUT the high-temperature
    approximation.  The eigenvalue density pi(x) of mu_{rho0} (x) mu_MP(c) (x = d*eigenvalue,
    c = d/k) is obtained by numerically inverting the S-transform, i.e. by solving the
    implicit equation (paper, lambda-H(u) relation)

        x = 1/u + c H(u),     H(u) = int dE sigma(E) / (U e^{beta E} - u)
                                   = (1/d) sum_j t_j / (1 - u t_j),

    with t_j = d*lam_j the rescaled eigenvalues of rho0 (i.e. sigma(E) is the exact spectral
    density of the subsystem Hamiltonian), choosing the root with Im u < 0, and then
        G(x) = u (1 + u H(u)) / (1 + c u H(u)),     pi(x) = -Im G(x + i0) / pi.
    Finally  S = ln d - int dx pi(x) x ln x   (the atom at x = 0 for c > 1 does not contribute).
    """
    if d == 1:
        return 0.0
    c = d / k
    t = d * lam[lam > 1e-300]                  # rescaled eigenvalues of rho0 (mean 1)
    w = 1.0 / d
    H  = lambda u: w * np.sum(t / (1 - u * t))
    Hp = lambda u: w * np.sum(t**2 / (1 - u * t)**2)

    # upper edge x_+ of the support: critical point of x(u) = 1/u + c H(u) on (0, 1/t_max)
    u_hi = (1.0 / t.max()) * (1 - 1e-13)
    u_c = brentq(lambda u: -1 / u**2 + c * Hp(u), 1e-12, u_hi, xtol=1e-15)
    x_hi = 1.05 * (1 / u_c + c * H(u_c))
    xs = np.exp(np.linspace(np.log(x_lo), np.log(x_hi), ngrid))   # log-spaced grid in x

    def newton(u, z):
        for _ in range(100):
            F = 1 / u + c * H(u) - z
            du = F / (-1 / u**2 + c * Hp(u))
            s = 1.0
            while abs(s * du) > 0.3 * abs(u) and s > 1e-4:
                s *= 0.5
            u = u - s * du
            if abs(F) < 1e-13 * max(1, abs(z)):
                break
        return u

    # stage 1: sweep downward in x along the ray z = x(1 + i a0), starting from u ~ 1/z
    u_ray = np.empty(ngrid, dtype=complex)
    u = None
    for i in range(ngrid - 1, -1, -1):
        z = xs[i] * (1 + 1j * a0)
        u = newton(1 / z if u is None else u, z)
        u_ray[i] = u

    # stage 2: for all x at once, continue analytically towards the real axis
    u = u_ray.copy()
    T = t[None, :]
    for j in range(1, nsteps + 1):
        z = xs * (1 + 1j * a0 * (a_end / a0)**(j / nsteps))
        for _ in range(15):
            D = 1 - u[:, None] * T
            F = 1 / u + c * w * np.sum(T / D, axis=1) - z
            Fp = -1 / u**2 + c * w * np.sum(T**2 / D**2, axis=1)
            du = F / Fp
            u = u - np.minimum(1, 0.3 * np.abs(u) / np.maximum(np.abs(du), 1e-300)) * du
            if np.max(np.abs(F) / np.maximum(1, np.abs(z))) < 1e-12:
                break

    # stage 3: polish exactly on the real axis, z = x + i0 (removes the smoothing of stage 2)
    for _ in range(40):
        D = 1 - u[:, None] * T
        F = 1 / u + c * w * np.sum(T / D, axis=1) - xs
        Fp = -1 / u**2 + c * w * np.sum(T**2 / D**2, axis=1)
        du = F / Fp
        u_new = u - np.minimum(1, 0.3 * np.abs(u) / np.maximum(np.abs(du), 1e-300)) * du
        u = np.where(np.isfinite(u_new), u_new, u)
        if np.max(np.abs(F) / np.maximum(1, xs)) < 1e-12:
            break

    # Cauchy transform -> density pi(x) = -Im G(x + i0)/pi
    Hv = w * np.sum(T / (1 - u[:, None] * T), axis=1)
    G = u * (1 + u * Hv) / (1 + c * u * Hv)
    pi_x = np.abs(G.imag) / np.pi

    # S = ln d - int dx pi(x) x ln x   (integrated in ln x: dx = x dlnx)
    lx = np.log(xs)
    trapz = getattr(np, 'trapezoid', None) or np.trapz   # numpy >= 2 renamed trapz
    integral_val = trapz(pi_x * xs**2 * lx, lx)
    return max(np.log(d) - integral_val, 0.0)

# ==================================================================in 33===
# 3. PARAMETERS AND EXPERIMENT RUN
# =====================================================================
nSpinsTotal = 10
Kd = 2**nSpinsTotal
J = 1.0
g = 1.0
betaValues = [0.4, 0.6, 1.0]
stateSamples = 200

subSpinsRange = np.arange(0, nSpinsTotal + 1)
divs = 2**subSpinsRange

# Analytical standard Page curve (beta -> 0)
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

# Data containers
multi_beta_data = []
thermal_entropy_data = []
multi_beta_fp_data = []  # Store Free Probability entropies

print("=== Simulating FTCWE & Calculating FP Curves ===")
for b in betaValues:
    b_x, b_y, therm_y, fp_y = [], [], [], []
    for d in divs:
        nSub = int(np.log2(d))
        k = Kd // d
        if d == 1:
            avgS = 0.0
            thermS = 0.0
            fpS = 0.0
        else:
            Hsub = TFIMHamiltonian(nSub, J, g)
            rho0Fixed = GibbsState(Hsub, b)
            thermS = VonNeumannEntropy(rho0Fixed)
            
            # (i) Monte Carlo Simulation
            entropies = [VonNeumannEntropy(WeightedInducedDensity(d, k, rho0Fixed)) 
                         for _ in range(stateSamples)]
            avgS = np.mean(entropies)
            
            # (ii) Free Probability: density from the full numerical inverse S-transform,
            #      using the exact spectrum of rho0 (no high-temperature approximation)
            fpS = free_prob_entropy(d, k, np.clip(la.eigvalsh(rho0Fixed), 0, None))
        
        b_x.append(nSub / nSpinsTotal)
        b_y.append(avgS)
        therm_y.append(thermS)
        fp_y.append(fpS)
        
    multi_beta_data.append((b_x, b_y))
    thermal_entropy_data.append((b_x, therm_y))
    multi_beta_fp_data.append((b_x, fp_y))

    s_beta = np.polyfit(subSpinsRange[1:], therm_y[1:], 1)[0]
    c_peak = 1.0 / (1.0 + s_beta / np.log(2))
    print(f"Beta = {b:0.1f} | s(beta) = {s_beta:.4f} nats/spin | Predicted c_peak = {c_peak:.4f}")
    print(f"          max |<S>_FTCWE - S_FP| over N_d = {np.max(np.abs(np.array(b_y) - np.array(fp_y))):.4f} nats")

# =====================================================================
# 4. PLOTTING
# =====================================================================

x_th = np.array(theory_x)
y_th = np.array(theory_y)
x_th_sm = np.linspace(x_th.min(), x_th.max(), 300)
spl_th = make_interp_spline(x_th, y_th, k=2)
y_th_sm = spl_th(x_th_sm)

# --- FIGURE 1: Empirical FTCWE curves (solid) with free probability overlaid (dotted) ---
fig1, ax1 = plt.subplots(figsize=(10, 6))

for i, b in enumerate(betaValues):
    x, y = multi_beta_data[i]
    _, y_fp = multi_beta_fp_data[i]
    
    x, y, y_fp = np.array(x), np.array(y), np.array(y_fp)
    x_smooth = np.linspace(x.min(), x.max(), 300)
    
    y_smooth = make_interp_spline(x, y, k=2)(x_smooth)
    y_fp_smooth = make_interp_spline(x, y_fp, k=2)(x_smooth)
    
    opacity = 0.3 + 0.7 * ((i + 1) / len(betaValues))**2
    color = (1.0, 0.0, 0.0, opacity)
    color_fp = (0.0, 0.0, 1.0, opacity) # Plotted in blue to distinguish from empirical
    
    ax1.plot(x_smooth, y_smooth, color=color, linewidth=2, label=rf'$\beta = {b}$')
    ax1.plot(x, y, ',',color=color, markersize=4)
    ax1.plot(x_smooth, y_fp_smooth, color=color_fp, linestyle='--', linewidth=2, label=rf'FP $\beta = {b}$')
    ax1.plot(x, y_fp,',', color=color_fp, markersize=4) # Squares for analytical points

ax1.plot(x_th_sm, y_th_sm, color='black', linestyle='--', linewidth=2, label='Page')
ax1.set_xlabel(r'$N_d / N$', fontsize=16)
ax1.set_ylabel(r'$\langle S(\rho) \rangle$', fontsize=16)
ax1.grid(True, linestyle='-', alpha=0.5)
ax1.legend(loc='upper right', fontsize=13,fancybox=True, edgecolor='0.8', facecolor='white', framealpha=1.0)
fig1.tight_layout()
ax1.set_ylim(0, 3)

plt.show()