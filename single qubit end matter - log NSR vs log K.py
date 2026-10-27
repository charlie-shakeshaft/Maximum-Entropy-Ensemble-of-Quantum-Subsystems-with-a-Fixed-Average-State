# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 11:59:18 2026

@author: j64461cs
"""

"""
Single-qubit subsystem: noise-to-signal ratio (NSR) of the energy Tr(H rho) versus K.
Python translation of the Mathematica notebook for Fig. 4 (log-log plot).

Requires: numpy, scipy, matplotlib
"""
import numpy as np
from scipy.special import ive
from scipy.optimize import brentq
import matplotlib as mpl
import matplotlib.pyplot as plt

# ----------------------------- parameters -----------------------------
K_MAX = 100
BETA = 1.0
DELTA = 1.0

SHOW_GRADIENT_PLOT = False          # in the notebook this plot is commented out of the GraphicsRow
SAVE_AS = "fig4_nsr_loglog"         # writes <name>.png and <name>.pdf

target = np.tanh(BETA * DELTA / 2)


# ------------------------------ functions ------------------------------
def bessel_ratio(k, m):
    """I_{k+1/2}(m/2) / I_{k-1/2}(m/2)."""
    x = m / 2
    return ive(k + 0.5, x) / ive(k - 0.5, x)


def solve_m(k):
    """Root m>0 of bessel_ratio(k, m) = tanh(beta*Delta/2). The ratio increases monotonically
    from 0 to 1, so the root is unique; bracket it using the large-K estimate
    x ~ 2 k r/(1-r^2) (x = m/2) to avoid underflow of ive at tiny arguments."""
    f = lambda m: bessel_ratio(k, m) - target
    m0 = 4 * k * target / (1 - target**2)
    lo, hi = 0.2 * m0, 5.0 * m0 + 10.0
    while f(lo) > 0:
        lo /= 2
    while f(hi) < 0:
        hi *= 2
    return brentq(f, lo, hi, xtol=1e-14, rtol=1e-14, maxiter=500)


def sqrt_var_Hs(k, m):
    """sqrt(kappa_2): the notebook's SqrtvarHs, term by term.
    ive replaces BesselI; the common factor exp(-m/2) cancels between numerator and denominator."""
    x = m / 2
    a = {n: ive(k + n, x) for n in (-2.5, -1.5, -0.5, 0.5, 1.5)}
    inner = (-m**2 * a[-1.5]**2
             + m**2 * a[-2.5] * a[-0.5]
             + 2 * (-4 + 8 * k + m**2) * a[-0.5]**2
             - 2 * m**2 * a[-1.5] * a[0.5]
             - m**2 * a[0.5]**2
             + m**2 * a[-0.5] * a[1.5])
    return np.sqrt(DELTA**2 * inner / (16 * m**2 * a[-0.5]**2))


def mean_Hs(k, m):
    """|kappa_1| = (Delta/2) I_{k+1/2}/I_{k-1/2}."""
    return DELTA * bessel_ratio(k, m) / 2


def kappa2_closed_form(k, m):
    r = bessel_ratio(k, m)
    return DELTA**2 / 4 * (1 - 4 * k / m * r - r**2)


# --------------------------- solve m(K), build data ---------------------------
Ks = np.arange(2, K_MAX + 1)
ms = np.array([solve_m(k) for k in Ks])

nsr = np.array([sqrt_var_Hs(k, m) / mean_Hs(k, m) for k, m in zip(Ks, ms)])

# cross-check against the closed form of the End Matter
k2_nb = np.array([sqrt_var_Hs(k, m) ** 2 for k, m in zip(Ks, ms)])
k2_cf = np.array([kappa2_closed_form(k, m) for k, m in zip(Ks, ms)])
print(f"max relative deviation, notebook kappa_2 vs closed form: {np.max(np.abs(k2_nb / k2_cf - 1)):.2e}")

# log-log data  {ln K, ln NSR}
logK, logNSR = np.log(Ks), np.log(nsr)

# ---------------- pointwise gradient d(ln NSR)/d(ln K) (central difference) ----------------
grad_x = logK[1:-1]
grad_y = (logNSR[2:] - logNSR[:-2]) / (logK[2:] - logK[:-2])

# ---------------- asymptotic slope: linear fit on the upper half of the K values ----------------
i0 = len(Ks) // 2 - 1                    # Mathematica: Floor[Length/2] ;; (1-based) -> K = 50 for kMax = 100
slope, intercept = np.polyfit(logK[i0:], logNSR[i0:], 1)

print("--- Gradient Analysis ---")
print(f"Linear Fit Slope (K >= {Ks[i0]}): {slope:.6f}")
print(f"Local Slope at K = {Ks[-2]}: {grad_y[-1]:.6f}")

# ----------------------------------- plots -----------------------------------
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 14,
    "xtick.direction": "in", "ytick.direction": "in",       # Mathematica "Frame" style
    "xtick.top": True, "ytick.right": True,
})

ncols = 2 if SHOW_GRADIENT_PLOT else 1
fig, axes = plt.subplots(1, ncols, figsize=(5.0 * ncols, 3.3), squeeze=False)

ax = axes[0, 0]
ax.plot(logK, logNSR, color="red", lw=1.5)
ax.set_xlabel(r"ln $K$")
ax.set_ylabel("ln NSR")

if SHOW_GRADIENT_PLOT:
    ax2 = axes[0, 1]
    ax2.plot(grad_x, grad_y, color="blue", lw=1.5)
    ax2.axhline(-0.5, color="black", ls="--", lw=1)
    ax2.set_ylim(-1.05, 0)
    ax2.set_xlabel(r"ln $K$")
    ax2.set_ylabel(r"d(ln NSR) / d(ln $K$)")

fig.tight_layout()
fig.savefig(SAVE_AS + ".png", dpi=300)
fig.savefig(SAVE_AS + ".pdf")
plt.show()