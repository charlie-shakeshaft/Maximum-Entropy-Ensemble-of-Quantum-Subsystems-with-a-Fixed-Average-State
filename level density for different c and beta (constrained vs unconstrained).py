"""
Created on Fri Jan 30 11:15:48 2026
@author: Charlie Shakeshaft
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import fsolve
from scipy.special import i1

def U_func(beta):
    if beta < 1e-6: return 1.0
    return i1(2*beta) / beta

def H_integral(u, beta):
    U = U_func(beta)
    # to optimise, separate real and imaginary integrands
    def integrand_real(E):
        denom = U * np.exp(beta * E) - u
        return (np.sqrt(4 - E**2) / denom).real
    def integrand_imag(E):
        denom = U * np.exp(beta * E) - u
        return (np.sqrt(4 - E**2) / denom).imag
    
    re, _ = quad(integrand_real, -2, 2, limit=100)
    im, _ = quad(integrand_imag, -2, 2, limit=100)
    return (re + 1j*im) / (2 * np.pi)

def compute_rho(lambdas, beta, c, eps=1e-3):
    rhos = []
    last_u = 0.5 - 0.5j 
    
    for lam in lambdas:
        w = 1.0 / (lam + 1j * eps)
        
        def objective(vars):
            u = vars[0] + 1j * vars[1]
            H = H_integral(u, beta)
            res = w - u / (1 + c * u * H)
            return [res.real, res.imag]
        
        sol = fsolve(objective, [last_u.real, last_u.imag], xtol=1e-8)
        u_sol = sol[0] + 1j * sol[1]
        last_u = u_sol 
        
        if abs(c) < 1e-9:
            G = u_sol**2 * H_integral(u_sol, beta)
        else:
            G = (u_sol - w) / c
            
        val = -np.imag(G) / np.pi
        rhos.append(max(0, val)) 
        
    return np.array(rhos)

def marchenko_pastur(x, c):
    if c == 0: return np.zeros_like(x)
    x_min = (1 - np.sqrt(c))**2
    x_max = (1 + np.sqrt(c))**2
    with np.errstate(divide='ignore', invalid='ignore'):
        res = np.sqrt(np.maximum(0, (x_max - x) * (x - x_min))) / (2 * np.pi * c * x)
    return np.nan_to_num(res)

# Parameters
beta_val = 0.5
c_val = 1.0 
lams = np.linspace(0.1, 5.0, 100) # Increased density for smoother plots

# Computation
print(f"Solving for beta={beta_val}, c={c_val}...")
rho_thermal = compute_rho(lams, beta_val, c_val)
rho_mp = marchenko_pastur(lams, c_val)

# Transformations
x_vals = np.sqrt(lams)
P_thermal = 2 * x_vals * rho_thermal
P_mp = 2 * x_vals * rho_mp

# --- Plotting ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

# Top Plot: Spectral Density rho(lambda)
ax1.plot(lams, rho_thermal, 'o-', markersize=4, label=fr'Constrained ($\beta={beta_val}$)', color='tab:red')
ax1.plot(lams, rho_mp, 'k--', label=fr'Marchenko-Pastur')
ax1.fill_between(lams, rho_thermal, alpha=0.1, color='tab:red')
ax1.set_xlabel(r"$\lambda$", fontsize=16)
ax1.set_ylabel(r"$\rho(\lambda)$", fontsize=16)
ax1.legend(loc='upper right', fontsize=14) # Specified location
ax1.grid(True, alpha=0.3)

# Bottom Plot: Transformed Distribution P(sqrt(lambda))
ax2.plot(x_vals, P_thermal, 'o-', markersize=4, label=fr'Constrained ($\beta={beta_val}$)', color='tab:red')
ax2.plot(x_vals, P_mp, 'k--', label='Marchenko-Pastur')
ax2.fill_between(x_vals, P_thermal, alpha=0.1, color='tab:red')
ax2.set_xlabel(r"$\sqrt{\lambda}$", fontsize=16)
ax2.set_ylabel(r"$P(\sqrt{\lambda})$", fontsize=16)
ax2.legend(loc='upper right', fontsize=14) # Added legend to the bottom plot
ax2.grid(True, alpha=0.3)

plt.tight_layout(pad=4.0)
plt.show()