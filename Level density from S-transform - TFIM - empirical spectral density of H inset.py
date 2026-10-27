import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.optimize import fsolve

#TFIM at couplings J = h = 1:

def build_tfim_hamiltonian(N_d: int, J: float = 1.0, h: float = 1.0) -> np.ndarray:

    d = 2**N_d
    sx = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    sz = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    I = np.eye(2, dtype=np.complex128)
    
    H = np.zeros((d, d), dtype=np.complex128)
    
    # Nearest-neighbour interaction (open boundary conditions):
    for i in range(N_d - 1):
        ops = [I] * N_d
        ops[i], ops[i + 1] = sz, sz
        term = ops[0]
        for op in ops[1:]:
            term = np.kron(term, op)
        H -= J * term
        
    # Transverse field:
    for i in range(N_d):
        ops = [I] * N_d
        ops[i] = sx
        term = ops[0]
        for op in ops[1:]:
            term = np.kron(term, op)
        H -= h * term
        
    return H

def compute_pi(lambdas: np.ndarray, beta: float, c: float, sigma0: float = 1.0, eps: float = 1e-8) -> np.ndarray:
    """Here we compute the level density by numerically inverting
    the S-transform equation 1/u + c*H(u) = lambda - i*eps [see Supplemental Material].
    """
    # Evalueate H(u) integral using Gauss-Legendre quadrature:
    E_nodes, E_weights = np.polynomial.legendre.leggauss(300)
    E_nodes = E_nodes * 8.0 * sigma0  # Scale bounds to [-8*sigma0, 8*sigma0]
    E_weights = E_weights * 8.0 * sigma0
    
    sigma_E = (1.0 / (sigma0 * np.sqrt(2.0 * np.pi))) * np.exp(-E_nodes**2 / (2.0 * sigma0**2))
    U_beta = np.exp((beta**2 * sigma0**2) / 2.0)
    
    def H_func(u: complex) -> complex:
        denom = U_beta * np.exp(beta * E_nodes) - u
        return np.sum(E_weights * sigma_E / denom)

    # Sweep from high to low lambda using numerical continuation
    rev_lams = lambdas[::-1]
    rev_pis = []
    last_u = 0.5 - 0.5j
    
    for lam in rev_lams:
        L = lam - 1j * eps
        
        def residual(u_vec):
            u = u_vec[0] + 1j * u_vec[1]
            H_val = H_func(u)
            res = (1.0 / u) + c * H_val - L
            return [res.real, res.imag]
        
        sol = fsolve(residual, [last_u.real, last_u.imag], xtol=1e-9)
        u_sol = sol[0] + 1j * sol[1]
        last_u = u_sol
        
        H_val = H_func(u_sol)
        G = (u_sol + u_sol**2 * H_val) / (1.0 + c * u_sol * H_val)
        
        val = np.abs(np.imag(G)) / np.pi
        rev_pis.append(val)
        
    return np.array(rev_pis)[::-1]

def marchenko_pastur(x: np.ndarray, c: float) -> np.ndarray:
    """Function for MP law (dotted)"""
    if c == 0: 
        return np.zeros_like(x)
    x_min = (1 - np.sqrt(c))**2
    x_max = (1 + np.sqrt(c))**2
    with np.errstate(divide='ignore', invalid='ignore'):
        res = np.sqrt(np.maximum(0, (x_max - x) * (x - x_min))) / (2 * np.pi * c * x)
    return np.nan_to_num(res)

# PARAMETERS:
N_d = 12       # nummber of qubits for inset empirical density; d = 2**N_d
J_val = 1.0     # coupling
h_val = 1.0     # transverse field
beta_val = 0.5  # inverse temperature
c_val = 1.0     # dimension ratio d/K (set to 1 for quarter-circular MP density in \sqrt{x} )

print(f"Constructing and diagonalising N_d = {N_d} qubit TFIM Hamiltonian...")
H = build_tfim_hamiltonian(N_d, J=J_val, h=h_val)

# eigenvalues of H (for empirical spectral density):
E_evals = np.linalg.eigvalsh(H)

# variance sigma_0^2 = Tr(H^2)/d (with H traceless):
sigma0_sq = np.trace(H @ H).real / (2**N_d)
sigma0_val = np.sqrt(sigma0_sq)

# normalise H so that sigma_0 = 1:
H = H / sigma0_val

# rescale eigenvalues E_vals accordingly:
E_evals = E_evals / sigma0_val

# Fix the normalised variance sigma_0:
sigma0_sq = 1.0
sigma0_val = 1.0

# grid for eigenvalues of the constrained states
lams = np.linspace(1e-7, 6.0, 100000)

# computes both the constrained and MP level densities for the specified parameters:
print(f"Solving numerical S-transform inversion for beta={beta_val}, c={c_val}...")
pi_thermal = compute_pi(lams, beta_val, c_val, sigma0_val)
pi_mp = marchenko_pastur(lams, c_val)

# Mapping P(\lambda) -> P(\sqrt{lambda})
x_vals = np.sqrt(lams)
P_thermal = 2 * x_vals * pi_thermal
P_mp = 2 * x_vals * pi_mp

# --- EMPIRICAL CWE ---
print("Generating empirical correlated Wishart ensemble (10 samples)...")
d = 2**N_d
K = int(d / c_val)
M = 10  # Number of samples

# Calculate normalized eigenvalues of rho_0 such that Tr(rho_0)/d = 1
rho_evals = np.exp(-beta_val * E_evals)
rho_evals /= np.mean(rho_evals) 

all_W_evals = []

for _ in range(M):
    # Draw complex Ginibre matrix X
    X = (np.random.randn(d, K) + 1j * np.random.randn(d, K)) / np.sqrt(2)
    
    # Due to unitary invariance of X, we can work in the diagonal basis of rho_0
    # Construct Y = rho_0^{1/2} X
    Y = np.sqrt(rho_evals)[:, None] * X
    W = (Y @ Y.conj().T) / K
    
    all_W_evals.append(np.linalg.eigvalsh(W))

print("Aggregating empirical Wishart eigenvalues...")
W_evals = np.concatenate(all_W_evals)

# Main figure:
fig, ax2 = plt.subplots(1, 1, figsize=(10, 6))

# histogram the empirical CWE eigenvalues:
ax2.hist(np.sqrt(np.maximum(W_evals, 0)), bins='fd', density=True, 
         facecolor=[mcolors.to_rgba('tab:red', 0.1)], edgecolor='black', linewidth=0.5, 
         label=fr'Empirical CWE ($\beta={beta_val}$)')

ax2.plot(x_vals, P_thermal, '-', linewidth=2.5, label=fr'Free probability ($\beta={beta_val}$)', color='tab:red')
ax2.plot(x_vals, P_mp, 'k--', linewidth=2.5, label='Marchenko-Pastur')
ax2.fill_between(x_vals, P_thermal, alpha=0.0, color='tab:red')

ax2.set_xlabel(r"$\sqrt{x}$", fontsize=16)
ax2.set_ylabel(r"$P(\sqrt{x})$", fontsize=16)
ax2.legend(loc='upper right', fontsize=13)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(left=0)
ax2.set_ylim(bottom=0)

# Inset:
axins = ax2.inset_axes([0.072, 0.098, 0.44, 0.44])

# Empirical histogram of renormalised energy eigenvalues
axins.hist(E_evals, bins='fd', density=True, color='gray', alpha=0.7,
           edgecolor='black', linewidth=0.5, label=r'TFIM')

# Gaussian \sigma(E) with sigma0 = 1:
E_grid = np.linspace(np.min(E_evals), np.max(E_evals), 200)
gaussian_dos = (1.0 / np.sqrt(2 * np.pi)) * np.exp(-E_grid**2 / 2)
axins.plot(E_grid, gaussian_dos, 'k--', linewidth=2, label=r'Gaussian $\sigma(E)$')

axins.set_xlabel(r'$E$', fontsize=12)
axins.set_ylabel(r'$\sigma(E)$', fontsize=12)
axins.tick_params(axis='both', which='major', labelsize=9)
axins.legend(loc='upper right', fontsize=10, frameon=True)
axins.grid(True, alpha=0.2)

plt.tight_layout(pad=2.0)
plt.show()