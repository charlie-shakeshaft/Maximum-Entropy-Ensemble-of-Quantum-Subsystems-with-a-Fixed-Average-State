"""
@author: Charlie Shakeshaft
"""
import numpy as np
import matplotlib.pyplot as plt

def get_mixed_field_ising_hamiltonian(N, J=1.0, h=1.0, g=0.0):
    I = np.array([[1, 0], [0, 1]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)

    def make_single_site_op(op, site):
        res = 1
        for i in range(N):
            res = np.kron(res, op if i == site else I)
        return res

    def make_two_site_op(op1, op2, site1, site2):
        res = 1
        for i in range(N):
            if i == site1:
                res = np.kron(res, op1)
            elif i == site2:
                res = np.kron(res, op2)
            else:
                res = np.kron(res, I)
        return res

    d = 2**N
    H = np.zeros((d, d), dtype=complex)

    for i in range(N - 1):
        H -= J * make_two_site_op(Z, Z, i, i + 1)

    for i in range(N):
        H -= h * make_single_site_op(X, i)
        H -= g * make_single_site_op(Z, i)

    return H

def get_thermal_state(H, beta):
    evals, evecs = np.linalg.eigh(H)
    
    evals_shifted = evals - np.min(evals)
    exp_evals = np.exp(-beta * evals_shifted)
    
    Z = np.sum(exp_evals)
    rho_0_evals = exp_evals / Z
    rho_0_half_evals = np.sqrt(rho_0_evals)

    rho_0 = evecs @ np.diag(rho_0_evals) @ evecs.conj().T
    rho_0_half = evecs @ np.diag(rho_0_half_evals) @ evecs.conj().T

    return rho_0, rho_0_half

def generate_wishart_ensembles(N, K, beta, num_samples):
    d = 2**N
    H = get_mixed_field_ising_hamiltonian(N)
    _, rho_0_half = get_thermal_state(H, beta)

    all_evals = []
    all_weights = []

    for _ in range(num_samples):
        X = (np.random.randn(d, K) + 1j * np.random.randn(d, K)) / np.sqrt(2)
        W_X = (rho_0_half @ X @ X.conj().T @ rho_0_half) / K
        p_X = np.real(np.trace(W_X))
        rho_X = W_X / p_X
        evals = np.linalg.eigvalsh(rho_X)
        
        all_evals.append(evals)
        all_weights.append([p_X] * d)

    return np.array(all_evals).flatten(), np.array(all_weights).flatten()

def calculate_nsr_data(N_max, beta, num_samples=1000):
    Nd_vals = []
    nsr_vals = []
    
    for n in range(2, N_max + 1):
        d = 2**n
        K = d
        H = get_mixed_field_ising_hamiltonian(n)
        _, rho_0_half = get_thermal_state(H, beta)
        
        traces = []
        for _ in range(num_samples):
            X = (np.random.randn(d, K) + 1j * np.random.randn(d, K)) / np.sqrt(2)
            W_X = (rho_0_half @ X @ X.conj().T @ rho_0_half) / K
            traces.append(np.real(np.trace(W_X)))
            
        traces = np.array(traces)
        mean_tr = np.mean(traces)
        std_tr = np.std(traces)
        
        Nd_vals.append(d)
        nsr_vals.append(std_tr / mean_tr)
        
    return Nd_vals, nsr_vals

def plot_ensembles_scaled(evals, weights, N, K, beta):
    d = 2**N
    
    # Scale variable: x = \sqrt{d \lambda}
    scaled_x = np.sqrt(d * np.maximum(evals, 0.0))

    fig, ax = plt.subplots(figsize=(10, 6))

    # (1) Standard histograms
    # Weighted ensemble 
    ax.hist(scaled_x, bins=50, density=True, weights=weights, color='lightskyblue', alpha=0.4, 
            edgecolor='royalblue', linewidth=1.2, label=r'$\mathcal{E}=\{p_X, \rho_X\}$')
    
    # Uniform ensemble 
    ax.hist(scaled_x, bins=50, density=True, color='crimson', 
            histtype='step', linewidth=2.5, linestyle='--', 
            label=r'FTCWE')

    # Formatting main plot
    ax.set_xlabel(r'$\sqrt{x}$', fontsize=16)
    ax.set_ylabel(r'$P(\sqrt{x})$', fontsize=16)
    ax.legend(fontsize=13, loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.5)

    # (2) log-log Inset for NSR
    Nd_vals, nsr_vals = calculate_nsr_data(N_max=6, beta=beta, num_samples=10000)
    
    ax_ins = ax.inset_axes([0.095, 0.11, 0.45, 0.45])
    ax_ins.patch.set_facecolor('white')
    ax_ins.patch.set_alpha(0.85)
    
    # Render on a log-log scale
    ax_ins.loglog(Nd_vals, nsr_vals, marker='o', color='k', linewidth=1.5, markersize=5)
    ax_ins.set_xlabel(r'$N_d$', fontsize=13)
    ax_ins.set_ylabel(r'NSR', fontsize=13)
    ax_ins.grid(True, which='both', linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    N = 10          # Number of qubits
    K = 2**N       # Wishart degrees of freedom (environment size)
    beta = 0.1     # Inverse temperature
    samples = 500  # Samples for the main histograms

    evals, weights = generate_wishart_ensembles(N, K, beta, num_samples=samples)
    plot_ensembles_scaled(evals, weights, N, K, beta)

