import numpy as np
import matplotlib.pyplot as plt

n_steps = 40
N = n_steps * 2 + 1

ks = np.linspace(-np.pi, np.pi, N, endpoint=False)
psi_p = np.array([1, 0], dtype=complex)
psi_m = np.array([0, 1], dtype=complex)

psi0 = psi_m
psi_k = np.zeros((N, 2), dtype=complex)

# k-space evolution
for i, k in enumerate(ks):

    M_k = (1/np.sqrt(2)) * np.array([
        [np.exp(-1j * k), np.exp(-1j * k)],
        [1, -1]
    ])

    M_k_n = np.linalg.matrix_power(M_k, n_steps)
    psi_k[i] = M_k_n @ psi0

# inverse FFT
psi_x = np.fft.ifft(psi_k, axis=0)
psi_x = np.fft.fftshift(psi_x, axes=0)
psi_x_p = psi_x @ psi_p
psi_x_m = psi_x @ psi_m

#######################################################
# -----------------------
# parameters
# -----------------------
hbar = 1.0

q = 4   # larger q → clearer interference

# grid
x = np.linspace(-2.5, 2.5, 100)
p = np.linspace(-2.5, n_steps*q+2.5, n_steps*20+100)
X, P = np.meshgrid(x, p)

# -----------------------
# initial Wigner function
# -----------------------
sigma_x = 1
sigma_p = hbar/(2*sigma_x)


def W0(x, p):
    return (1/(np.pi*hbar)) * np.exp(
        -x**2/(2*sigma_x**2)
        -p**2/(2*sigma_p**2)
    )


W_qrw = np.zeros_like(X, dtype=complex)
for n in range(n_steps):
    for n_ in range(n_steps):
        W_qrw += W0(X, P - q*n/2 - q*n_/2) * psi_x_p[n] * psi_x_p[n_].conj() * np.exp(1j * q * (n - n_) * X / hbar)


W_qrw = np.real(W_qrw)

# Compute x-marginal by integrating over p
dp = p[1] - p[0]
x_marginal = np.sum(W_qrw, axis=0) * dp

# Compute p-marginal by integrating over x
dx = x[1] - x[0]
p_marginal = np.sum(W_qrw, axis=1) * dx

# Create subplots
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(16, 4))

# Plot Wigner function
im = ax1.imshow(
    W_qrw.T,
    extent=[p[0], p[-1], x[0], x[-1]],
    aspect='auto',
    origin='lower',
    cmap='RdBu_r'
)
ax1.set_xlabel("p")
ax1.set_ylabel("x")
ax1.set_title("Wigner function")
plt.colorbar(im, ax=ax1, label="W(x,p)")

# Plot x-marginal
ax2.plot(x, x_marginal)
ax2.set_xlabel("x")
ax2.set_ylabel("ρ(x)")
ax2.set_title("Position marginal distribution")
ax2.grid()

# Plot p-marginal
ax3.plot(p, p_marginal)
ax3.set_xlabel("p")
ax3.set_ylabel("ρ(p)")
ax3.set_title("Momentum marginal distribution")
ax3.grid()

# Plot psi_x_p
ax4.plot(np.abs(psi_x_p[N//2:N//2+n_steps])**2)
ax4.set_xlabel("x")
ax4.set_ylabel("|ψ(x)|²")
ax4.set_title("Probability amplitude squared")
ax4.grid()

plt.tight_layout()
plt.show()


