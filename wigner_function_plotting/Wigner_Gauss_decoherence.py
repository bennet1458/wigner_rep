import numpy as np
import matplotlib.pyplot as plt

# -----------------------
# parameters
# -----------------------
hbar = 1.0
eta = .5
t = 1.0   # time after kick
m = 1.0   # mass
gamma = .2

q = 4   # larger q → clearer interference

# grid
x = np.linspace(-2.5*q/5, 17.5*q/5, 400)
p = np.linspace(-2.5*q/5, 12.5*q/5, 400)
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

def decoherence(W):
    # grid spacing
    dx = x[1] - x[0]
    dp = p[1] - p[0]

    Nx = len(x)
    Np = len(p)

    # Fourier frequencies (correct variables!)
    q = 2*np.pi * np.fft.fftfreq(Nx, d=dx)
    k = 2*np.pi * np.fft.fftfreq(Np, d=dp)

    Q, K = np.meshgrid(q, k, indexing='ij')

    # inverse transform
    return np.fft.ifft2(np.exp(-gamma * t/2 * (Q**2 + K**2)) * np.fft.fft2(W)).real

# -----------------------
# exact transformed Wigner
# -----------------------
W_k1= (
    eta**2 * W0(X, P - q)
    + eta*(1 - eta) * 2 * np.cos(q/hbar*X) * W0(X, P - q/2)
    + (1 - eta)**2 * W0(X, P)
)

W_dc1_k1 = decoherence(W_k1)


# -----------------------
# plot subplots
# -----------------------
fig, axes = plt.subplots(1, 2, figsize=(8, 4))


# Subplot 1: Transformed Wigner function
im1 = axes[0].imshow(
    W_k1.T,
    extent=[p[0], p[-1], x[0], x[-1]],
    aspect='auto',
    origin='lower',
    cmap='RdBu_r'
)
axes[0].set_xlabel("p")
axes[0].set_ylabel("x")
axes[0].set_title("First kick")
plt.colorbar(im1, ax=axes[0], label="W(x,p)")

# Subplot 2: Wigner function after time evolution
im2 = axes[1].imshow(
    W_dc1_k1.T,
    extent=[p[0], p[-1], x[0], x[-1]],
    aspect='auto',
    origin='lower',
    cmap='RdBu_r'
)
axes[1].set_xlabel("p")
axes[1].set_ylabel("x")
axes[1].set_title("First time evolution")
plt.colorbar(im2, ax=axes[1], label="W(x,p)")


plt.tight_layout()
plt.show()
