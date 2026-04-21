import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

# -----------------------
# parameters
# -----------------------
hbar = 1
eta = .5
t = 1.0   # time after kick
m = 1.0   # mass
q = 9   # larger q → clearer interference

M = 1.0
Omega = 1.0
m_number = 3

# grid
x = np.linspace(-2.5*q/5, 17.5*q/5, 400)
p = np.linspace(-2.5*q/5, 12.5*q/5, 400)
X, P = np.meshgrid(x, p)

# -----------------------
# initial Wigner function
# -----------------------


kappa = np.sqrt(M*Omega/hbar)

def W(x, p):
    y = (p/(hbar*kappa))**2 + (kappa*x)**2
    norm = np.sqrt(sp.special.factorial(m_number) / sp.special.gamma(m_number + 1))
    return (-1)**m_number/(np.pi * hbar) * np.exp(-y) * norm * sp.special.eval_genlaguerre(m_number, 0, 2*y)


# -----------------------
# exact transformed Wigner function
# -----------------------
W_k1= (
    eta**2 * W(X, P - q)
    + eta*(1 - eta) * 2 * np.cos(q/hbar*X) * W(X, P - q/2)
    + (1 - eta)**2 * W(X, P)
)

W_t1_k1 = (
    eta**2 * W(X-t*P/m, P - q)
    + eta*(1 - eta) * 2 * np.cos(q/hbar*(X-t*P/m)) * W(X-t*P/m, P - q/2)
    + (1 - eta)**2 * W(X-t*P/m, P)
)

W_k2_t1_k1 = ( eta**4 * W(X - (P - q)*t/m, P - 2*q)

    + eta**3 * (1 - eta) * (
        2 * np.cos(q/hbar * (X - (P - q)*t/m)) * W(X - (P - q)*t/m, P - 3*q/2)
        + 2 * np.cos(q*X/hbar) * W(X - (P - q/2)*t/m, P - 3*q/2)
    )

    +
    eta**2 * (1 - eta)**2 * (
        W(X - (P - q)*t/m, P - q)
        + 4 * np.cos(q*X/hbar) * np.cos(q/hbar * (X - (P - q/2)*t/m)) * W(X - (P - q/2)*t/m, P - q)
        + W(X - P*t/m, P - q)
    )

    + eta * (1 - eta)**3 * (
        2 * np.cos(q*X/hbar) * W(X - (P - q/2)*t/m, P - q/2)
        + 2 * np.cos(q/hbar * (X - P*t/m)) * W(X - P*t/m, P - q/2)
    )

    + (1 - eta)**4 * W(X - P*t/m, P)
    )

W_t2_k2_t1_k1 = ( eta**4 * W(X - (2*P - q)*t/m, P - 2*q)

    + eta**3 * (1 - eta) * (
        2 * np.cos(q/hbar * (X - (2*P - q)*t/m)) 
        * W(X - (2*P - q)*t/m, P - 3*q/2)

        + 2 * np.cos(q*(X - P*t/m)/hbar) 
        * W(X - (2*P - q/2)*t/m, P - 3*q/2)
    )

    + eta**2 * (1 - eta)**2 * (
        W(X - (2*P - q)*t/m, P - q)

        + 4 * np.cos(q*(X - P*t/m)/hbar) 
        * np.cos(q/hbar * (X - (2*P - q/2)*t/m)) 
        * W(X - (2*P - q/2)*t/m, P - q)

        + W(X - (2*P)*t/m, P - q)
    )

    + eta * (1 - eta)**3 * (
        2 * np.cos(q*(X - P*t/m)/hbar) 
        * W(X - (2*P - q/2)*t/m, P - q/2)

        + 2 * np.cos(q/hbar * (X - (2*P)*t/m)) 
        * W(X - (2*P)*t/m, P - q/2)
    )

    + (1 - eta)**4 * W(X - (2*P)*t/m, P)
)

# -----------------------
# plot subplots
# -----------------------
fig, axes = plt.subplots(2, 2, figsize=(8, 8))


# Subplot 1: Transformed Wigner function
im1 = axes[0, 0].imshow(
    W_k1.T,
    extent=[p[0], p[-1], x[0], x[-1]],
    aspect='auto',
    origin='lower',
    cmap='RdBu_r'
)
axes[0, 0].set_xlabel("p")
axes[0, 0].set_ylabel("x")
axes[0, 0].set_title("First kick")
plt.colorbar(im1, ax=axes[0, 0], label="W(x,p)")

# Subplot 2: Wigner function after time evolution
im2 = axes[0, 1].imshow(
    W_t1_k1.T,
    extent=[p[0], p[-1], x[0], x[-1]],
    aspect='auto',
    origin='lower',
    cmap='RdBu_r'
)
axes[0, 1].set_xlabel("p")
axes[0, 1].set_ylabel("x")
axes[0, 1].set_title("First time evolution")
plt.colorbar(im2, ax=axes[0, 1], label="W(x,p)")

# Subplot 3: Wigner function after second kick
im3 = axes[1, 0].imshow(
    W_k2_t1_k1.T,
    extent=[p[0], p[-1], x[0], x[-1]],
    aspect='auto',
    origin='lower',
    cmap='RdBu_r'
)
axes[1, 0].set_xlabel("p")
axes[1, 0].set_ylabel("x")
axes[1, 0].set_title("Second kick")
plt.colorbar(im3, ax=axes[1, 0], label="W(x,p)")


# Subplot 4: Wigner function after second kick and time evolution
im4 = axes[1, 1].imshow(
    W_t2_k2_t1_k1.T,
    extent=[p[0], p[-1], x[0], x[-1]],
    aspect='auto',
    origin='lower',
    cmap='RdBu_r'
)
axes[1, 1].set_xlabel("p")
axes[1, 1].set_ylabel("x")
axes[1, 1].set_title("Second time evolution")
plt.colorbar(im4, ax=axes[1, 1], label="W(x,p)")

plt.tight_layout()
plt.show()
