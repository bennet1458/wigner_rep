import numpy as np
import matplotlib.pyplot as plt

# -----------------------
# parameters
# -----------------------
hbar = 1.0
eta = .5
t = 4.0   # time after kick
m = 1.0   # mass

q = 4   # larger q → clearer interference

# grid
x = np.linspace(-2.5, 17.5, 400)
p = np.linspace(-2.5, 12.5, 400)
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


# Function to integrate W over p at each x
def integrate_over_p(W_2d):
    """Integrate 2D Wigner function over p axis (sum over p for each x)"""
    dp = p[1] - p[0]  # spacing in p
    return np.sum(W_2d, axis=0) * dp


# -----------------------
# exact transformed Wigner
# -----------------------
W_k1= (
    eta**2 * W0(X, P - q)
    + eta*(1 - eta) * 2 * np.cos(q/hbar*X) * W0(X, P - q/2)
    + (1 - eta)**2 * W0(X, P)
)

W_t1_k1 = (
    eta**2 * W0(X-t*P/m, P - q)
    + eta*(1 - eta) * 2 * np.cos(q/hbar*(X-t*P/m)) * W0(X-t*P/m, P - q/2)
    + (1 - eta)**2 * W0(X-t*P/m, P)
)

W_k2_t1_k1 = ( eta**4 * W0(X - (P - q)*t/m, P - 2*q)

    + eta**3 * (1 - eta) * (
        2 * np.cos(q/hbar * (X - (P - q)*t/m)) * W0(X - (P - q)*t/m, P - 3*q/2)
        + 2 * np.cos(q*X/hbar) * W0(X - (P - q/2)*t/m, P - 3*q/2)
    )

    +
    eta**2 * (1 - eta)**2 * (
        W0(X - (P - q)*t/m, P - q)
        + 4 * np.cos(q*X/hbar) * np.cos(q/hbar * (X - (P - q/2)*t/m)) * W0(X - (P - q/2)*t/m, P - q)
        + W0(X - P*t/m, P - q)
    )

    + eta * (1 - eta)**3 * (
        2 * np.cos(q*X/hbar) * W0(X - (P - q/2)*t/m, P - q/2)
        + 2 * np.cos(q/hbar * (X - P*t/m)) * W0(X - P*t/m, P - q/2)
    )

    + (1 - eta)**4 * W0(X - P*t/m, P)
    )

W_t2_k2_t1_k1 = ( eta**4 * W0(X - (2*P - q)*t/m, P - 2*q)

    + eta**3 * (1 - eta) * (
        2 * np.cos(q/hbar * (X - (2*P - q)*t/m)) 
        * W0(X - (2*P - q)*t/m, P - 3*q/2)

        + 2 * np.cos(q*(X - P*t/m)/hbar) 
        * W0(X - (2*P - q/2)*t/m, P - 3*q/2)
    )

    + eta**2 * (1 - eta)**2 * (
        W0(X - (2*P - q)*t/m, P - q)

        + 4 * np.cos(q*(X - P*t/m)/hbar) 
        * np.cos(q/hbar * (X - (2*P - q/2)*t/m)) 
        * W0(X - (2*P - q/2)*t/m, P - q)

        + W0(X - (2*P)*t/m, P - q)
    )

    + eta * (1 - eta)**3 * (
        2 * np.cos(q*(X - P*t/m)/hbar) 
        * W0(X - (2*P - q/2)*t/m, P - q/2)

        + 2 * np.cos(q/hbar * (X - (2*P)*t/m)) 
        * W0(X - (2*P)*t/m, P - q/2)
    )

    + (1 - eta)**4 * W0(X - (2*P)*t/m, P)
)

# -----------------------
# Compute x-marginals by integrating over p
# -----------------------
W_k1_marginal = integrate_over_p(W_k1)
W_t1_k1_marginal = integrate_over_p(W_t1_k1)
W_k2_t1_k1_marginal = integrate_over_p(W_k2_t1_k1)
W_t2_k2_t1_k1_marginal = integrate_over_p(W_t2_k2_t1_k1)

# -----------------------
# plot subplots
# -----------------------
fig, axes = plt.subplots(4, 2, figsize=(6, 10))


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
axes[0, 0].set_title("First kick - W(x,p)")
plt.colorbar(im1, ax=axes[0, 0], label="W(x,p)")

# Subplot 2: x-marginal (integrated over p)
axes[0, 1].plot(W_k1_marginal, x, linewidth=2)
axes[0, 1].set_xlabel("Marginal")
axes[0, 1].set_ylabel("x")
axes[0, 1].set_title("First kick - Marginal(x)")
axes[0, 1].grid(True, alpha=0.3)

# Subplot 3: Wigner function after time evolution
im2 = axes[1, 0].imshow(
    W_t1_k1.T,
    extent=[p[0], p[-1], x[0], x[-1]],
    aspect='auto',
    origin='lower',
    cmap='RdBu_r'
)
axes[1, 0].set_xlabel("p")
axes[1, 0].set_ylabel("x")
axes[1, 0].set_title("First time evolution - W(x,p)")
plt.colorbar(im2, ax=axes[1, 0], label="W(x,p)")

# Subplot 4: x-marginal (integrated over p)
axes[1, 1].plot(W_t1_k1_marginal, x, linewidth=2)
axes[1, 1].set_xlabel("Marginal")
axes[1, 1].set_ylabel("x")
axes[1, 1].set_title("First time evolution - Marginal(x)")
axes[1, 1].grid(True, alpha=0.3)

# Subplot 5: Wigner function after second kick
im3 = axes[2, 0].imshow(
    W_k2_t1_k1.T,
    extent=[p[0], p[-1], x[0], x[-1]],
    aspect='auto',
    origin='lower',
    cmap='RdBu_r'
)
axes[2, 0].set_xlabel("p")
axes[2, 0].set_ylabel("x")
axes[2, 0].set_title("Second kick - W(x,p)")
plt.colorbar(im3, ax=axes[2, 0], label="W(x,p)")

# Subplot 6: x-marginal (integrated over p)
axes[2, 1].plot(W_k2_t1_k1_marginal, x, linewidth=2)
axes[2, 1].set_xlabel("Marginal")
axes[2, 1].set_ylabel("x")
axes[2, 1].set_title("Second kick - Marginal(x)")
axes[2, 1].grid(True, alpha=0.3)

# Subplot 7: Wigner function after second kick and time evolution
im4 = axes[3, 0].imshow(
    W_t2_k2_t1_k1.T,
    extent=[p[0], p[-1], x[0], x[-1]],
    aspect='auto',
    origin='lower',
    cmap='RdBu_r'
)
axes[3, 0].set_xlabel("p")
axes[3, 0].set_ylabel("x")
axes[3, 0].set_title("Second time evolution - W(x,p)")
plt.colorbar(im4, ax=axes[3, 0], label="W(x,p)")

# Subplot 8: x-marginal (integrated over p)
axes[3, 1].plot(W_t2_k2_t1_k1_marginal, x, linewidth=2)
axes[3, 1].set_xlabel("Marginal")
axes[3, 1].set_ylabel("x")
axes[3, 1].set_title("Second time evolution - Marginal(x)")
axes[3, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
