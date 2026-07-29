import numpy as np
import matplotlib.pyplot as plt
import wigner_funcs as wf

# -----------------------
# parameters
# -----------------------
hbar = 6.62607015e-34 / (2*np.pi)  # Planck's constant
eta = .5
t_1 = 1.0e-6   # time after first kick
t_2 = 150e-6 #150.0e-6   # time after second kick
m = 1.4e-19   # mass
omega_0 = 2 * np.pi * 50e3
sigma_x = np.sqrt(hbar/(2*m*omega_0))
sigma_p = np.sqrt(hbar*m*omega_0/2)


# grid
res = 1000
n = 1
q = 2/n*1e-23#10e-23   # larger q → clearer interference
x = np.linspace(-4*sigma_x, 4*sigma_x, res)
p = np.linspace(-4*sigma_p, 4*sigma_p+n*q, res)
X = x[:, np.newaxis]
P = p[np.newaxis, :]
phi = np.pi
t_pi = 2*np.pi*m*hbar/q**2
########## Calculate Wigner function at each step ##########

W = wf.W0(X, P, sigma_x, sigma_p, hbar)

W_plus = wf.kick(W, X, P, q, phi, hbar)
W_minus = wf.kick(W, X, P, q, -phi, hbar)
W_ev = wf.time_evolution(W_minus, X, P, t_pi, m)
Z = W_minus 
Z2 = W_ev


# -----------------------
# plot subplots
# -----------------------
fig, axes = plt.subplots(2, 2, figsize=(8, 8))


# Subplot 1: Transformed Wigner function
im1 = axes[0, 0].imshow(
    np.real(Z).T,
    extent=[x[0], x[-1], p[0], p[-1]],
    aspect='auto',
    origin='lower',
    cmap='RdBu_r'
)
axes[0, 0].set_xlabel("x")
axes[0, 0].set_ylabel("p")
plt.colorbar(im1, ax=axes[0, 0], label="W(x,p)")

# Subplot 2: Wigner function after time evolution
im2 = axes[0, 1].imshow(
    np.real(Z2).T,
    extent=[x[0], x[-1], p[0], p[-1]],
    aspect='auto',
    origin='lower',
    cmap='RdBu_r'
)
axes[0, 1].set_xlabel("x")
axes[0, 1].set_ylabel("p")
plt.colorbar(im2, ax=axes[0, 1], label="W(x,p)")



#################### marginal plot ####################
# Function to integrate W over p at each x


# Subplot 3: Wigner function after second kick
im3 = axes[1, 0].plot(x, wf.x_marginal(Z, p), linewidth=2)
axes[1, 0].set_xlabel("x")
axes[1, 0].set_ylabel("Marginal")


# Subplot 4: Wigner function after second kick and time evolution
im4 = axes[1, 1].plot(x, wf.x_marginal(Z2, p), linewidth=2)
axes[1, 1].set_xlabel("x")
axes[1, 1].set_ylabel("Marginal")

plt.tight_layout()
plt.show()