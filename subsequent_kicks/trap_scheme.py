import numpy as np
import matplotlib.pyplot as plt
import wigner_functions as wfs

# -----------------------
# parameters
# -----------------------
hbar = 6.62607015e-34 / (2*np.pi)  # Planck's constant
t_1 = 1.0e-6   # time after first kick
t_2 = 150e-6 #150.0e-6   # time after second kick
m = 1.4e-19   # mass
omega_0 = 2 * np.pi * 50e3
nbar = 0
sigma_x = np.sqrt(hbar*(2*nbar+1)/(2*m*omega_0))
sigma_p = np.sqrt(hbar*(2*nbar+1)*m*omega_0/2)


# grid
res_q = 2000
res_x = 1000
q = 10e-23
t_quater =np.pi/(2*omega_0)
# x_center = 1/3e-9
###################
t_pi = 2*np.pi*m*hbar/q**2
# p = np.linspace(-4*sigma_p-q_tot, 4*sigma_p+q_tot, res_q)
p = np.linspace(-3*sigma_p, 3*sigma_p, res_q)
# x = np.linspace(-4*sigma_x, 4*sigma_x, res_x)
x = np.linspace(-4*sigma_x-2*sigma_p*t_2/m, 4*sigma_x+4*max(p)*t_quater/m+3*sigma_p*t_2/m, res_x)

X = x[:, np.newaxis]
P = p[np.newaxis, :]
phi0 = 0
phi2 = np.pi
########## Calculate Wigner function at each step ##########

W0 = wfs.W0(sigma_x, sigma_p, hbar)
W = wfs.kick(W0, q, phi0, hbar)
W = wfs.harmonic_evolution(W, t_quater, m, omega_0)
# W = wfs.kick(W, q, phi0, hbar)
# W = wfs.harmonic_evolution(W, t_quater/2, m, omega_0)
Z = W(X, P)
W = wfs.time_evolution(W, t_2, m)
Z2 = W(X, P)

# W0 = wfs.W0(sigma_x, sigma_p, hbar)
# W2 = wfs.kick(W0, q, phi0, hbar)
# Z2 = W2(X,P)

# -----------------------
# plot subplots
# -----------------------
fig, axes = plt.subplots(2, 2, figsize=(8, 8))


# Subplot 1: Transformed Wigner function
im1 = axes[0, 0].imshow(
   Z.T,
    extent=[x[0], x[-1], p[0], p[-1]],
    aspect='auto',
    origin='lower',
    cmap='RdBu_r'
)
axes[0, 0].set_xlabel("x")
axes[0, 0].set_ylabel("p")
axes[0, 0].set_title("After trap is turned off")
plt.colorbar(im1, ax=axes[0, 0], label="W(x,p)")

# Subplot 2: Wigner function after time evolution
im2 = axes[0, 1].imshow(
    Z2.T,
    extent=[x[0], x[-1], p[0], p[-1]],
    aspect='auto',
    origin='lower',
    cmap='RdBu_r'
)
axes[0, 1].set_xlabel("x")
axes[0, 1].set_ylabel("p")
axes[0, 1].set_title("After {:.0f} µs of free evolution".format(t_2*1e6))
plt.colorbar(im2, ax=axes[0, 1], label="W(x,p)")



#################### marginal plot ####################
# Function to integrate W over p at each x


# Subplot 3: Wigner function after second kick
im3 = axes[1, 0].plot(x, wfs.x_marginal(Z, p), linewidth=2)
axes[1, 0].set_xlabel("x")
axes[1, 0].set_ylabel("Marginal")


# Subplot 4: Wigner function after second kick and time evolution
im4 = axes[1, 1].plot(x, wfs.x_marginal(Z2, p), linewidth=2)
axes[1, 1].set_xlabel("x")
axes[1, 1].set_ylabel("Marginal")

plt.tight_layout()
plt.show()

dx = x[1] - x[0]
dp = p[1] - p[0]

print(np.sum(Z) * dx * dp)
print(np.sum(Z2) * dx * dp)
print(1/omega_0)