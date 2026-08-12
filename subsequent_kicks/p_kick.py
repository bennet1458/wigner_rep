import numpy as np
import matplotlib.pyplot as plt
import wigner_functions as wfs

# -----------------------
# parameters
# -----------------------
hbar = 6.62607015e-34 / (2*np.pi)  # Planck's constant
t_1 = 1.0e-6   # time after first kick
t_2 = 3e-6 #150.0e-6   # time after second kick
m = 1.4e-19  # mass
omega_0 = 2 * np.pi * 50e3
nbar = 0
sigma_x = np.sqrt(hbar*(2*nbar+1)/(2*m*omega_0))
sigma_p = np.sqrt(hbar*(2*nbar+1)*m*omega_0/2)
d = 50e-9
q = 2e-23

# grid
res_q = 500
res_x = 1000


###################
p = np.linspace(-4*sigma_p, 4*sigma_p+q, res_q)
x = np.linspace(-4*sigma_x-2*sigma_p*t_2/m, 4*sigma_x+(2*sigma_p+q)*t_2/m, res_x)

X = x[:, np.newaxis]
P = p[np.newaxis, :]
########## Calculate Wigner function at each step ##########

W = wfs.W0(sigma_x, sigma_p, hbar)

W = wfs.kick(W, q, 0, hbar)


Z = W(X, P)

W2 = W = wfs.time_evolution(W, t_2, m)
Z2 = W2(X, P)

# W0 = wfs.W0(sigma_x, sigma_p, hbar)
# W2 = wfs.kick(W0, q, phi0, hbar)
# Z2 = W2(X,P)

# -----------------------
# plot subplots
# -----------------------
fig, axes = plt.subplots(2, 2, figsize=(8, 5), constrained_layout=True)


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
axes[0, 0].set_title('Momentum kick')
axes[0, 0].set_xticks([0])
axes[0, 0].set_yticks([0])



# Subplot 2: Wigner function after time evolution
im2 = axes[0, 1].imshow(
    Z2.T,
    extent=[x[0], x[-1], p[0], p[-1]],
    aspect='auto',
    origin='lower',
    cmap='RdBu_r'
)
axes[0, 1].set_xlabel("x ")
axes[0, 1].set_ylabel("p ")
axes[0, 1].set_title("After time evolution")
axes[0, 1].set_xticks([0])
axes[0, 1].set_yticks([0])

cbar = fig.colorbar(im2, ax=axes[0,1], label='$w(x,p)$')
cbar.set_ticks([])
cbar.ax.text(2.5, 0.98, '+', ha='center', va='top',
             transform=cbar.ax.transAxes)
cbar.ax.text(2.5, 0.02, '−', ha='center', va='bottom',
             transform=cbar.ax.transAxes)



#################### marginal plot ####################
# Function to integrate W over p at each x


# Subplot 3: Wigner function after second kick
im3 = axes[1, 0].plot(x, wfs.x_marginal(Z, p), linewidth=2)
axes[1, 0].set_xlabel("x")
axes[1, 0].set_ylabel("$w(x)$")
axes[1, 0].set_xticks([0])
axes[1, 0].set_yticks([0])


# Subplot 4: Wigner function after second kick and time evolution
im4 = axes[1, 1].plot(x, wfs.x_marginal(Z2, p), linewidth=2)
axes[1, 1].set_xlabel("x")
axes[1, 1].set_ylabel("$w(x)$")
axes[1, 1].set_xticks([0])
axes[1, 1].set_yticks([0])

# plt.tight_layout()
plt.show()