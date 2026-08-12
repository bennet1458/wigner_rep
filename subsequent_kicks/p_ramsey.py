import numpy as np
import matplotlib.pyplot as plt
import wigner_functions as wfs

# -----------------------
# parameters
# -----------------------
hbar = 6.62607015e-34 / (2*np.pi)  # Planck's constant
t_1 = 3.0e-6   # time after first kick
t_2 = 15.0e-6 #150.0e-6   # time after second kick
m = 1.4e-19  # mass
omega_0 = 2 * np.pi * 50e3
nbar = 0
sigma_x = np.sqrt(hbar*(2*nbar+1)/(2*m*omega_0))
sigma_p = np.sqrt(hbar*(2*nbar+1)*m*omega_0/2)

d = 50e-9
q = 2e-23

# grid
res_p = 500
res_x = 500

fig, axes = plt.subplots(2, 4, figsize=(12, 5), constrained_layout=True)
###################
x = np.linspace(-4*sigma_x, 4*sigma_x+q*t_1/m, res_x)
p = np.linspace(-4*sigma_p, 4*sigma_p+q, res_p)
X = x[:, np.newaxis]
P = p[np.newaxis, :]
W = wfs.W0(sigma_x, sigma_p, hbar)
W = wfs.kick(W, q, 0, hbar)
Z = W(X, P)

im1 = axes[0, 0].imshow(
   Z.T,
    extent=[x[0], x[-1], p[0], p[-1]],
    aspect='auto',
    origin='lower',
    cmap='RdBu_r'
)
axes[0, 0].set_xlabel("x")
axes[0, 0].set_ylabel("p")
axes[0, 0].set_title('First Kick')
axes[0, 0].set_xticks([0])
axes[0, 0].set_yticks([0])

im1a = axes[1, 0].plot(x, wfs.x_marginal(Z, p), linewidth=2)
axes[1, 0].set_xlabel("x")
axes[1, 0].set_ylabel("$w(x)$")
axes[1, 0].set_xticks([0])
axes[1, 0].set_yticks([0])

#################################################################################
x = np.linspace(-4*sigma_x, 4*sigma_x+q*t_1/m, res_x)
p = np.linspace(-4*sigma_p, 4*sigma_p+q, res_p)
X = x[:, np.newaxis]
P = p[np.newaxis, :]
W2 = wfs.W0(sigma_x, sigma_p, hbar)
W2 = wfs.kick(W2, q, 0, hbar)
W2 = wfs.time_evolution(W, t_1, m)
Z2 = W2(X, P)

im2 = axes[0, 1].imshow(
   Z2.T,
    extent=[x[0], x[-1], p[0], p[-1]],
    aspect='auto',
    origin='lower',
    cmap='RdBu_r'
)
axes[0, 1].set_xlabel("x")
axes[0, 1].set_ylabel("p")
axes[0, 1].set_title('First time evolution')
axes[0, 1].set_xticks([0])
axes[0, 1].set_yticks([0])

im2a = axes[1, 1].plot(x, wfs.x_marginal(Z2, p), linewidth=2)
axes[1, 1].set_xlabel("x")
axes[1, 1].set_ylabel("$w(x)$")
axes[1, 1].set_xticks([0])
axes[1, 1].set_yticks([0])

##################################################################################
x = np.linspace(-4*sigma_x, 4*sigma_x+q*t_1/m, res_x)
p = np.linspace(-4*sigma_p, 4*sigma_p+2*q, res_p)
X = x[:, np.newaxis]
P = p[np.newaxis, :]
W3 = wfs.W0(sigma_x, sigma_p, hbar)
W3 = wfs.kick(W3, q, 0, hbar)
W3 = wfs.time_evolution(W3, t_1, m)
W3 = wfs.kick(W3, q, 0, hbar)
Z3 = W3(X, P)

im3 = axes[0, 2].imshow(
   Z3.T,
    extent=[x[0], x[-1], p[0], p[-1]],
    aspect='auto',
    origin='lower',
    cmap='RdBu_r'
)
axes[0, 2].set_xlabel("x")
axes[0, 2].set_ylabel("p")
axes[0, 2].set_title('Second Kick')
axes[0, 2].set_xticks([0])
axes[0, 2].set_yticks([0])

im3a = axes[1, 2].plot(x, wfs.x_marginal(Z3, p), linewidth=2)
axes[1, 2].set_xlabel("x")
axes[1, 2].set_ylabel("$w(x)$")
axes[1, 2].set_xticks([0])
axes[1, 2].set_yticks([0])

##################################################################################

x = np.linspace(-4*sigma_x-2*sigma_p*t_2/m, 4*sigma_x+q*t_1/m+2*q*t_2/m+2*sigma_p*t_2/m, res_x)
p = np.linspace(-4*sigma_p, 4*sigma_p+2*q, res_p)
X = x[:, np.newaxis]
P = p[np.newaxis, :]
W4 = wfs.W0(sigma_x, sigma_p, hbar)
W4 = wfs.kick(W4, q, 0, hbar)
W4 = wfs.time_evolution(W4, t_1, m)
W4 = wfs.kick(W4, q, 0, hbar)
W4 = wfs.time_evolution(W4, t_2, m)
Z4 = W4(X, P)

im4 = axes[0, 3].imshow(
    Z4.T,
    extent=[x[0], x[-1], p[0], p[-1]],
    aspect='auto',
    origin='lower',
    cmap='RdBu_r'
)
axes[0, 3].set_xlabel("x ")
axes[0, 3].set_ylabel("p ")
axes[0, 3].set_title("Second time evolution")
axes[0, 3].set_xticks([0])
axes[0, 3].set_yticks([0])

cbar = fig.colorbar(im4, ax=axes[0,3], label='$w(x,p)$')
cbar.set_ticks([])
cbar.ax.text(2.5, 0.98, '+', ha='center', va='top',
             transform=cbar.ax.transAxes)
cbar.ax.text(2.5, 0.02, '−', ha='center', va='bottom',
             transform=cbar.ax.transAxes)

im4a = axes[1, 3].plot(x, wfs.x_marginal(Z4, p), linewidth=2)
axes[1, 3].set_xlabel("x")
axes[1, 3].set_ylabel("$w(x)$")
axes[1, 3].set_xticks([0])
axes[1, 3].set_yticks([0])

# plt.tight_layout()
plt.show()