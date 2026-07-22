import numpy as np
import matplotlib.pyplot as plt
import wigner_funcs as wf

# -----------------------
# parameters
# -----------------------
hbar = 6.62607015e-34 / (2*np.pi)  # Planck's constant
t_1 = 1.0e-6   # time after first kick
t_2 = 150e-6 #150.0e-6   # time after second kick
m = 1.4e-19   # mass
omega_0 = 2 * np.pi * 50e3
sigma_x = np.sqrt(hbar/(2*m*omega_0))
sigma_p = np.sqrt(hbar*m*omega_0/2)


# grid
res_x = 1000
res_q = 10
n = 20
q_tot = 2e-23

###################
q = q_tot/n
dp = q/res_q
dp = round(dp, 30)
q = round(dp * res_q, 30)
q_tot = round(q*n, 30)
###################

print(dp)
print(q)
print(q_tot)

sigma_p_in_dp_units = int(sigma_p//dp+2)

p_index = np.arange(-4*sigma_p_in_dp_units, res_q*n+4*sigma_p_in_dp_units)

p = p_index * dp
x = np.linspace(-4*sigma_x, 4*sigma_x, res_x)
# p = np.arange(-4*sigma_p_in_q_units, 4*sigma_p_in_q_units+n+1/20, 1/10)*q
# p = np.linspace(-4*sigma_p_in_q_units, 4*sigma_p_in_q_units+n, res)*q
dp = p[1]-p[0]

X = x[:, np.newaxis]
P = p[np.newaxis, :]
phi0 = 0
phi2 = np.pi
########## Calculate Wigner function at each step ##########

W = wf.W0(X, P, sigma_x, sigma_p, hbar)

k = np.arange(0, n)
print(k)
phis = 2*np.pi*k/n
for phi in phis:
    W = wf .kick(W, X, P, dp, q, phi, hbar)

Z = W

W0 = wf.W0(X, P, sigma_x, sigma_p, hbar)
Z2 =  wf .kick(W0, X, P, dp, q_tot, phi0, hbar)


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
axes[0, 0].set_title("{} small kicks".format(n))
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
axes[0, 1].set_title("One large kick")
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