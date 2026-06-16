import numpy as np
import matplotlib.pyplot as plt
import wigner_functions as wf

# -----------------------
# parameters
# -----------------------
hbar = 1.0
eta = .5
t_1 = 1   # time after kick
m = 1.0   # mass
Lambda = 0.2
sigma_x = 1
sigma_p = hbar/(2*sigma_x)

q = 10   # larger q → clearer interference

# grid
x = np.linspace(-2.5*sigma_x, 2.5*sigma_x+q*t_1/m, 400)
p = np.linspace(-2.5*sigma_p, 2.5*sigma_p+q, 400)
X = x[:, np.newaxis]
P = p[np.newaxis, :]

########## Calculate Wigner function at each step ##########

W = wf.W0(sigma_x, sigma_p, hbar)

W = wf.kick(W, q, eta, hbar)

W = wf.time_evolution(W, t_1, m)


W_dc = wf.decoherence(W, x, p, t_1, Lambda, m, hbar)

Z = W(X, P)
Z_dc = W_dc(X, P)



# -----------------------
# plot subplots
# -----------------------
fig, axes = plt.subplots(1, 2, figsize=(8, 4))


# Subplot 1: Transformed Wigner function
im1 = axes[0].imshow(
    Z.T,
    extent=[x[0], x[-1], p[0], p[-1]],
    aspect='auto',
    origin='lower',
    cmap='RdBu_r'
)
axes[0].set_xlabel("x")
axes[0].set_ylabel("p")
axes[0].set_title("No decoherence")
plt.colorbar(im1, ax=axes[0], label="W(x,p)")

# Subplot 2: Wigner function after time evolution
im2 = axes[1].imshow(
    Z_dc.T,
    extent=[x[0], x[-1], p[0], p[-1]],
    aspect='auto',
    origin='lower',
    cmap='RdBu_r'
)
axes[1].set_xlabel("x")
axes[1].set_ylabel("p")
axes[1].set_title("With decoherence ($\Lambda$={:.0e})".format(Lambda))
plt.colorbar(im2, ax=axes[1], label="W(x,p)")


plt.tight_layout()
plt.show()