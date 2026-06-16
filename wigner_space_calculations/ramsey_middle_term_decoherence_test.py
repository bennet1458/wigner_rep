import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import wigner_functions as wf

# -----------------------
# parameters
# -----------------------
hbar = 6.62607015e-34 / (2*np.pi)  # Planck's constant
eta = .5
t_1 = 1.0e-6   # time after first kick
m = 1.4e-19   # mass
omega_0 = 2 * np.pi * 50e3
sigma_x = np.sqrt(hbar/(2*m*omega_0))
sigma_p = np.sqrt(hbar*m*omega_0/2)

### Decoherence parameters ###
diameter = 50e-9
volume = (4/3)*np.pi*(diameter/2)**3
T_i = 315
T_e = 300

w = 5
t_2 = 175e-6 #150.0e-6   # time after second kick
q = 10e-23#10e-23   # larger q → clearer interference
def gamma_bb(T):
    return 1.0e38 #1.91e31 * T**(8.38) * np.exp(-0.19 * np.log(T)**2 )
Lambda = 2 * volume * (gamma_bb(T_i) + gamma_bb(T_e))
print("Lambda:", Lambda)
Lambda = 1e21

# grid
res = 900#200 + int(q/20e-23 * 600)+100
print(res)
x = np.linspace(-4*sigma_x+(q-2*sigma_p)*(t_2)/m, 4*sigma_x+(q+2*sigma_p)*(t_1+t_2)/m, res)
p = np.linspace(-4*sigma_p+q, 4*sigma_p+q, res)
X = x[:, np.newaxis]
P = p[np.newaxis, :]

########## Calculate Wigner function at each step ##########

W = wf.W0(sigma_x, sigma_p, hbar)
W_dc = wf.W0(sigma_x, sigma_p, hbar)

W = wf.kick(W, q, eta, hbar)
W_dc = wf.kick(W_dc, q, eta, hbar)

W = wf.time_evolution(W, t_1, m)
W_dc = wf.time_evolution(W_dc, t_1, m)

W_dc = wf.decoherence(W_dc, x, p, t_1, Lambda, m, hbar)

W = wf.kick(W, q, eta, hbar)
W_dc = wf.kick(W_dc, q, eta, hbar)

W = wf.time_evolution(W, t_2, m)
W_dc = wf.time_evolution(W_dc, t_2, m)

W_dc = wf.decoherence(W_dc, x, p, t_2, Lambda, m, hbar)

Z = W(X, P)
Z_dc = W_dc(X, P)



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
axes[0, 0].set_title("No decoherence")
plt.colorbar(im1, ax=axes[0, 0], label="W(x,p)")

# Subplot 2: Wigner function after time evolution
im2 = axes[0, 1].imshow(
    Z_dc.T,
    extent=[x[0], x[-1], p[0], p[-1]],
    aspect='auto',
    origin='lower',
    cmap='RdBu_r'
)
axes[0, 1].set_xlabel("x")
axes[0, 1].set_ylabel("p")
axes[0, 1].set_title("With decoherence ($\Lambda$={:.0e})".format(Lambda))
plt.colorbar(im2, ax=axes[0, 1], label="W(x,p)")



#################### marginal plot ####################
# Function to integrate W over p at each x


# Subplot 3: Wigner function after second kick
Z_x_marginal = wf.x_marginal(Z, p)
print(wf.modulation_depth(Z_x_marginal, w=w))
# im3 = axes[1, 0].plot(x, Z_x_marginal, linewidth=2)
im3 = axes[1, 0].plot(wf.smoothing(Z_x_marginal), linewidth=2)
axes[1, 0].set_xlabel("x")
axes[1, 0].set_ylabel("Marginal")


# Subplot 4: Wigner function after second kick and time evolution
Z_dc_x_marginal = wf.x_marginal(Z_dc, p)
print(wf.modulation_depth(Z_dc_x_marginal, w=w))
# im4 = axes[1, 1].plot(x, Z_dc_x_marginal, linewidth=2)
im4 = axes[1, 1].plot(wf.smoothing(Z_dc_x_marginal), linewidth=2)
axes[1, 1].set_xlabel("x")
axes[1, 1].set_ylabel("Marginal")

plt.tight_layout()
plt.show()