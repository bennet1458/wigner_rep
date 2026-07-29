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
omega_0 = 2 * np.pi * 500e3
nbar = 0
sigma_x = np.sqrt(hbar*(2*nbar+1)/(2*m*omega_0))
sigma_p = np.sqrt(hbar*(2*nbar+1)*m*omega_0/2)


# grid
res_q = 100
res_x = 1
n = 5
q_tot = 20e-23


###################
q = q_tot/n
dp = q/res_q
t_pi = 2*np.pi*m*hbar/q**2
dp = round(dp, 30)
q = round(dp * res_q, 30)
q_tot = round(q*n, 30)
dx = dp*t_pi/m/res_x
print(t_pi)
print(dx)
print(sigma_x)
###################

print(dp)
print(q)
print(q_tot)

sigma_p_in_dp_units = int(sigma_p//dp+2)
sigma_x_in_dp_units = int(sigma_x//dx+2)
print(sigma_p_in_dp_units)
print(sigma_x_in_dp_units)
width_x = 2
width_p = 2
p_index = np.arange(-width_p*sigma_p_in_dp_units, res_q*n+width_p*sigma_p_in_dp_units)
print(np.shape(p_index))

p = p_index * dp
x_max = np.max(p)*t_pi/m/dx
# x_index = np.arange(-width_x*sigma_x_in_dp_units, x_max*n//2+width_x*sigma_x_in_dp_units)
x_index = np.arange(-width_x*sigma_x_in_dp_units, x_max*n//2+width_x*sigma_x_in_dp_units)
print(np.shape(x_index))
x = x_index * dx

# p = np.arange(-4*sigma_p_in_q_units, 4*sigma_p_in_q_units+n+1/20, 1/10)*q
# p = np.linspace(-4*sigma_p_in_q_units, 4*sigma_p_in_q_units+n, res)*q
dp = p[1]-p[0]

X = x[:, np.newaxis]
P = p[np.newaxis, :]
phi0 = 0
phi2 = np.pi
########## Calculate Wigner function at each step ##########

W = wf.W0(X, P, sigma_x, sigma_p, hbar)

ks = np.arange(0, n)
print(ks)
phis = 2*np.pi*ks/n
random = np.random.randint(0, 2, n)
print(random)
for k, phi in enumerate(phis):
    if random[k] == 0:
        W = wf.kick(W, X, P, dp, q, phi, hbar)
    if random[k] == 1:
        W = wf.kick(W, X, P, dp, q, -phi, hbar)
        currert_kick = q*(k+1)
        t_pi = 2*np.pi*m*hbar/currert_kick**2
        W = wf.time_evolution(W, X, P, t_pi, m, dx)

Z = W

W2 = wf.W0(X, P, sigma_x, sigma_p, hbar)
for k, phi in enumerate(phis):
    W2 = wf.kick(W2, X, P, dp, q, phi, hbar)
    if k%2 == 0:
        currert_kick = q*(k+1)
        t_pi = 2*np.pi*m*hbar/currert_kick**2
        print(dp*t_pi/m/res_x/dx)
        W2 = wf.time_evolution(W2, X, P, 2*t_pi, m, dx)
        

Z2 = W2

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