import numpy as np
import matplotlib.pyplot as plt
import momentum_distribution_funcs as md
import wigner_functions as wf
import p_marginal_funcs as pm

# -----------------------
# parameters
# -----------------------
hbar = 6.62607015e-34 / (2*np.pi)  # Planck's constant
t_0 = 0  # time after first kick (t_0 = t_1)
t_1 = 1.0e-6   # time after first kick
m = 1.4e-19   # mass
omega_0 = 2 * np.pi * 50e3
sigma_x = np.sqrt(hbar/(2*m*omega_0))
sigma_p = np.sqrt(hbar*m*omega_0/2)

t_2 = 150e-6 #150.0e-6   # time after second kick
q_av = 10e-23#10e-23   # larger q → clearer interference
q = q_av
q_1 = q
q_2 = q

##########################################
epsilon_0 = 8.854187817e-12  # vacuum permittivity
e = 1.602176634e-19  # elementary charge
c = 2.99792458e8  # speed of light

Q = 500*e  # particle charge
k_e = 1/(4*np.pi*epsilon_0)  # Coulomb's constant
v = 2e4
A = 2*k_e*e*Q/v

q_av = 10e-23
b_min = 50e-9/2
q_max = A/b_min

# grid
res_x = 500
res_b = 100
res_t = 20
plotwidth = 4
x = np.linspace(-4*sigma_x+(q-plotwidth*sigma_p)*(t_2)/m, 4*sigma_x+(q+plotwidth*sigma_p)*(t_1+t_2)/m, res_x)
w = 1


########################################

def gauss_dist_t(dt_0, dt_1, sigma_t):
    factor = 1/(2*np.pi*sigma_t**2)
    exp = np.exp(-(dt_0**2+dt_1**2)/(2*sigma_t**2))
    # plt.figure(figsize=(8, 6))
    # plt.plot(dt_0[:,0,0], exp[:,0,0])
    # plt.show()
    return exp


def avg_marginal(sigma_t):
    
    # dt_0_vals = np.linspace(-t_0, t_1, res_t)
    # dt_1_vals = np.linspace(-t_1, t_2, res_t)
    order = 5
    dt_0_vals = np.linspace(-order*sigma_t, order*sigma_t, res_t)
    dt_1_vals = np.linspace(-order*sigma_t, order*sigma_t, res_t)

    dt_0 = dt_0_vals[:, np.newaxis, np.newaxis]
    dt_1 = dt_1_vals[np.newaxis, :, np.newaxis]

    t_0_new = t_0 + dt_0
    t_1_new = t_1 - dt_0 + dt_1
    t_2_new = t_2 - dt_1

    marginal = pm.entire_marginal(x, q_1, q_2, t_0_new, t_1_new, t_2_new)
    gauss = gauss_dist_t(dt_0, dt_1, sigma_t)
    integrant = marginal*gauss
    result = np.sum(integrant, axis=(0, 1))/res_t**2

    return result


fig, axes = plt.subplots(2, 1, figsize=(8, 8))


# Subplot 1: Transformed Wigner function
# im1 = axes[0].plot(x, entire_marginal(q_1, q_2, t_0, t_1, t_2), linewidth=2, linestyle='-', label='exact kick')
# axes[0].set_xlabel("x")
# axes[0].set_ylabel("Marginal")
# axes[0].legend()


# sigma_t_values = np.array([1, 2, 5, 10])*1e-9
sigma_t_values = np.logspace(-3.5, -2, 4)*1e-6
print(sigma_t_values)
visibility_values = np.zeros(len(sigma_t_values))
for i, sigma_t in enumerate(sigma_t_values):
    marginal = avg_marginal(sigma_t)
    visibility_values[i] = wf.modulation_depth(marginal, w=w)
    axes[0].plot(x, marginal, linewidth=2, linestyle='-', label='$\sigma_t$ = {:.1f}ns'.format(sigma_t*1e9))


axes[0].set_xlabel("x [m]")
axes[0].set_ylabel("Marginal")
axes[0].legend()
axes[1].plot(sigma_t_values, visibility_values, color='black')
axes[1].set_xlabel("$\sigma_t$")
axes[1].set_ylim(0, 1)
axes[1].hlines(np.exp(-1), sigma_t_values[0], sigma_t_values[-1], linestyle='--', color='black')
axes[1].set_ylabel("Visibility of the Ramsey fringes")
plt.tight_layout()
plt.show()

