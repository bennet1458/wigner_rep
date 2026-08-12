import numpy as np
import matplotlib.pyplot as plt
import momentum_distribution_funcs as md
import wigner_functions as wf
import p_marginal_funcs as pm

# -----------------------
# parameters
# -----------------------
hbar = 6.62607015e-34 / (2*np.pi)  # Planck's constant
t_1 = 1.0e-6   # time after first kick
m = 1.4e-19   # mass
omega_0 = 2 * np.pi * 50e3
sigma_x = np.sqrt(hbar/(2*m*omega_0))
sigma_p = np.sqrt(hbar*m*omega_0/2)

t_2 = 150e-6 #150.0e-6   # time after second kick
q_av = 10e-23#10e-23   # larger q → clearer interference
q = q_av

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
w = 0


########################################

def gauss_dist(b_1, b_2, mu_b, sigma_b):
    factor = 1/(2*np.pi*sigma_b**2)
    return factor*np.exp(-((b_1-mu_b)**2+(b_2-mu_b)**2)/(2*sigma_b**2))


def avg_marginal(sigma_b_dash):
    mu_b = md.find_mu(sigma_b_dash, b_min=b_min, q_av=q_av, A=A)
    order = 2
    sigma_b = sigma_b_dash*mu_b
    # step_b = 2*sigma_b/res_b
    b_minimum = np.max([b_min, mu_b-sigma_b*order])
    b_maximum = mu_b+sigma_b*order
    b = np.linspace(b_minimum, b_maximum, res_b)
    # b = np.arange(np.max([b_min, mu_b-sigma_b*order]), mu_b+sigma_b*order, step_b)
    db = b[1]-b[0]
    b_1 = b[:, np.newaxis, np.newaxis]
    b_2 = b[np.newaxis, :, np.newaxis]
    q_1 = A/b_1
    q_2 = A/b_2
    marginal = pm.entire_marginal(x, q_1, q_2, t_0=0, t_1=t_1, t_2=t_2)
    gauss = gauss_dist(b_1, b_2, mu_b, sigma_b)
    integrant = marginal*gauss
    result = np.sum(integrant, axis=(0, 1))*db**2
    norm = md.norm2(mu_b, sigma_b_dash, b_minimum, b_maximum)
    return result/norm**2



fig, axes = plt.subplots(3, 1, figsize=(12, 8))

sigma_b_dash_values = np.logspace(-2, -0.0001, 3)
print(sigma_b_dash_values)

# Subplot 1: Transformed Wigner function
res_x = 1000
res_b = 100
plotwidth = 40
x = np.linspace(-4*sigma_x+(q-plotwidth*sigma_p)*(t_2)/m, 4*sigma_x+(q+plotwidth*sigma_p)*(t_1+t_2)/m, res_x)
im1 = axes[0].plot(x, pm.entire_marginal(x, q_av, q_av, t_0=0, t_1=t_1, t_2=t_2), linewidth=2, linestyle='-', label='exact kick')
visibility_values = np.zeros(len(sigma_b_dash_values))
for i, sigma_b_dash in enumerate(sigma_b_dash_values):
    marginal = avg_marginal(sigma_b_dash)
    visibility_values[i] = wf.modulation_depth(marginal, w=w)
    axes[0].plot(x, marginal, linewidth=2, linestyle='-', label='$\sigma_b/\mu_b$ = {:.2f}'.format(sigma_b_dash))

axes[0].set_xlabel("x [m]")
axes[0].set_ylabel("Marginal")
axes[0].legend()

################################################################################
res_x = 500
res_b = 100
plotwidth = 4
x = np.linspace(-4*sigma_x+(q-plotwidth*sigma_p)*(t_2)/m, 4*sigma_x+(q+plotwidth*sigma_p)*(t_1+t_2)/m, res_x)
im1 = axes[1].plot(x, pm.entire_marginal(x, q_av, q_av, t_0=0, t_1=t_1, t_2=t_2), linewidth=2, linestyle='-', label='exact kick')
visibility_values_2 = np.zeros(len(sigma_b_dash_values))
for i, sigma_b_dash in enumerate(sigma_b_dash_values):
    marginal = avg_marginal(sigma_b_dash)
    visibility_values_2[i] = wf.modulation_depth(marginal, w=w)
    axes[1].plot(x, marginal, linewidth=2, linestyle='-', label='$\sigma_b^\\prime$ = {:.2f}'.format(sigma_b_dash))

axes[1].set_xlabel("x [m]")
axes[1].set_ylabel("Marginal")
# axes[1].legend()


################################################################################


axes[2].plot(sigma_b_dash_values, visibility_values, label='maximum visibility')
axes[2].plot(sigma_b_dash_values, visibility_values_2, label=' visibility at the center')
axes[2].set_xlabel("$\sigma_b^\\prime$")
axes[2].set_ylim(0, 1)
axes[2].hlines(np.exp(-1), sigma_b_dash_values[0], sigma_b_dash_values[-1], linestyle='--', color='black', label='$e^{-1}$')
axes[2].set_ylabel("Visibility of the Ramsey fringes")
axes[2].legend()
axes[2].grid()

plt.tight_layout()
plt.show()

