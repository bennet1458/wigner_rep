import numpy as np
import matplotlib.pyplot as plt
import momentum_distribution_funcs as md
import wigner_functions as wf
import p_marginal_funcs as pm

# -----------------------
# parameters
# -----------------------
hbar = 6.62607015e-34 / (2*np.pi)  # Planck's constant
t_0 = 0e-6   # time after first kick (t_0 = t_1)
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

def gauss_dist_q(b_1, b_2, mu_b, sigma_b):
    factor = 1/(2*np.pi*sigma_b**2)
    return np.exp(-((b_1-mu_b)**2+(b_2-mu_b)**2)/(2*sigma_b**2))

def gauss_dist_t(dt_0, dt_1, sigma_t):
    factor = 1/(2*np.pi*sigma_t**2)
    exp = np.exp(-(dt_0**2+dt_1**2)/(2*sigma_t**2))
    return exp


def avg_marginal(sigma_b_dash, sigma_t):
    order_t = 5
    order_b = 1
    mu_b = md.find_mu(sigma_b_dash, b_min=b_min, q_av=q_av, A=A)
    sigma_b = sigma_b_dash*mu_b
    b = np.linspace(np.max([b_min, mu_b-sigma_b*order_b]), mu_b+sigma_b*order_b, res_b)

    b_1 = b[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis]
    b_2 = b[np.newaxis, :, np.newaxis, np.newaxis, np.newaxis]
    q_1 = A/b_1
    q_2 = A/b_2

    dt_0_vals = np.linspace(-order_t*sigma_t, order_t*sigma_t, res_t)
    dt_1_vals = np.linspace(-order_t*sigma_t, order_t*sigma_t, res_t)

    dt_0 = dt_0_vals[np.newaxis, np.newaxis,:, np.newaxis, np.newaxis]
    dt_1 = dt_1_vals[np.newaxis, np.newaxis, np.newaxis, :, np.newaxis]

    t_0_new = t_0 + dt_0
    t_1_new = t_1 - dt_0 + dt_1
    t_2_new = t_2 - dt_1

    marginal = pm.entire_marginal(x, q_1, q_2, t_0_new, t_1_new, t_2_new)

    gauss_q = gauss_dist_q(b_1, b_2, mu_b, sigma_b)
    gauss_t = gauss_dist_t(dt_0, dt_1, sigma_t)
    integrant = marginal*gauss_q*gauss_t
    result = np.sum(integrant, axis=(0, 1, 2, 3))/(res_b**2*res_t**2)
    return result


def central_peak(q_1, q_2):
    alpha = q_1*t_0/m+q_2*(t_0+t_1)/(2*m)
    beta = (t_0+t_1+t_2)/m
    gamma = q_1/2+q_2/2

    a = beta**2/(2*sigma_x**2) + 1/(2*sigma_p**2)
    b = beta*(x_r+alpha)/(sigma_x**2) + gamma/(sigma_p**2)
    c = -(x_r+alpha)**2/(2*sigma_x**2) - gamma**2/(2*sigma_p**2)

    exp = np.exp(c+b**2/(4*a))
    ffactor = 1/2*np.sqrt(np.pi/a)
    wfactor = 1/(2*np.pi*sigma_x*sigma_p)
    func_t6 = wfactor*ffactor*exp
    func_t4 = pm.p_marginal_t4(x_r, q_1, q_2, t_0=0, t_1=t_1, t_2=t_2)
    func_t5 = pm.p_marginal_t5(x_r, q_1, q_2, t_0=0, t_1=t_1, t_2=t_2)
    # plt.plot(x_r, exp[0,0,:])
    # plt.show()
    return func_t6*func_t4*func_t5

def peak_pos(sigma_b_dash):
    mu_b = md.find_mu(sigma_b_dash, b_min=b_min, q_av=q_av, A=A)
    sigma_b = sigma_b_dash*mu_b
    b = np.linspace(mu_b-sigma_b, mu_b+sigma_b, res_b)
    b_1 = b[:, np.newaxis, np.newaxis]
    b_2 = b[np.newaxis, :, np.newaxis]
    q_1 = A/b_1
    q_2 = A/b_2
    marginal = central_peak(q_1, q_2)
    gauss = gauss_dist_q(b_1, b_2, mu_b, sigma_b)
    integrant = marginal*gauss
    result = np.sum(integrant, axis=(0, 1))
    peak = x_r[np.argmax(result)]
    return peak

fig, axes = plt.subplots(1, 2, figsize=(8, 8))


# Subplot 1: Transformed Wigner function
res_x = 20
res_b = 40
res_t = 20
plotwidth = 40

# sigma_b_dash_values = np.logspace(-2, -0.0001, 8)
# sigma_t_values = np.logspace(-3.5, -2, 8)*1e-6
sigma_b_dash_values = np.linspace(0.01, 0.99, 8)
sigma_t_values = np.linspace(0.01, 0.6, 8)*1e-8
visibility_values = np.zeros((len(sigma_b_dash_values), len(sigma_t_values)))
for i, sigma_b_dash in enumerate(sigma_b_dash_values):
    print(i)
    x_r = np.linspace(-4*sigma_x+(q-plotwidth*sigma_p)*(t_2)/m, 4*sigma_x+(q+plotwidth*sigma_p)*(t_1+t_2)/m, 200)
    peak = peak_pos(sigma_b_dash)
    q_peak = peak*m/t_2
    ramsey_wavenumber = q_peak/hbar*(t_1*(t_1+t_2))/((t_1+t_2)**2+(1/omega_0)**2)
    ramsey_period = 2*np.pi/ramsey_wavenumber
    x = np.linspace(-2*ramsey_period+peak, 2*ramsey_period+peak, res_x)


    for j, sigma_t in enumerate(sigma_t_values):
        marginal = avg_marginal(sigma_b_dash, sigma_t)
        visibility_values[i, j] = wf.modulation_depth(marginal, w=w)
    
    if j == 7:
        axes[0].plot(x, marginal, linewidth=2, linestyle='-', label='$\sigma_b^\\prime$ = {:.2f}'.format(sigma_b_dash))
    
axes[0].set_xlabel("x")
axes[0].set_ylabel("Marginal")
axes[0].set_title("Average marginalfor a timing uncertainty of $\sigma_t$ = {:.2e} s".format(sigma_t_values[-1]*1e8))


im1 = axes[1].imshow(visibility_values.T, extent=(sigma_b_dash_values[0], sigma_b_dash_values[-1], sigma_t_values[0], sigma_t_values[-1]), aspect='auto', origin='lower')
axes[1].set_xlabel("Normalized electron beam width $\sigma_b^\\prime$")
axes[1].set_ylabel("Timing uncertainty $\sigma_t$ [s]")
axes[1].set_title("Visibility of the Ramsey fringes")
plt.colorbar(im1, ax=axes, label="Visibility")

plt.show()

