import numpy as np
import matplotlib.pyplot as plt

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
q_1 = q
q_2 = q

##########################################
epsilon_0 = 8.854187817e-12  # vacuum permittivity
e = 1.602176634e-19  # elementary charge
c = 2.99792458e8  # speed of light
Q = 500*e  # particle charge
k_e = 1/(4*np.pi*epsilon_0)  # Coulomb's constant
v = 0.5*c  # electron velocity
A = 2*k_e*e*Q/v

G = np.array([0, 6.09215, 1.80936, 0.487572, 0.273211, 0.209277, 0.17175, 0.145973, 0.127053, 0.112537, 0.101032])

mu_b_dash = 4
sigma_b = A/q_av*G[mu_b_dash]
mu_b = mu_b_dash*sigma_b
print(sigma_b)
print(mu_b)

# grid
res_x = 200
res_b = 200
x = np.linspace(-4*sigma_x+(q-10*sigma_p)*(t_2)/m, 4*sigma_x+(q+10*sigma_p)*(t_1+t_2)/m, res_x)
b = np.linspace(mu_b-(mu_b_dash-0.01)*sigma_b, mu_b+mu_b_dash*sigma_b, res_b)

b_1 = b[:, np.newaxis, np.newaxis]
b_2 = b[np.newaxis, :, np.newaxis]
q_1 = A/b_1
q_2 = A/b_2
idx_p_av = np.argmin(np.abs(q_1[:, 0, 0]-q_av))
idx_mu_b = np.argmin(np.abs(b - mu_b))


def p_marginal():
    a = (t_1+t_2)**2/(2*sigma_x**2*m**2)+1/(2*sigma_p**2)
    b = (t_1+t_2)/(sigma_x**2*m)*(x+q_2*t_1/(2*m)) + (q_1+q_2)/(2*sigma_p**2)
    c = -1/(2*sigma_x**2)*(x+q_2*t_1/(2*m))**2 - (q_1+q_2)**2/(8*sigma_p**2)
    b_1 = -q_2*t_2/(hbar*m)
    c_1 = q_2*x/hbar
    b_2 = -q_1*(t_1+t_2)/(hbar*m)
    c_2 = q_1/hbar*(x+q_2*t_1/(2*m))
    ffactor = 1/2*np.sqrt(np.pi/a)
    wfactor = 1/4*1/(2*np.pi*sigma_x*sigma_p)

    inner_exp_1 = np.exp(c+(b**2-b_1**2-b_2**2)/(4*a) - b_1*b_2/(2*a))
    inner_exp_2 = np.exp(c+(b**2-b_1**2-b_2**2)/(4*a) + b_1*b_2/(2*a))
    cos1 = np.cos(b*(b_1+b_2)/(2*a)+c_1+c_2)
    cos2 = np.cos(b*(b_1-b_2)/(2*a)+c_1-c_2)

    func = wfactor*ffactor*(0
        # +inner_exp_1*cos1
        +inner_exp_2*cos2
        )

    return func


def p_marginal_g1():
    a = (t_1+t_2)**2/(2*sigma_x**2*m**2)+1/(2*sigma_p**2)
    b = (t_1+t_2)/(sigma_x**2*m)*(x+q_2*t_1/m) + q_2/(sigma_p**2)
    c = -1/(2*sigma_x**2)*(x+q_2*t_1/m)**2 - q_2**2/(2*sigma_p**2)
    
    ffactor = np.sqrt(np.pi/a)
    wfactor = 1/16*1/(2*np.pi*sigma_x*sigma_p)

    exp = np.exp(c+(b**2)/(4*a))
    func = wfactor*ffactor*exp
    return func


def p_marginal_g2():
    a = (t_1+t_2)**2/(2*sigma_x**2*m**2)+1/(2*sigma_p**2)
    b = (t_1+t_2)/(sigma_x**2*m)*x + q_1/(sigma_p**2)
    c = -1/(2*sigma_x**2)*x**2 - q_1**2/(2*sigma_p**2)
    
    ffactor = np.sqrt(np.pi/a)
    wfactor = 1/16*1/(2*np.pi*sigma_x*sigma_p)

    exp = np.exp(c+(b**2)/(4*a))
    func = wfactor*ffactor*exp
    return func

def gauss_dist():
    return np.exp(-(b_1**2+b_2**2)/(2*sigma_b**2))

marginal = p_marginal()+p_marginal_g1()+p_marginal_g2()
marginal_ft =np.fft.fft2(marginal, axes=(0,1))
gauss = gauss_dist()
# plt.plot(b, gauss[0, :, 0])
# print(gauss[idx_mu_b, idx_mu_b, :].shape)
gauss_ft = np.fft.fft2(gauss, axes=(0,1))
# plt.plot(np.abs(gauss_ft[0, :, 0]))
inverse = np.fft.ifft2(gauss_ft*marginal_ft, axes=(0,1))
factor = 1/(2*np.pi*sigma_b**2)
result = factor*np.real(inverse[idx_mu_b, idx_mu_b, :])


###############
# weights = np.exp(-(b-mu_b)**2/(2*sigma_b**2))
# weights /= np.sum(weights)

# avg = np.tensordot(weights,
#                    np.tensordot(weights,
#                                 marginal,
#                                 axes=(0,1)),
#                    axes=(0,0))
# ##########
# -----------------------
# plot subplots
# -----------------------
fig, axes = plt.subplots(2, 1, figsize=(8, 8))


# Subplot 1: Transformed Wigner function
im1 = axes[0].plot(x, marginal[idx_p_av, idx_p_av, :], linewidth=2, linestyle='-', label='analytic marginal')
axes[0].set_xlabel("x")
axes[0].set_ylabel("Marginal")
axes[0].legend()


# Subplot 2: Wigner function after second kick
im3 = axes[1].plot(x, result, linewidth=2, linestyle='-', label='analytic marginal')
# im3 = axes[1].plot(x, avg, linewidth=2)

axes[1].set_xlabel("x")
axes[1].set_ylabel("Marginal")
axes[1].legend()

print(np.average(p_marginal()))
plt.tight_layout()
plt.show()

print(idx_p_av)
print(idx_mu_b)
