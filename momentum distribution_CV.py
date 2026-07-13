import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import root_scalar
from scipy.integrate import simpson
from scipy.special import erfc



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


qs = np.linspace(1e-40, q_max, 1000)

def norm(mu_b, sigma_b_dash):
    sigma_b = sigma_b_dash*mu_b
    return  erfc((b_min-mu_b)/(sigma_b*np.sqrt(2)))/2


def find_mu(sigma_b_dash):

    def G(mu_b):
        lower = b_min / mu_b

        def integrand(bd):
            return np.exp(-(bd-1)**2/(2*sigma_b_dash**2))/(bd)

        normv = 1/norm(mu_b, sigma_b_dash)
        print('normv')
        print(normv)
        return quad(integrand, lower, np.inf)[0]/np.sqrt(2*np.pi)*normv

    def f(mu_b):
        return A/(sigma_b_dash*mu_b)*G(mu_b) - q_av

    # mus = np.logspace(-7.5, -6.5, 200)
    # for mu in [1e-8, 1e-7, 1e-6, 1e-5]:
    #     print(mu, G(mu))
    # vals = [f(mu) for mu in mus]
    # print(sigma_b_dash)
    # plt.semilogx(mus, vals)
    # plt.axhline(0, color='k')
    # plt.show()

    # mu = 1e-7
    # for s in [0.01, 0.1, 1, 10, 100]:
    #     sigma_b_dash = s
    #     print(s, G(mu), G(mu)/s)

    print(f(10**(-7.5)), f(10**(-6.5)))
    sol = root_scalar(f, bracket=[10**(-7.5), 10**(-6.5)], method='brentq')
    print(sol.root)
    return sol.root

def p2_fix_mu(q, sigma_b_dash):
    mu_b = A/q_av
    sigma_b = sigma_b_dash*mu_b
    factor = A/(np.sqrt(2*np.pi)*sigma_b*q**2)
    normv = 1/norm(mu_b, sigma_b_dash)
    exp = np.exp(-(A/q-mu_b)**2/(2*sigma_b**2))
    return factor*exp*normv


def q_av_p2_fix_mu(q, sigma_b_dash):
    p_ = p2_fix_mu(q, sigma_b_dash)
    q_av = np.sum(q*p_)/np.sum(p_)
    return q_av

def p2(q, sigma_b_dash):
    mu_b = find_mu(sigma_b_dash)
    sigma_b = sigma_b_dash*mu_b
    factor = A/(np.sqrt(2*np.pi)*sigma_b*q**2)
    normv = 1/norm(mu_b, sigma_b_dash)
    exp = np.exp(-(A/q-mu_b)**2/(2*sigma_b**2))
    return factor*exp*normv

def q_av_p2(q, sigma_b_dash):
    p_ = p2(q, sigma_b_dash)
    q_av = np.sum(q*p_)/np.sum(p_)
    return q_av


sigma_b_dash_values = np.linspace(0.01, 2, 100)
q_av_values = np.zeros(len(sigma_b_dash_values))
for i, sigma_b_dash in enumerate(sigma_b_dash_values):
    print(sigma_b_dash)
    q_av_values[i] = q_av_p2_fix_mu(qs, sigma_b_dash)


fig, axes = plt.subplots(3, 1, figsize=(10, 9))

max_p = np.max(p2(qs, 1/8))

axes[0].plot(qs, p2_fix_mu(qs, 1), label = '$\sigma_b^\\prime$ = 1', color='blue')
axes[0].vlines(q_av_p2_fix_mu(qs, 1), 0, max_p, linestyle='--', color='blue')
axes[0].plot(qs, p2_fix_mu(qs, 1/2), label = '$\sigma_b^\\prime$ = 1/2', color = 'darkorange')
axes[0].vlines(q_av_p2_fix_mu(qs, 1/2), 0, max_p, linestyle='--', color='darkorange')
axes[0].plot(qs, p2_fix_mu(qs, 1/3), label = '$\sigma_b^\\prime$ = 1/3', color = 'green')
axes[0].vlines(q_av_p2_fix_mu(qs, 1/3), 0, max_p, linestyle='--', color='green')
axes[0].plot(qs, p2_fix_mu(qs, 1/4), label = '$\sigma_b^\\prime$ = 1/4', color = 'red')
axes[0].vlines(q_av_p2_fix_mu(qs, 1/4), 0, max_p, linestyle='--', color='red')
axes[0].plot(qs, p2_fix_mu(qs, 1/6), label = '$\sigma_b^\\prime$ = 1/6', color = 'purple')
axes[0].vlines(q_av_p2_fix_mu(qs, 1/6), 0, max_p, linestyle='--', color='purple')
axes[0].plot(qs, p2_fix_mu(qs, 1/8), label = '$\sigma_b^\\prime$ = 1/8', color = 'brown')
axes[0].vlines(q_av_p2_fix_mu(qs, 1/8), 0, max_p, linestyle='--', color='brown')
axes[0].set_xlabel("Kick strength $q$ [kg*m/s]")
axes[0].set_ylabel("Kick distribution $p(q)$")
axes[0].grid(True, alpha=0.3)
axes[0].legend()

axes[1].plot(sigma_b_dash_values, q_av_values, color='black')
axes[1].set_xlabel("$\sigma_b^\\prime$")
axes[1].set_ylabel("Average kick strength $q_{av}$ [kg*m/s]")
axes[1].grid(True, alpha=0.3)

axes[2].plot(qs, p2(qs, 1), label = '$\sigma_b^\\prime$ = 1', color='blue')
axes[2].vlines(q_av_p2(qs, 1), 0, max_p, linestyle='--', color='blue')
axes[2].plot(qs, p2(qs, 1/2), label = '$\sigma_b^\\prime$ = 1/2', color = 'darkorange')
axes[2].vlines(q_av_p2(qs, 1/2), 0, max_p, linestyle='--', color='darkorange')
axes[2].plot(qs, p2(qs, 1/3), label = '$\sigma_b^\\prime$ = 1/3', color = 'green')
axes[2].vlines(q_av_p2(qs, 1/3), 0, max_p, linestyle='--', color='green')
axes[2].plot(qs, p2(qs, 1/4), label = '$\sigma_b^\\prime$ = 1/4', color = 'red')
axes[2].vlines(q_av_p2(qs, 1/4), 0, max_p, linestyle='--', color='red')
axes[2].plot(qs, p2(qs, 1/6), label = '$\sigma_b^\\prime$ = 1/6', color = 'purple')
axes[2].vlines(q_av_p2(qs, 1/6), 0, max_p, linestyle='--', color='purple')
axes[2].plot(qs, p2(qs, 1/8), label = '$\sigma_b^\\prime$ = 1/8', color = 'brown')
axes[2].vlines(q_av_p2(qs, 1/8), 0, max_p, linestyle='--', color='black')
axes[2].set_xlabel("Kick strength $q$ [kg*m/s]")
axes[2].set_ylabel("Kick distribution $p(q)$")
axes[2].grid(True, alpha=0.3)
axes[2].legend()


plt.tight_layout()
plt.show()



