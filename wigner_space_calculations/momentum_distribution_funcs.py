import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import root_scalar
from scipy.special import erfc

epsilon_0 = 8.854187817e-12  # vacuum permittivity
e = 1.602176634e-19  # elementary charge
c = 2.99792458e8  # speed of light


def norm(mu_b, sigma_b_dash, b_min):
    sigma_b = sigma_b_dash*mu_b
    return erfc((b_min-mu_b)/(sigma_b*np.sqrt(2)))/2


def find_mu(sigma_b_dash, b_min, q_av, A):

    def G(mu_b):
        lower = b_min / mu_b

        def integrand(bd):
            return np.exp(-(bd-1)**2/(2*sigma_b_dash**2))/(bd)

        normv = 1/norm(mu_b, sigma_b_dash, b_min)
        return quad(integrand, lower, np.inf)[0]/np.sqrt(2*np.pi)*normv

    def f(mu_b):
        return A/(sigma_b_dash*mu_b)*G(mu_b) - q_av
    

    sol = root_scalar(f, bracket=[A/q_av, 2*A/q_av], method='brentq')
    return sol.root

