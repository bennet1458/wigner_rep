import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

# -----------------------
# parameters
# -----------------------
hbar = 6.62607015e-34 / (2*np.pi)  # Planck's constant
h = 6.62607015e-34  # Planck's constant
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
r = 25e-9
V = (4/3)*np.pi*(r/2)**3
c = 2.99792458e8  # speed of light
n_L= 1.44+1j*2.5e-9
eps_r = n_L**2
c = 2.99792458e8  # speed of light
wavelength = 1550e-9
tau_f = t_1+t_2
tau_c = 1000e-6
T_e = 300
beta = np.imag((eps_r-1)/(eps_r+2))/np.real((eps_r-1)/(eps_r+2))
P_abs = m*omega_0**2*c*wavelength/(2*np.pi)*beta

def p_bb(T):
    return T**(-5.79)*np.exp(3.14*np.log(T)**2-0.265*np.log(T)**3)

def eq(T):
    return P_abs*tau_c/(tau_c+tau_f)+V*(p_bb(T_e)-p_bb(T))

def eq2(T):
    return P_abs+V*(p_bb(T_e)-p_bb(T))

T_root = brentq(eq, 300.0, 400.0)

print(f"Root: T = {T_root:.3f} K")
print(f"eq(T) = {eq(T_root):.3e}")

T2_root = brentq(eq2, 300.0, 400.0)
print(f"Root: T = {T2_root:.3f} K")
print(f"eq2(T) = {eq2(T2_root):.3e}")

def p_bb(T):
    return 1e40

Lambda = 2*V*(p_bb(T2_root)+p_bb(T_e))
print(Lambda)