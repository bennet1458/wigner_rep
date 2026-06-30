import numpy as np

def Lambda_x(rho, omega0, V, hbar, lam, eps_r):
    return (4*np.pi**2/3) * (rho * omega0**2 * V**2) / (hbar * lam**3) * np.real(
        (eps_r - 1) / (eps_r + 2)
    )


hbar = 1.054e-34

omega0 = 2*np.pi*50e3
d = 50e-9
M = 1.4e-19
lam = 1550e-9
n_L= 1.44+1j*2.5e-9

eps_r = n_L**2
print(eps_r)
V = (4/3)*np.pi*(d/2)**3
rho = M/V 

print(Lambda_x(rho, omega0, V, hbar, lam, eps_r))