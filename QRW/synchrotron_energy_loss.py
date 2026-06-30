import numpy as np
import matplotlib.pyplot as plt

c = 299792458  # speed of light in m/s
epsilon_0 = 8.854187817e-12  # vacuum permittivity in F/m
m_e = 9.10938356e-31  # mass of electron
e = 1.602176634e-19  # elementary charge in C

# Range of velocities as fraction of c
betas = np.linspace(0.1, 0.9, 100)
velocities = betas * c

# Four different magnetic field values
B_values = [0.01, 0.1, 1.0]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

for B in B_values:
    Delta_E_ratios = []
    R_values = []
    
    for v in velocities:
        gamma = 1/np.sqrt(1 - (v/c)**2)
        beta = v/c
        
        R = m_e * gamma * v / (e * B)
        Delta_E = 1/(3 * epsilon_0) * (e**2 * beta**4 * gamma**4) / R
        E_el = (gamma - 1) * m_e * c**2
        
        Delta_E_ratios.append(Delta_E / E_el)
        R_values.append(R)
    
    # Plot energy loss on left
    ax1.semilogy(betas, Delta_E_ratios, label=f'B = {B:.0e} T')
    
    # Plot R on right
    ax2.semilogy(betas, R_values, label=f'B = {B:.0e} T')

ax1.set_xlabel('Velocity (fraction of c)')
ax1.set_ylabel('Energy loss per round (ΔE / E_el)')
ax1.legend()
ax1.grid(True)

ax2.set_xlabel('Velocity (fraction of c)')
ax2.set_ylabel('Radius R (m)')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()