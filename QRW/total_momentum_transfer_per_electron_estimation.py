import numpy as np
import matplotlib.pyplot as plt

m_e = 9.10938356e-31  # mass of electron
amu = 1.66053906660e-27  # atomic mass unit in kg
 # mass of particle (100,000 amu)
c = 299792458  # speed of light in m/s


def delta_p(v_units, m_pa_units):
    v = v_units * c
    m_pa = m_pa_units * amu
    gamma = 1/np.sqrt(1 - (v/c)**2)
    return np.sqrt((gamma-1) * m_e * c**2 * 2 * m_pa)

def delta_p0(v_units):
    v = v_units * c
    gamma = 1/np.sqrt(1 - (v/c)**2)
    return 2 * gamma * m_e * v

vs_units = np.linspace(0.1, 0.9, 200)
mass_units = [1e8, 1e7, 1e6, 1e5]
plt.figure(figsize=(5, 4))
for m in mass_units:
    dp = delta_p(vs_units, m)
    plt.semilogy(vs_units, dp, label=f'{m:.1g} amu')

plt.semilogy(vs_units, delta_p0(vs_units), label='Single scattering event', linestyle='dashed')
plt.xlabel('Electron Velocity (c)')
plt.ylabel('Momentum Transfer (kg m/s)')
plt.legend()
plt.grid(True)
plt.show()
