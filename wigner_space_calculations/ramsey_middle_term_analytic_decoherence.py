import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import wigner_functions as wf

# -----------------------
# parameters
# -----------------------
hbar = 6.62607015e-34 / (2*np.pi)  # Planck's constant
eta = .5
t_1 = 1.0e-6   # time after first kick
m = 1.4e-19   # mass
omega_0 = 2 * np.pi * 50e3
sigma_x = np.sqrt(hbar/(2*m*omega_0))
sigma_p = np.sqrt(hbar*m*omega_0/2)
tau_o = 2 * m * sigma_x**2 / hbar

Lambda_values = [0, 1e20, 1e21, 1e22, 1e23]
Lambda_fix = 1e20
t_2_fix = 150e-6
q_fix = 10e-23

w = 5
t_res = 300
t_end = 5000e-6
t_start = 0

q_res = 300
q_start = 0
q_end = 100e-23
q_step = q_end / q_res

t_2_values = np.linspace(t_start, t_end, t_res)
q_values = np.linspace(q_start, q_end, q_res)
dec = np.zeros((t_res, q_res))

for j, q in enumerate(q_values):
    for i, t_2 in enumerate(t_2_values):
        period = hbar*2*np.pi/(2*q*t_1)*((t_1+t_2)**2+tau_o**2)/(t_1+t_2)
        k_c = 2*np.pi/period
        dec[i, j] = np.exp(-(hbar**2/m**2 * Lambda_fix/3 * (t_1 + t_2)**3) * k_c**2)

# Create figure with 3 subplots in a row
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Heatmap
im = ax1.imshow(dec, extent=(q_values[0], q_values[-1], t_2_values[0], t_2_values[-1]), aspect='auto', origin='lower')
plt.colorbar(im, ax=ax1, label='Decoherence Factor')
ax1.hlines(t_2_fix, q_values[0], q_values[-1], colors='white', linestyle='--', label='$t_2$ = {} µs'.format(t_2_fix*1e6))
ax1.vlines(q_fix, t_2_values[0], t_2_values[-1], colors='white', linestyle='--', label='$q$ = {} eV'.format(q_fix))
ax1.legend()
ax1.set_xlabel('Kick strength $q$ [kg*m/s]')
ax1.set_ylabel('Time after second kick $t_2$ [s]')
ax1.set_title('Decoherence of the Ramsey Fringes ($\Lambda$ = {:.0e})'.format(Lambda_fix))

# Plot 2: Decoherence vs t_2 for different Lambda values
for i, Lambda in enumerate(Lambda_values):
    for j, t_2 in enumerate(t_2_values):
        period = hbar*2*np.pi/(2*q_fix*t_1)*((t_1+t_2)**2+tau_o**2)/(t_1+t_2)
        k_c = 2*np.pi/period
        dec[i, j] = np.exp(-(hbar**2/m**2 * Lambda/3 * (t_1 + t_2)**3) * k_c**2)
    if Lambda == 0:
        label = 'No Decoherence'
    else:
        label = '$\Lambda$ = {:.0e}'.format(Lambda)
    ax2.plot(t_2_values, dec[i], label=label)
ax2.set_xlabel('Time after second kick $t_2$ [s]')
ax2.set_ylabel('Decoherence Factor')
ax2.set_title('Decoherence for a kick strength of $q$ = {} eV'.format(q_fix))
ax2.legend()

# Plot 3: Decoherence vs q for different Lambda values
for i, Lambda in enumerate(Lambda_values):
    for j, q in enumerate(q_values):
        period = hbar*2*np.pi/(2*q*t_1)*((t_1+t_2_fix)**2+tau_o**2)/(t_1+t_2_fix)
        k_c = 2*np.pi/period
        dec[i, j] = np.exp(-(hbar**2/m**2 * Lambda/3 * (t_1 + t_2_fix)**3) * k_c**2)

    if Lambda == 0:
        label = 'No Decoherence'
    else:
        label = '$\Lambda$ = {:.0e}'.format(Lambda)
    ax3.plot(q_values, dec[i], label=label)
ax3.set_xlabel('Kick strength $q$ [kg*m/s]')
ax3.set_ylabel('Decoherence Factor')
ax3.set_title('Decoherence for a free evolution time of $t_2$ = {} µs'.format(t_2_fix*1e6))
ax3.legend()

plt.tight_layout()
plt.show()
