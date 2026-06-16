import numpy as np
import matplotlib.pyplot as plt


sep = 4 # separation of the two wave packets after the kick, in units of sigma_x
frequencies = [1e3, 50e3]  # Frequencies (Hz) - 50 kHz and 100 kHz
freq_colors = ['blue', 'red']  # Colors for different frequencies
freq_labels = ['1 kHz', '50 kHz']

# Physical constants
hbar = 1.054571817e-34  # Reduced Planck constant (J·s)
k_B = 1.380649e-23  # Boltzmann constant (J/K)

# Different mass units (in amu)
mass_units = [1e8, 1e7, 1e6, 1e5]  # amu
mass_linestyles = ['-', '--', '-.', ':']
mass_labels = [f'{m:.1g} amu' for m in mass_units]

print(np.sqrt(mass_units[0]* 1.66053906892e-27*hbar*2*np.pi*frequencies[0]/2))
def q_function(T, hbar=hbar, m=1, omega=1, k_B=k_B):
    q = sep*np.sqrt(hbar*m*omega/2)*np.sqrt(2/(np.exp((hbar*omega)/(k_B*T))-1)+1)
    return q

T_values = np.logspace(-8, 3, 500)

# Create the plot
plt.figure(figsize=(8, 6))

for freq_idx, freq in enumerate(frequencies):
    omega = 2*np.pi*freq  # Angular frequency (rad/s)
    
    for mass_idx, massunits in enumerate(mass_units):
        m = massunits * 1.66053906892e-27  # Mass (kg)
        q_values = q_function(T_values, m=m, omega=omega)
        
        # Create label combining frequency and mass info
        label = f'{mass_labels[mass_idx]} - {freq_labels[freq_idx]}'
        plt.loglog(T_values, q_values, color=freq_colors[freq_idx], 
                   linestyle=mass_linestyles[mass_idx], linewidth=2, label=label)

ax1 = plt.gca()
ax1.set_xlabel('Temperature T [K]', fontsize=12)
# ax1.set_ylabel('Momentum standard deviation $\sigma_p$ [kg*m/s]', fontsize=12)
ax1.set_ylabel('Kick strength $q$ [kg*m/s]', fontsize=12)
# plt.title('Momentum distribution of thermal state', fontsize=14)
# plt.title(f'Minimum kick strength needed for wave package seperation of a thermal state.', fontsize=14)
ax1.legend(fontsize=10, loc='best', ncol=2)
ax1.grid(True, alpha=0.3, which='both')
# plt.hlines(sep*np.sqrt(mass_units[1]*1.66053906892e-27*hbar*2*np.pi*frequencies[1]/2), T_values[0], T_values[-1], colors='gray', linestyles='dashed', label='Minimum kick strength for 1 kHz')



# Add secondary y-axis for d = hbar/q
ax2 = ax1.twinx()
ax2.set_yscale('log')  # Match the log scale of ax1
ax2.set_ylabel('Equivalent grating period $d$ [m]', fontsize=12)

# Get the q-axis limits and convert to d values
q_min, q_max = ax1.get_ylim()
d_max = hbar / q_max  # When q is minimum, d is maximum
d_min = hbar / q_min  # When q is maximum, d is minimum
ax2.set_ylim(d_min, d_max)
plt.show()