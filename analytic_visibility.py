import numpy as np
import matplotlib.pyplot as plt

# -----------------------
# parameters
# -----------------------
hbar = 6.62607015e-34 / (2*np.pi)  # Planck's constant
t_1 = 1.0e-6   # time after first kick
t_2 = 150.0e-6   # time after second kick
m = 1.4e-19   # mass
omega_0 = 2 * np.pi * 50e3
sigma_x = np.sqrt(hbar/(2*m*omega_0))
sigma_p = np.sqrt(hbar*m*omega_0/2)



# q = 10e-23   # larger q → clearer interference
qs = np.linspace(0.5e-22, 2e-22, 20)
### Decoherence parameters ###
Lambdas = [0, 1e20, 1e21, 1e22, 1e23]

tau_o = 2*m*sigma_x**2/hbar

dec = np.zeros((len(Lambdas), len(qs)))

for j, Lambda in enumerate(Lambdas):
    for i, q in enumerate(qs):
            period = hbar*2*np.pi/(q*t_1)*((t_1+t_2)**2+tau_o**2)/(t_1+t_2)
            k_c = 2*np.pi/period
            dec[j, i] = np.exp(-(hbar**2/m**2 * Lambda/3 * (t_1 + t_2)**3) * k_c**2)
    plt.plot(qs, dec[j], label=f"Lambda = {Lambda:.0e}")


plt.xlabel("Kick strength q")
plt.ylabel("Decoherence factor")
plt.legend()
plt.show()

q = 1e-22
t_pi = 2*np.pi*m*hbar/q**2
print(f"t_pi = {t_pi:.2e} s")
t_2s = np.linspace(0, 500e-6, 20)

for j, Lambda in enumerate(Lambdas):
    for i, t_2 in enumerate(t_2s):
        period = hbar*2*np.pi/(q*t_1)*((t_1+t_2)**2+tau_o**2)/(t_1+t_2)
        k_c = 2*np.pi/period
        dec[j, i] = np.exp(-(hbar**2/m**2 * Lambda/3 * (t_1 + t_2)**3) * k_c**2)
    plt.plot(t_2s, dec[j], label=f"Lambda = {Lambda:.0e}")

plt.xlabel("Time between kicks t_2")
plt.ylabel("Decoherence factor")
plt.legend()
plt.show()
