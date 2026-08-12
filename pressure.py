import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.colors import BoundaryNorm

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

##########################################
epsilon_0 = 8.854187817e-12  # vacuum permittivity
e = 1.602176634e-19  # elementary charge
c = 2.99792458e8  # speed of light

Q = 500*e  # particle charge
k_e = 1/(4*np.pi*epsilon_0)  # Coulomb's constant
v = 2e4
A = 2*k_e*e*Q/v

d = q*t_1/m
print(d)

amu = 1.66053906892e-27
m_amu = m/amu
print(m_amu)
k_B = 1.380649e-23
m_g = 2*amu
T_e = 300
lambda_th = np.sqrt(2*np.pi*hbar**2/(m_g*k_B*T_e))
print(lambda_th)

v = q/m
print(v)

v_av = np.sqrt(8*k_B*T_e/(np.pi*m_g))
print(v_av)

# t = t_1+t_2
r_pa = 25e-9
P_x = 0.9
print('##########')
print(np.sqrt(9*k_B*T_e*m_g/(8*np.pi))*1/((t_1+t_2)*r_pa**2)*np.log(1/P_x))
t_s = np.logspace(-6, -1, 100)
r_s = np.linspace(10e-9, 100e-9, 100)


vals_p = np.zeros((len(t_s), len(r_s)))
for i, t in enumerate(t_s):
    for j, r in enumerate(r_s):
        vals_p[i, j] = np.sqrt(9*k_B*T_e*m_g/(8*np.pi))*1/(t*r**2)*np.log(1/P_x)

bounds = [1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4]
norm = BoundaryNorm(bounds, ncolors=plt.get_cmap("viridis").N)

fig, ax = plt.subplots()
# pcm = ax.pcolormesh(
#     r_s/1e-9,
#     t_s/1e-3,
#     vals_p/100,
#     shading="auto",
#     norm=LogNorm(vmin=np.min(vals_p)/100, vmax=np.max(vals_p)/100),
#     cmap="viridis"
# )
pcm = ax.pcolormesh(
    r_s/1e-9,
    t_s/1e-3,
    vals_p/100,
    shading="auto",
    norm=norm,
    cmap="viridis"
)


ax.scatter(r_pa/1e-9, (t_1+t_2)/1e-3, color="white", marker=".", s=100, label="chosen parameters")
# ax.legend()
ax.text(r_pa/1e-9*1.1, (t_1+t_2)/1e-3, "chosen parameters", color="white", fontsize=12, ha="left", va="bottom")
ax.set_yscale("log")
ax.set_xlabel("particle radius r [nm]")
ax.set_ylabel("experimental time t [ms]")

rho = 2200

def r_to_m(r):
    r = r*1e-9
    return (4/3) * np.pi * rho * r**3 / amu   # mass in amu

def m_to_r(m):
    r = ((m * amu) / ((4/3) * np.pi * rho))**(1/3)
    return r/1e-9

secax = ax.secondary_xaxis('top', functions=(r_to_m, m_to_r))
secax.set_xlabel("mass [amu]")
secax.set_xticks([1e7, 1e8, 1e9, 1e10])
secax.set_xticklabels([r"$10^7$", r"$10^8$", r"$10^9$", r"$10^{10}$"])

# plt.colorbar(pcm, ax=ax, label="required pressure p [mbar]")
cbar = plt.colorbar(pcm, ax=ax, label="required pressure p [mbar]")
cbar.set_ticks(bounds)
cbar.set_ticklabels([
    r"$10^{-12}$",
    r"$10^{-11}$",
    r"$10^{-10}$",
    r"$10^{-9}$",
    r"$10^{-8}$",
    r"$10^{-7}$",
    r"$10^{-6}$",
    r"$10^{-5}$",
    r"$10^{-4}$"
])
plt.tight_layout()
plt.show()

