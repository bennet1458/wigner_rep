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

t_2 = 150e-6 #150.0e-6   # time after second kick
q = 10e-23#10e-23   # larger q → clearer interference
m = 1.4e-19   # mass
omega_0 = 2 * np.pi * 50e3
sigma_x = np.sqrt(hbar/(2*m*omega_0))
sigma_p = np.sqrt(hbar*m*omega_0/2)
sigma_q = sigma_p/5
print(sigma_q)
sigma_q = 5e-24#1/np.sqrt(q**2*t_1**2/(4*hbar**2*m**2))*10
print(sigma_q)


w = 5

# grid
res = 100#900#200 + int(q/20e-23 * 600)+100
print(res)
x = np.linspace(-4*sigma_x+(q-2*sigma_p)*(t_2)/m, 4*sigma_x+(q+2*sigma_p)*(t_1+t_2)/m, res)
p = np.linspace(-4*sigma_p+q, 4*sigma_p+q, res)
X = x[:, np.newaxis]
P = p[np.newaxis, :]


def p_marginal(q_1, q_2):
    a = (t_1+t_2)**2/(2*sigma_x**2*m**2)+1/(2*sigma_p**2)
    b = (t_1+t_2)/(sigma_x**2*m)*(x+q_2*t_1/(2*m)) + (q_1+q_2)/(2*sigma_p**2)
    c = -1/(2*sigma_x**2)*(x+q_2*t_1/(2*m))**2 - (q_1+q_2)**2/(8*sigma_p**2)
    b_1 = -q_2*t_2/(hbar*m)
    c_1 = q_2*x/hbar
    b_2 = -q_1*(t_1+t_2)/(hbar*m)
    c_2 = q_1/hbar*(x+q_2*t_1/(2*m))
    ffactor = 1/2*np.sqrt(np.pi/a)
    wfactor = 1/4*1/(2*np.pi*sigma_x*sigma_p)

    # inner_exp_1 = np.exp(c+(b**2-b_1**2-b_2**2)/(4*a) - b_1*b_2/(2*a))
    inner_exp_2 = np.exp(c+(b**2-b_1**2-b_2**2)/(4*a) + b_1*b_2/(2*a))
    # cos1 = np.cos(b*(b_1+b_2)/(2*a)+c_1+c_2)
    cos2 = np.cos(b*(b_1-b_2)/(2*a)+c_1-c_2)

    func = wfactor*ffactor*(0
        # +inner_exp_1*cos1
        +inner_exp_2*cos2
        )
    

    return func


marg = p_marginal(q, q)

len_q = 10000
q_1_distr = np.random.normal(q, sigma_q, len_q)
q_2_distr = np.random.normal(q, sigma_q, len_q)
# q_1_distr = np.random.rand(len_q)*sigma_q+ q - sigma_q/2
# q_2_distr = np.random.rand(len_q)*sigma_q+ q - sigma_q/2
marg_av = np.zeros((len_q, len(x)))

for i in range(len_q):
    marg_av[i] = p_marginal(q_1_distr[i], q_2_distr[i])

marg_avd = np.average(marg_av, axis=0)



# -----------------------
# plot subplots
# -----------------------
fig, axes = plt.subplots(2, 1, figsize=(8, 8))

# Subplot 2: Wigner function after second kick
print(wf.modulation_depth(marg, w=w))
im1 = axes[0].plot(x, marg, linewidth=2, linestyle='--', label='exact kick strength')
im1 = axes[0].plot(x, marg_avd, linewidth=2, label='average over varying kick strength', color='red')
im1 = axes[0].plot(x, marg_av[0], linewidth=1,  color='green', label='4 random kick strengths')
im1 = axes[0].plot(x, marg_av[1], linewidth=1,  color='green')
im1 = axes[0].plot(x, marg_av[2], linewidth=1,  color='green')
im1 = axes[0].plot(x, marg_av[3], linewidth=1,  color='green')
axes[0].set_xlabel("x")
axes[0].set_ylabel("Marginal")
axes[0].legend()


# Subplot 2: Wigner function after second kick
print(wf.modulation_depth(marg_avd, w=w))
im3 = axes[1].plot(x, marg_avd, linewidth=2, label='average over varying kick strength', color='red')
axes[1].set_xlabel("x")
axes[1].set_ylabel("Marginal")
axes[1].legend()

plt.tight_layout()
plt.show()