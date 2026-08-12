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
sigma_q = 1/np.sqrt(q**2*t_1**2/(4*hbar**2*m**2))*10
print(sigma_q)


w = 5

# grid
res = 900#200 + int(q/20e-23 * 600)+100
print(res)



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


q_res = 30
q_start = 5e-23
q_end = 50e-23
q_step = q_end / q_res
q_values = np.linspace(q_start, q_end, q_res)
len_q = 10000
amp_marg = np.zeros(q_res)
amp_marg_av = np.zeros(q_res)

for j, q in enumerate(q_values):
    x = np.linspace(-4*sigma_x+(q-2*sigma_p)*(t_2)/m, 4*sigma_x+(q+2*sigma_p)*(t_1+t_2)/m, res)
    X = x[:, np.newaxis]

    marg = p_marginal(q, q)
    amp_marg[j] = (np.max(marg) - np.min(marg))/2
    marg_av = np.zeros((len_q, len(x)))


    q_1_distr = np.random.normal(q, sigma_q, len_q)
    q_2_distr = np.random.normal(q, sigma_q, len_q)

    for i in range(len_q):
        marg_av[i] = p_marginal(q_1_distr[i], q_2_distr[i])
    
    marg_avd = np.average(marg_av, axis=0)
    amp_marg_av[j] = (np.max(marg_avd) - np.min(marg_avd))/2



# -----------------------
# plot subplots
# -----------------------
fig, axes = plt.subplots(1, 2, figsize=(8, 8))

# Subplot 2: Wigner function after second kick
im1 = axes[0].plot(q_values, amp_marg, linewidth=2, linestyle='--', label='exact kick strength')
im1 = axes[0].plot(q_values, amp_marg_av, linewidth=2,  color='green', label='average over varying kick strength')
axes[0].set_xlabel("Average kick strength $q$")
axes[0].set_ylabel("Amplitude of Marginal")
axes[0].legend()

im2 = axes[1].plot(q_values, amp_marg_av/amp_marg, linewidth=2,  color='blue')
axes[1].set_xlabel("Average kick strength $q$")
axes[1].set_ylabel("Visibility of the Ramsey fringes")

plt.tight_layout()
plt.show()