import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import wigner_functions as wf
from scipy.optimize import curve_fit

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
res = 200#900#200 + int(q/20e-23 * 600)+100
print(res)
x = np.linspace(-4*sigma_x+(q-2*sigma_p)*(t_2)/m, 4*sigma_x+(q+2*sigma_p)*(t_1+t_2)/m, res)
X = x[:, np.newaxis]


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

# def cos(x, vis):
#     q_1 = q
#     q_2 = q
#     a = (t_1+t_2)**2/(2*sigma_x**2*m**2)+1/(2*sigma_p**2)
#     b = (t_1+t_2)/(sigma_x**2*m)*(x+q_2*t_1/(2*m)) + (q_1+q_2)/(2*sigma_p**2)
#     b_1 = -q_2*t_2/(hbar*m)
#     c_1 = q_2*x/hbar
#     b_2 = -q_1*(t_1+t_2)/(hbar*m)
#     c_2 = q_1/hbar*(x+q_2*t_1/(2*m))
#     return vis * np.cos(b*(b_1-b_2)/(2*a)+c_1-c_2)

def cos(x, vis):
    q_1=q
    q_2=q
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
    return vis * func

sigma_q_res = 20
sigma_q_start = 1e-25
sigma_q_end = 2e-23
sigma_q_step = sigma_q_end / sigma_q_res
sigma_q_values = np.linspace(sigma_q_start, sigma_q_end, sigma_q_res)

len_q = 1000
amp_marg_av = np.zeros(sigma_q_res)

marg = p_marginal(q, q)
popt_cos, _ = curve_fit(cos, x, marg, p0=[1])
amp_marg = popt_cos[0]
print("amp_marg: ", amp_marg)


for j, sigma_q in enumerate(sigma_q_values):
    
    marg_av = np.zeros((len_q, len(x)))

    q_1_distr = np.random.normal(q, sigma_q, len_q)
    q_2_distr = np.random.normal(q, sigma_q, len_q)

    for i in range(len_q):
        marg_av[i] = p_marginal(q_1_distr[i], q_2_distr[i])
    
    marg_avd = np.average(marg_av, axis=0)
    popt_cos, _ = curve_fit(cos, x, marg_avd, p0=[1])
    amp_marg_av[j] = popt_cos[0]



# -----------------------
# plot subplots
# -----------------------
fig = plt.figure(figsize=(8, 6))

# Subplot 2: Wigner function after second kick
plt.plot(sigma_q_values, amp_marg_av/amp_marg, linewidth=2,  color='blue', label='Numerically determined values')


def func(x, b):
    return np.exp(-(x*b))

def func2(x, b):
    return np.exp(-(x*b)**2)

popt, pcov = curve_fit(func, sigma_q_values, amp_marg_av/amp_marg, p0=[3e23])
popt2, pcov2 = curve_fit(func2, sigma_q_values, amp_marg_av/amp_marg, p0=[3e23])
print(popt)
plt.plot(sigma_q_values, func(sigma_q_values, *popt), 'r-', label='fit exp(-b*$\sigma_q$)')
plt.plot(sigma_q_values, func2(sigma_q_values, *popt2), 'g-', label='fit exp(-b*$\sigma_q^2$)')
plt.vlines(x=sigma_p, ymin=0, ymax=1, color='black', linestyle='--', label='$\sigma_p$')
plt.vlines(x=1/np.sqrt(q**2*t_1**2/(4*hbar**2*m**2)), ymin=0, ymax=1, color='black', linestyle=':', label='calculated $\sigma_{q, max}=1/sqrt(C)$')
plt.grid()
plt.xlabel("Kick distribution width $\sigma_q$")
plt.ylabel("Visibility of the Ramsey fringes")
plt.legend()



plt.tight_layout()
plt.show()

print("calculated sigma_q: ", np.sqrt(q**2*t_1**2/(4*hbar**2*m**2)))