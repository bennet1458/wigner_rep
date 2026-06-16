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

### Decoherence parameters ###
diameter = 50e-9
volume = (4/3)*np.pi*(diameter/2)**3
T_i = 315
T_e = 300
def gamma_bb(T):
    return 0.5e44#1.0e47 #1.91e31 * T**(8.38) * np.exp(-0.19 * np.log(T)**2 )
Lambda = 2 * volume * (gamma_bb(T_i) + gamma_bb(T_e))

# gamma_bb_values = np.array([0.0, .5e42, .5e43, .5e44])
# Lambda_values = 4 * volume * gamma_bb_values
Lambda_values = [0, 1e20, 1e21, 1e22, 1e23]

w = 7
t_2 = 150e-6
q_res = 16
q_start = 5e-23
q_end = 20e-23
q_step = q_end / q_res

q_values = np.linspace(q_start, q_end, q_res)
print(q_values)
md = np.zeros((q_res, len(Lambda_values)))

plt.figure(figsize=(8, 6))

for i, Lambda in enumerate(Lambda_values):
    for j, q in enumerate(q_values):
        res = 500 + int(j/q_res * 400)
        print(res)
        x = np.linspace(-4*sigma_x+(q-2*sigma_p)*(t_2)/m, 4*sigma_x+(q+2*sigma_p)*(t_1+t_2)/m, res)
        X = x[:, np.newaxis]
        p = np.linspace(-4*sigma_p+q, 4*sigma_p+q, res)
        P = p[np.newaxis, :]
        W_dc = wf.W0(sigma_x, sigma_p, hbar)
        W_dc = wf.kick(W_dc, q, eta, hbar)
        W_dc = wf.time_evolution(W_dc, t_1, m)
        W_dc = wf.decoherence(W_dc, x, p, t_1, Lambda, m, hbar)
        W_dc = wf.kick(W_dc, q, eta, hbar)
        W_dc = wf.time_evolution(W_dc, t_2, m)
        W_dc = wf.decoherence(W_dc, x, p, t_2, Lambda, m, hbar)
        Z_dc = W_dc(X, P)
        Z_dc_x_marginal = wf.x_marginal(Z_dc, p)
        md[j, i] = wf.modulation_depth(Z_dc_x_marginal, w)
    if Lambda == 0:

        label = 'No Decoherence'
    else:
        label = '$\Lambda$ = {:.0e}'.format(Lambda)
    plt.plot(q_values, md[:, i], marker='o', label=label)

plt.legend()
plt.xlabel('Kick Strength q')
plt.ylabel('Modulation Depth of x-Marginal')
plt.title('Decoherence for a free evolution time of $t_2$ = {} µs'.format(t_2*1e6))
plt.show()