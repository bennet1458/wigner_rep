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

# ### Decoherence parameters ###
# diameter = 50e-9
# volume = (4/3)*np.pi*(diameter/2)**3
# T_i = 315
# T_e = 300
# def gamma_bb(T):
#     return 0.5e44#1.0e47 #1.91e31 * T**(8.38) * np.exp(-0.19 * np.log(T)**2 )
# Lambda = 2 * volume * (gamma_bb(T_i) + gamma_bb(T_e))
Lambda = 1e22

w = 5
t_res = 20
t_end = 500e-6
t_step = t_end / t_res

q_res = 16
q_start = 5e-23
q_end = 20e-23
q_step = q_end / q_res

t_2_values = np.linspace(t_step, t_end, t_res)
print(t_2_values)
q_values = np.linspace(q_start, q_end, q_res)
print(q_values)
md = np.zeros((t_res, q_res))

for j, q in enumerate(q_values):
    # res = 500 + int(j/q_res * 400)
    res = 900
    print(res)
    p = np.linspace(-4*sigma_p+q, 4*sigma_p+q, res)
    P = p[np.newaxis, :]

    for i, t_2 in enumerate(t_2_values):
        x = np.linspace(-4*sigma_x+(q-2*sigma_p)*(t_2)/m, 4*sigma_x+(q+2*sigma_p)*(t_1+t_2)/m, res)
        X = x[:, np.newaxis]
        

        W_dc = wf.W0(sigma_x, sigma_p, hbar)
        W_dc = wf.kick(W_dc, q, eta, hbar)
        W_dc = wf.time_evolution(W_dc, t_1, m)
        W_dc = wf.decoherence(W_dc, x, p, t_1, Lambda, m, hbar)
        W_dc = wf.kick(W_dc, q, eta, hbar)
        W_dc = wf.time_evolution(W_dc, t_2, m)
        W_dc = wf.decoherence(W_dc, x, p, t_2, Lambda, m, hbar)
        
        
        
   
Z_dc = W_dc(X, P)
Z_dc_x_marginal = wf.x_marginal(Z_dc, p)
md[i, j] = wf.modulation_depth(Z_dc_x_marginal, w)

plt.figure(figsize=(8, 6))
plt.imshow(md, extent=(q_values[0], q_values[-1], t_2_values[0], t_2_values[-1]), aspect='auto', origin='lower')
plt.colorbar(label='Modulation Depth')
plt.xlabel('Kick Strength q')
plt.ylabel('Time after second kick t_2')
plt.title('Modulation Depth of Wigner Function Marginal ($\Lambda$ = {:.2e})'.format(Lambda))
plt.show()