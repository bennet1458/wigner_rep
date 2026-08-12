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

t_2 = 150e-6 #150.0e-6   # time after second kick
q = 10e-23#10e-23   # larger q → clearer interference
q_1 = q
q_2 = q
w = 5

# grid
res = 900#200 + int(q/20e-23 * 600)+100
print(res)
x = np.linspace(-4*sigma_x+(q-2*sigma_p)*(t_2)/m, 4*sigma_x+(q+2*sigma_p)*(t_1+t_2)/m, res)
p = np.linspace(-4*sigma_p+q, 4*sigma_p+q, res)
X = x[:, np.newaxis]
P = p[np.newaxis, :]

########## Calculate Wigner function at each step ##########

W = wf.W0(sigma_x, sigma_p, hbar)

W = wf.kick(W, q, eta, hbar)

W = wf.time_evolution(W, t_1, m)

W = wf.kick(W, q, eta, hbar)

W = wf.time_evolution(W, t_2, m)

Z = W(X, P)


def p_marginal():
    a = (t_1+t_2)**2/(2*sigma_x**2*m**2)+1/(2*sigma_p**2)
    b = (t_1+t_2)/(sigma_x**2*m)*(x+q_2*t_1/(2*m)) + (q_1+q_2)/(2*sigma_p**2)
    c = -1/(2*sigma_x**2)*(x+q_2*t_1/(2*m))**2 - (q_1+q_2)**2/(8*sigma_p**2)
    b_1 = -q_2*t_2/(hbar*m)
    c_1 = q_2*x/hbar
    b_2 = -q_1*(t_1+t_2)/(hbar*m)
    c_2 = q_1/hbar*(x+q_2*t_1/(2*m))
    ffactor = 1/2*np.sqrt(np.pi/a)
    wfactor = 1/4*1/(2*np.pi*sigma_x*sigma_p)

    inner_exp_1 = np.exp(c+(b**2-b_1**2-b_2**2)/(4*a) - b_1*b_2/(2*a))
    inner_exp_2 = np.exp(c+(b**2-b_1**2-b_2**2)/(4*a) + b_1*b_2/(2*a))
    cos1 = np.cos(b*(b_1+b_2)/(2*a)+c_1+c_2)
    cos2 = np.cos(b*(b_1-b_2)/(2*a)+c_1-c_2)

    func = wfactor*ffactor*(0
        # +inner_exp_1*cos1
        +inner_exp_2*cos2
        )
    
    E1 = c + (b**2-b_1**2-b_2**2)/(4*a) - b_1*b_2/(2*a)
    E2 = c + (b**2-b_1**2-b_2**2)/(4*a) + b_1*b_2/(2*a)

    print(E1.min(), E1.max())
    print(E2.min(), E2.max())

    return func


def p_marginal_g1():
    a = (t_1+t_2)**2/(2*sigma_x**2*m**2)+1/(2*sigma_p**2)
    b = (t_1+t_2)/(sigma_x**2*m)*(x+q_2*t_1/m) + q_2/(sigma_p**2)
    c = -1/(2*sigma_x**2)*(x+q_2*t_1/m)**2 - q_2**2/(2*sigma_p**2)
    
    ffactor = np.sqrt(np.pi/a)
    wfactor = 1/16*1/(2*np.pi*sigma_x*sigma_p)

    exp = np.exp(c+(b**2)/(4*a))
    func = wfactor*ffactor*exp
    return func


def p_marginal_g2():
    a = (t_1+t_2)**2/(2*sigma_x**2*m**2)+1/(2*sigma_p**2)
    b = (t_1+t_2)/(sigma_x**2*m)*x + q_1/(sigma_p**2)
    c = -1/(2*sigma_x**2)*x**2 - q_1**2/(2*sigma_p**2)
    
    ffactor = np.sqrt(np.pi/a)
    wfactor = 1/16*1/(2*np.pi*sigma_x*sigma_p)

    exp = np.exp(c+(b**2)/(4*a))
    func = wfactor*ffactor*exp
    return func


# -----------------------
# plot subplots
# -----------------------
fig, axes = plt.subplots(2, 1, figsize=(8, 8), constrained_layout=True )


# Subplot 1: Transformed Wigner function
im1 = axes[0].imshow(
    Z.T,
    extent=[x[0], x[-1], p[0], p[-1]],
    aspect='auto',
    origin='lower',
    cmap='RdBu_r'
)
axes[0].set_xlabel("x [m]")
axes[0].set_ylabel("p [kg m/s]")
# axes[0].set_title("No decoherence")
plt.colorbar(im1, ax=axes[0], label="W(x,p)")




# Subplot 2: Wigner function after second kick
Z_x_marginal = wf.x_marginal(Z, p)
print(wf.modulation_depth(Z_x_marginal, w=w))
im3 = axes[1].plot(x, Z_x_marginal, linewidth=2, label='numerical marginal')
# im3 = axes[1].plot(wf.smoothing(Z_x_marginal), linewidth=2)
im3 = axes[1].plot(x, p_marginal()+p_marginal_g1()+p_marginal_g2(), linewidth=2, linestyle='--', label='analytic marginal')
im3 = axes[1].plot(x, p_marginal(), linewidth=2, linestyle=':', )
im3 = axes[1].plot(x, p_marginal_g1(), linewidth=2, linestyle=':')
im3 = axes[1].plot(x, p_marginal_g2(), linewidth=2, linestyle=':',)

axes[1].set_xlabel("x [m]")
axes[1].set_ylabel("W(x)")
axes[1].legend()

print(np.average(p_marginal()))
# plt.tight_layout()
plt.show()