import numpy as np
import matplotlib.pyplot as plt
import wigner_functions as wf

# -----------------------
# parameters
# -----------------------
hbar = 6.62607015e-34 / (2 * np.pi)  # reduced Planck constant
eta = 0.5
t_1 = 1.0e-6
m = 1.4e-19
omega_0 = 2 * np.pi * 50e3

sigma_x = np.sqrt(hbar / (2 * m * omega_0))
sigma_p = np.sqrt(hbar * m * omega_0 / 2)

tau_o = 2 * m * sigma_x**2 / hbar

res = 900





# -----------------------
# t2 scan
# -----------------------

t_res = 10
t_end = 500e-6
t_start = 50e-6

t_2_values = np.linspace(t_start, t_end, t_res)
t_2_fix = 150e-6

# q scan parameters
q_res = 8
q_start = 5e-23
q_fix = 10e-23
q_end = 20e-23
q_values = np.linspace(q_start, q_end, q_res)

k_R_exp_t2 = np.zeros(t_res)
k_R_theo_t2 = np.zeros(t_res)

k_R_exp_q = np.zeros(q_res)
k_R_theo_q = np.zeros(q_res)



# -----------------------
# calculate fringe wave number for t2 scan
# -----------------------

for j, t_2 in enumerate(t_2_values):
    x = np.linspace(-4*sigma_x+(q_fix-2*sigma_p)*(t_2)/m, 4*sigma_x+(q_fix+2*sigma_p)*(t_1+t_2)/m, res)
    p = np.linspace(-4*sigma_p+q_fix, 4*sigma_p+q_fix, res)
    X = x[:, np.newaxis]
    P = p[np.newaxis, :]
    # -----------------------
    # FFT wave-number axis
    # -----------------------

    dx = x[1] - x[0]

    # physical wave numbers (rad/m)
    k_values = 2 * np.pi * np.fft.rfftfreq(len(x), d=dx)
    # -----------------------
    # initial state
    # -----------------------

    W = wf.W0(sigma_x, sigma_p, hbar)

    W = wf.kick(W, q_fix, eta, hbar)

    W = wf.time_evolution(W, t_1, m)

    W = wf.kick(W, q_fix, eta, hbar)
    # evolve to current t2
    W_t2 = wf.time_evolution(W, t_2, m)

    # evaluate Wigner function
    Z = W_t2(X, P)

    # x marginal
    Z_x_marginal = wf.x_marginal(Z, p)

    # FFT
    fft = np.fft.rfft(Z_x_marginal)

    # remove DC component
    # fft[0] = 0
    fft[:5] = 0  # also remove low-k components to reduce noise

    # dominant fringe wave number
    peak_idx = np.argmax(np.abs(fft))
    k_R_exp_t2[j] = k_values[peak_idx]

    # theoretical fringe spacing

    fringe_period = (
        hbar
        * 2
        * np.pi
        / (2 * q_fix * t_1)
        * ((t_1 + t_2) ** 2 + tau_o**2)
        / (t_1 + t_2)
    )

    # theoretical wave number
    
    # k_R_theo_t2[j]  = ((
    #         (t_1 + t_2) * (q_fix * t_1 + q_fix * (t_1 + t_2)) * sigma_p**2
    #         + m**2 * q_fix * sigma_x**2
    #     ) / (
    #         hbar * (t_1 + t_2)**2 * sigma_p**2
    #         + hbar * m**2 * sigma_x**2
    #     ))-q_fix/hbar
    
    k_R_theo_t2[j]  = ((
        q_fix * (
        (t_1 + t_2) * (t_1 - (t_1 + t_2)) * sigma_p**2
        + m**2 * sigma_x**2
         )
        ) / (
        hbar * ((t_1 + t_2)**2 * sigma_p**2 + m**2 * sigma_x**2)
        ))+q_fix/hbar
# -----------------------
# calculate fringe wave number for q scan
# -----------------------

for i, q in enumerate(q_values):
    x = np.linspace(-4*sigma_x+(q-2*sigma_p)*(t_2_fix)/m, 4*sigma_x+(q+2*sigma_p)*(t_1+t_2_fix)/m, res)
    p = np.linspace(-4*sigma_p+q, 4*sigma_p+q, res)
    X = x[:, np.newaxis]
    P = p[np.newaxis, :]

    dx = x[1] - x[0]
    k_values = 2 * np.pi * np.fft.rfftfreq(len(x), d=dx)
    # -----------------------
    # initial state
    # -----------------------

    W = wf.W0(sigma_x, sigma_p, hbar)

    W = wf.kick(W, q, eta, hbar)

    W = wf.time_evolution(W, t_1, m)

    W = wf.kick(W, q, eta, hbar)

    # evolve to fixed t_2
    W_t2 = wf.time_evolution(W, t_2_fix, m)

    # evaluate Wigner function
    Z = W_t2(X, P)

    # x marginal
    Z_x_marginal = wf.x_marginal(Z, p)

    # FFT
    fft = np.fft.rfft(Z_x_marginal)
    fft[:5] = 0

    # dominant fringe wave number
    peak_idx = np.argmax(np.abs(fft))
    k_R_exp_q[i] = k_values[peak_idx]

    # theoretical fringe spacing
    fringe_period = (
        hbar
        * 2
        * np.pi
        / (2 * q * t_1)
        * ((t_1 + t_2_fix) ** 2 + tau_o**2)
        / (t_1 + t_2_fix)
    )

    # theoretical wave number
    k_R_theo_q[i] = 2 * np.pi / fringe_period

# -----------------------
# plot
# -----------------------

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: t2 scan
ax1.plot(
    t_2_values * 1e6,
    k_R_exp_t2,
    marker="o",
    label="Numerical FFT Peak",
)

ax1.plot(
    t_2_values * 1e6,
    k_R_theo_t2,
    "--",
    linewidth=2,
    label="Theory",
)

ax1.set_xlabel(r"Free Evolution Time $t_2$ ($\mu$s)")
ax1.set_ylabel(r"Fringe Wave Number $k_R$ (rad/m)")
ax1.set_title(r"Ramsey Fringe Wave Number vs. $t_2$ (fixed $q$ = {:.0e} [kg·m/s])".format(q_fix))
ax1.legend()
ax1.grid(True)

# Plot 2: q scan
ax2.plot(
    q_values * 1e23,
    k_R_exp_q,
    marker="o",
    label="Numerical FFT Peak",
)

ax2.plot(
    q_values * 1e23,
    k_R_theo_q,
    "--",
    linewidth=2,
    label="Theory",
)

ax2.set_xlabel(r"Kick Strength $q$ ($\times 10^{-23}$ kg·m/s)")
ax2.set_ylabel(r"Fringe Wave Number $k_R$ (rad/m)")
ax2.set_title(r"Ramsey Fringe Wave Number vs. $q$ (fixed $t_2$ = {:.1f} $\mu$s)".format(t_2_fix * 1e6))
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()