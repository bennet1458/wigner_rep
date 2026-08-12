import numpy as np
import matplotlib.pyplot as plt

# -----------------------
# parameters
hbar = 6.62607015e-34 / (2*np.pi)  # Planck's constant
t_1 = 1.0e-6   # time after first kick
t_2 = 15.0e-6 #150.0e-6   # time after second kick
q = 10e-23#10e-23   # larger q → clearer interference
q_1 = q
q_2 = q
m = 1.4e-19   # mass
omega_0 = 2 * np.pi * 50e3
sigma_x = np.sqrt(hbar/(2*m*omega_0))
sigma_p = np.sqrt(hbar*m*omega_0/2)


# grid
w_max = 2*q/m
w_min = -2*q/m
s_min = -q*t_2/m-(t_1+t_2)*w_max
s_max = +q*t_2/m-(t_1+t_2)*w_min

w = np.linspace(w_min, w_max, 400)
s = np.linspace(s_min, s_max, 400)
W = w[:, np.newaxis]
S = s[np.newaxis, :]

# -----------------------
# initial Wigner function
# -----------------------


def chi0(w, s):
    return np.exp(
        -m**2*w**2/(2*hbar**2)*sigma_x**2
        -s**2/(2*hbar**2)*sigma_p**2
    )


# ---------- useful phases ----------
Phi0 = (
    (q_1 + q_2)*s
    - (q_1*t_1 + q_1*t_2 + q_2*t_2)*W
)

Phi_minus = Phi0 + q_1*q_2*t_1/m
Phi_plus  = Phi0 - q_1*q_2*t_1/m

# ---------- second kick ----------
chi_t2_k2_t1_k1 = 1/16*(
    (
        1
        + np.exp(-1j*q_2*(S-t_2*W)/hbar)
        + np.exp(-1j*q_1*(S-(t_1+t_2)*W)/hbar)
        + np.exp(-1j*Phi0/hbar)
    )
    * chi0(W, S-(t_1+t_2)*W)

    + (
        np.exp(-1j*q_1*(S-(t_1+t_2)*W)/hbar)
        + np.exp(-1j*Phi0/hbar)
    )
    * (
        chi0(W-q_1/m, S-(t_1+t_2)*W)
        + chi0(W+q_1/m, S-(t_1+t_2)*W)
    )

    + (
        np.exp(-1j*q_2*(S-t_2*W)/hbar)
        + np.exp(-1j*Phi_minus/hbar)
    )
    * chi0(
        W-q_2/m,
        S + q_2*t_1/m - (t_1+t_2)*W
    )

    + (
        np.exp(-1j*q_2*(S-t_2*W)/hbar)
        + np.exp(-1j*Phi_plus/hbar)
    )
    * chi0(
        W+q_2/m,
        S - q_2*t_1/m - (t_1+t_2)*W
    )

    + np.exp(-1j*Phi_minus/hbar)
    * (
        chi0(
            W-(q_1+q_2)/m,
            S + q_2*t_1/m - (t_1+t_2)*W
        )
        +
        chi0(
            W+(q_1-q_2)/m,
            S + q_2*t_1/m - (t_1+t_2)*W
        )
    )

    + np.exp(-1j*Phi_plus/hbar)
    * (
        chi0(
            W+(q_1+q_2)/m,
            S - q_2*t_1/m - (t_1+t_2)*W   
        )
        +
        chi0(
            W+(q_2-q_1)/m,
            S - q_2*t_1/m - (t_1+t_2)*W
        )
    )
)

figure = plt.figure(figsize=(8, 6))
plt.imshow(
    np.absolute(chi_t2_k2_t1_k1).T,
    extent=[ w[0], w[-1], s[0], s[-1]],
    aspect='auto',
    origin='lower',
    cmap='RdBu_r'
)
plt.xlabel("w") 
plt.ylabel("s")
plt.title("First kick")
plt.colorbar(label="$\chi$(w,s)")
plt.show()

