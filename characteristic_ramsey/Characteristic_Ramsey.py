import numpy as np
import matplotlib.pyplot as plt

# -----------------------
# parameters
# -----------------------
hbar = 1.0
eta = .5
t = 1   # time after kick
m = 1.0   # mass
Lambda = 0.2
t_1 = t
t_2 = t
q = 5   # larger q → clearer interference
q_1 = q
q_2 = q
# grid
w = np.linspace(-2.5*q, 2.5*q, 400)
s = np.linspace(-4*q, 4*q, 400)
W = w[:, np.newaxis]
S = s[np.newaxis, :]

# -----------------------
# initial Wigner function
# -----------------------
sigma_x = 1
sigma_p = hbar/(2*sigma_x)


def chi0(w, s):
    return np.exp(
        -m**2*w**2/(2*hbar**2)*sigma_x**2
        -s**2/(2*hbar**2)*sigma_p**2
    )

# -----------------------
# exact transformed Wigner function
# -----------------------
chi_k1= 1/2*(
    (1 + np.exp(-1j*q_1*S/hbar))*chi0(W, S)
    + np.exp(-1j*q_1*S/hbar)*(chi0(W - q_1/m , S) + chi0(W + q_1/m, S))
)

chi_t1_k1 = 1/2*(
    (1 + np.exp(-1j*q_1*(S - W*t_1)/hbar))
    * chi0(W, S - W*t_1)

    + np.exp(-1j*q_1*(S - W*t_1)/hbar)
    * (
        chi0(W - q_1/m, S - W*t_1)
        + chi0(W + q_1/m, S - W*t_1)
    )
)

############################################################################################################

t_2 = 0
# ---------- useful phases ----------
Phi0 = (
    (q_1 + q_2)*s
    - (q_1*t_1 + q_1*t_2 + q_2*t_2)*W
)

Phi_minus = Phi0 + q_1*q_2*t_1/m
Phi_plus  = Phi0 - q_1*q_2*t_1/m

# ---------- second kick ----------
chi_k2_t1_k1 = 1/4*(
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
############################################################################################################
t_2 = t
# ---------- useful phases ----------
Phi0 = (
    (q_1 + q_2)*s
    - (q_1*t_1 + q_1*t_2 + q_2*t_2)*W
)

Phi_minus = Phi0 + q_1*q_2*t_1/m
Phi_plus  = Phi0 - q_1*q_2*t_1/m

# ---------- second kick ----------
chi_t2_k2_t1_k1 = 1/4*(
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

# -----------------------
# plot subplots
# -----------------------
fig, axes = plt.subplots(2, 2, figsize=(8, 8))


# Subplot 1: Transformed Wigner function
im1 = axes[0, 0].imshow(
    np.absolute(chi_k1).T**2,
    extent=[ w[0], w[-1], s[0], s[-1]],
    aspect='auto',
    origin='lower',
    cmap='RdBu_r'
)
axes[0, 0].set_xlabel("w")
axes[0, 0].set_ylabel("s")
axes[0, 0].set_title("First kick")
plt.colorbar(im1, ax=axes[0, 0], label="$\chi$(w,s)")

# Subplot 2: Wigner function after time evolution
im2 = axes[0, 1].imshow(
    np.absolute(chi_t1_k1).T**2,
    extent=[ w[0], w[-1], s[0], s[-1]],
    aspect='auto',
    origin='lower',
    cmap='RdBu_r'
)
axes[0, 1].set_xlabel("w")
axes[0, 1].set_ylabel("s")
axes[0, 1].set_title("First time evolution")
plt.colorbar(im2, ax=axes[0, 1], label="$\chi$(w,s)")

# Subplot 3: Wigner function after second kick
im3 = axes[1, 0].imshow(
    np.absolute(chi_k2_t1_k1).T**2,
    extent=[ w[0], w[-1], s[0], s[-1]],
    aspect='auto',
    origin='lower',
    cmap='RdBu_r'
)
axes[1, 0].set_xlabel("w")
axes[1, 0].set_ylabel("s")
axes[1, 0].set_title("Second kick")
plt.colorbar(im3, ax=axes[1, 0], label="$\chi$(w,s)")


# Subplot 4: Wigner function after second kick and time evolution
im4 = axes[1, 1].imshow(
    np.absolute(chi_t2_k2_t1_k1).T**2,
    extent=[ w[0], w[-1], s[0], s[-1]],
    aspect='auto',
    origin='lower',
    cmap='RdBu_r'
)
axes[1, 1].set_xlabel("w")
axes[1, 1].set_ylabel("s")
axes[1, 1].set_title("Second time evolution")
plt.colorbar(im4, ax=axes[1, 1], label="$\chi$(w,s)")

plt.tight_layout()
plt.show()
