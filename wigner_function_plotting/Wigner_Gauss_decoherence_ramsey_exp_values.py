import numpy as np
import matplotlib.pyplot as plt

# -----------------------
# parameters
# -----------------------
hbar = 6.62607015e-34 / (2*np.pi)  # Planck's constant
eta = .5
t_1 = 1.0e-6   # time after first kick
t_2 = 1e-6 #150.0e-6   # time after second kick
m = 1.4e-19   # mass
omega_0 = 2 * np.pi * 50e3
sigma_x = np.sqrt(hbar/(2*m*omega_0))
sigma_p = np.sqrt(hbar*m*omega_0/2)

q = 10e-23#10e-23   # larger q → clearer interference

### Decoherence parameters ###
diameter = 50e-9
volume = (4/3)*np.pi*(diameter/2)**3
T_i = 315
T_e = 300
def gamma_bb(T):
    return 1.0e41#1.0e47 #1.91e31 * T**(8.38) * np.exp(-0.19 * np.log(T)**2 )
Lambda = 2 * volume * (gamma_bb(T_i) + gamma_bb(T_e))


# grid
x = np.linspace(-2*sigma_x, 2*sigma_x+2*q*(t_1+t_2)/m, 1000)
p = np.linspace(-0.5*q, 2.5*q, 1000)
X = x[:, np.newaxis]
P = p[np.newaxis, :]

# -----------------------
# initial Wigner function
# -----------------------



def W0(x, p):
    return (1/(np.pi*hbar)) * np.exp(
        -x**2/(2*sigma_x**2)
        -p**2/(2*sigma_p**2)
    )

def decoherence(W, t_i):
    # grid spacing
    dx = x[1] - x[0]
    dp = p[1] - p[0]

    Nx = len(x)
    Np = len(p)

    # Fourier frequencies (correct variables!)
    k_x = 2*np.pi * np.fft.fftfreq(Nx, d=dx)
    k_p = 2*np.pi * np.fft.fftfreq(Np, d=dp)

    K_x = k_x[np.newaxis, :]
    K_p = k_p[:, np.newaxis]
    
    factor = np.exp(-hbar**2 * Lambda / 2 *
                    ((2 * t_i**3)/(3*m**2)*K_x**2 + (t_i**2/m)*K_x*K_p + t_i*K_p**2)
                    )
    
    # inverse transform
    # return np.fft.ifft2(np.exp(-hbar**2 * gamma * t/2 * (Q**2 + K**2)) * np.fft.fft2(W)).real
    return np.fft.ifft2(factor * np.fft.fft2(W)).real
# -----------------------
# exact transformed Wigner
# -----------------------
W_k1= (
    eta**2 * W0(X, P - q)
    + eta*(1 - eta) * 2 * np.cos(q/hbar*X) * W0(X, P - q/2)
    + (1 - eta)**2 * W0(X, P)
)

W_t1_k1 = (
    eta**2 * W0(X-t_1*P/m, P - q)
    + eta*(1 - eta) * 2 * np.cos(q/hbar*(X-t_1*P/m)) * W0(X-t_1*P/m, P - q/2)
    + (1 - eta)**2 * W0(X-t_1*P/m, P)
)

W_dc1_t1_k1 = decoherence(W_t1_k1, t_1)

W_k2_t1_k1 = ( eta**4 * W0(X - (P - q)*t_1/m, P - 2*q)

    + eta**3 * (1 - eta) * (
        2 * np.cos(q/hbar * (X - (P - q)*t_1/m)) * W0(X - (P - q)*t_1/m, P - 3*q/2)
        + 2 * np.cos(q*X/hbar) * W0(X - (P - q/2)*t_1/m, P - 3*q/2)
    )

    +
    eta**2 * (1 - eta)**2 * (
        W0(X - (P - q)*t_1/m, P - q)
        + 4 * np.cos(q*X/hbar) * np.cos(q/hbar * (X - (P - q/2)*t_1/m)) * W0(X - (P - q/2)*t_1/m, P - q)
        + W0(X - P*t_1/m, P - q)
    )

    + eta * (1 - eta)**3 * (
        2 * np.cos(q*X/hbar) * W0(X - (P - q/2)*t_1/m, P - q/2)
        + 2 * np.cos(q/hbar * (X - P*t_1/m)) * W0(X - P*t_1/m, P - q/2)
    )

    + (1 - eta)**4 * W0(X - P*t_1/m, P)
    )

W_k2_dc1_t1_k1 = decoherence(W_k2_t1_k1, t_1)

W_t2_k2_t1_k1 = (
    eta**4
    * W0(
        X - (P*(t_1 + t_2) - q*t_1)/m,
        P - 2*q
    )
    + eta**3 * (1 - eta) * (
        2 * np.cos(
            q/hbar * (
                X - (P*(t_1 + t_2) - q*t_1)/m
            )
        )
        * W0(
            X - (P*(t_1 + t_2) - q*t_1)/m,
            P - 3*q/2
        )
        + 2 * np.cos(
            q/hbar * (
                X - P*t_2/m
            )
        )
        * W0(
            X - (P*(t_1 + t_2) - q*t_1/2)/m,
            P - 3*q/2
        )
    )
    + eta**2 * (1 - eta)**2 * (
        W0(
            X - (P*(t_1 + t_2) - q*t_1)/m,
            P - q
        )
        + 4
        * np.cos(
            q/hbar * (
                X - P*t_2/m
            )
        )
        * np.cos(
            q/hbar * (
                X - (P*(t_1 + t_2) - q*t_1/2)/m
            )
        )
        * W0(
            X - (P*(t_1 + t_2) - q*t_1/2)/m,
            P - q
        )
        + W0(
            X - P*(t_1 + t_2)/m,
            P - q
        )
    )
    + eta * (1 - eta)**3 * (
        2 * np.cos(
            q/hbar * (
                X - P*t_2/m
            )
        )
        * W0(
            X - (P*(t_1 + t_2) - q*t_1/2)/m,
            P - q/2
        )
        + 2 * np.cos(
            q/hbar * (
                X - P*(t_1 + t_2)/m
            )
        )
        * W0(
            X - P*(t_1 + t_2)/m,
            P - q/2
        )
    )

    + (1 - eta)**4
    * W0(
        X - P*(t_1 + t_2)/m,
        P
    )
)

W_dc2_t2_k2_dc1_t1_k1 = decoherence(W_t2_k2_t1_k1, t_1+t_2)

# -----------------------
# plot subplots
# -----------------------
fig, axes = plt.subplots(2, 2, figsize=(8, 8))


# Subplot 1: Transformed Wigner function
im1 = axes[0, 0].imshow(
    W_k1,
    extent=[p[0], p[-1], x[0], x[-1]],
    aspect='auto',
    origin='lower',
    cmap='RdBu_r'
)
axes[0, 0].set_xlabel("p")
axes[0, 0].set_ylabel("x")
axes[0, 0].set_title("First kick")
plt.colorbar(im1, ax=axes[0, 0], label="W(x,p)")

# Subplot 2: Wigner function after time evolution
im2 = axes[0, 1].imshow(
    W_dc1_t1_k1,
    extent=[p[0], p[-1], x[0], x[-1]],
    aspect='auto',
    origin='lower',
    cmap='RdBu_r'
)
axes[0, 1].set_xlabel("p")
axes[0, 1].set_ylabel("x")
axes[0, 1].set_title("First time evolution")
plt.colorbar(im2, ax=axes[0, 1], label="W(x,p)")

# Subplot 3: Wigner function after second kick
im3 = axes[1, 0].imshow(
    W_k2_dc1_t1_k1.T,
    extent=[x[0], x[-1], p[0], p[-1]],
    aspect='auto',
    origin='lower',
    cmap='RdBu_r'
)
axes[1, 0].set_xlabel("x")
axes[1, 0].set_ylabel("p")
axes[1, 0].set_title("Second kick")
plt.colorbar(im3, ax=axes[1, 0], label="W(x,p)")


# Subplot 4: Wigner function after second kick and time evolution
im4 = axes[1, 1].imshow(
    W_dc2_t2_k2_dc1_t1_k1.T,
    extent=[x[0], x[-1], p[0], p[-1]],
    aspect='auto',
    origin='lower',
    cmap='RdBu_r'
)
axes[1, 1].set_xlabel("x")
axes[1, 1].set_ylabel("p")
axes[1, 1].set_title("Second time evolution")
plt.colorbar(im4, ax=axes[1, 1], label="W(x,p)")

plt.tight_layout()
plt.show()



#################### marginal plot ####################
# Function to integrate W over p at each x
def integrate_over_p(W_2d):
    """Integrate 2D Wigner function over p axis (sum over p for each x)"""
    dp = p[1] - p[0]  # spacing in p
    return np.sum(W_2d, axis=0) * dp


W_t2_k2_t1_k1_marginal = integrate_over_p(W_t2_k2_t1_k1)
W_dc2_t2_k2_dc1_t1_k1_marginal = integrate_over_p(W_dc2_t2_k2_dc1_t1_k1)
plt.figure(figsize=(6, 4))
plt.plot(x[450:550], W_t2_k2_t1_k1_marginal[450:550], linewidth=2)
plt.plot(x[450:550], W_dc2_t2_k2_dc1_t1_k1_marginal[450:550], linewidth=2, linestyle='--')
plt.xlabel("x")
plt.ylabel("Marginal")
plt.title("Marginal of W(x,p) after second time evolution")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()