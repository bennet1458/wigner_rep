import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Parameters
# ============================================================

N = 512
ks = np.linspace(-np.pi, np.pi, N, endpoint=False)

# ============================================================
# Arrays for diagnostics
# ============================================================

t_plus_vals  = np.zeros(N)
t_minus_vals = np.zeros(N)

Phi_plus_vals  = np.zeros(N, dtype=complex)
Phi_minus_vals = np.zeros(N, dtype=complex)

alpha_vals = np.zeros(N, dtype=complex)
beta_vals  = np.zeros(N, dtype=complex)

log_alpha_vals = np.zeros(N, dtype=complex)
log_beta_vals  = np.zeros(N, dtype=complex)

# ============================================================
# Helper: z log(z) with correct limit at z=0
# ============================================================

def xlogx(z):
    z = complex(z)
    if abs(z) < 1e-14:
        return 0.0 + 0.0j
    return z * np.emath.log(z)

# ============================================================
# Main loop
# ============================================================

for i, k in enumerate(ks):

    # Quantum walk variables
    l = np.exp(-1j * k)
    x = (l - 1) / np.sqrt(2)
    y = l

    # Saddle points
    s = np.sin(k / 2.0) ** 2
    root = np.sqrt(s / (s + 2.0))

    t_plus  = 0.5 * (1 + root)
    t_minus = 0.5 * (1 - root)

    t_plus_vals[i]  = t_plus
    t_minus_vals[i] = t_minus

    # --------------------------------------------------------
    # Entropy-like part h(t)
    # h = (1-t)log(1-t) - t log t - (1-2t)log(1-2t)
    # using xlogx avoids 0*log(0)=nan
    # --------------------------------------------------------

    h_plus = (
        xlogx(1 - t_plus)
        - xlogx(t_plus)
        - xlogx(1 - 2 * t_plus)
    )

    h_minus = (
        xlogx(1 - t_minus)
        - xlogx(t_minus)
        - xlogx(1 - 2 * t_minus)
    )

    # --------------------------------------------------------
    # Remaining logarithms
    # --------------------------------------------------------

    if abs(x) < 1e-14:
        logx = 0.0 + 0.0j
    else:
        logx = np.emath.log(x)

    logy = np.emath.log(y)

    # --------------------------------------------------------
    # Saddle exponents
    # --------------------------------------------------------

    Phi_plus = h_plus + t_plus * logy + (1 - 2 * t_plus) * logx
    Phi_minus = h_minus + t_minus * logy + (1 - 2 * t_minus) * logx

    Phi_plus_vals[i] = Phi_plus
    Phi_minus_vals[i] = Phi_minus

    # --------------------------------------------------------
    # Exact Binet roots
    # --------------------------------------------------------

    disc = np.emath.sqrt(x * x + 4 * y)

    alpha = (x + disc) / 2
    beta = (x - disc) / 2

    alpha_vals[i] = alpha
    beta_vals[i] = beta

    log_alpha_vals[i] = np.emath.log(alpha)
    log_beta_vals[i] = np.emath.log(beta)

# ============================================================
# Plot saddle locations
# ============================================================

plt.figure(figsize=(8, 4))
plt.plot(ks, t_plus_vals, label=r"$t_+$")
plt.plot(ks, t_minus_vals, label=r"$t_-$")
plt.xlabel(r"$k$")
plt.ylabel(r"$t$")
plt.title("Stationary points")
plt.grid(True)
plt.legend()
plt.tight_layout()

# ============================================================
# Compare Re(Phi) with Re(log(alpha))
# ============================================================

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(ks, np.real(Phi_plus_vals), label=r"Re($\Phi_+$)")
plt.plot(ks, np.real(Phi_minus_vals), label=r"Re($\Phi_-$)")
plt.plot(ks, np.real(log_alpha_vals), "--", label=r"Re(log $\alpha$)")
plt.plot(ks, np.real(log_beta_vals), "--", label=r"Re(log $\beta$)")
plt.xlabel(r"$k$")
plt.ylabel("Real part")
plt.title("Real parts")
plt.grid(True)
plt.legend()

# ============================================================
# Compare Im(Phi) with Im(log(alpha))
# ============================================================

plt.subplot(1, 2, 2)
plt.plot(ks, np.imag(Phi_plus_vals), label=r"Im($\Phi_+$)")
plt.plot(ks, np.imag(Phi_minus_vals), label=r"Im($\Phi_-$)")
plt.plot(ks, np.imag(log_alpha_vals), "--", label=r"Im(log $\alpha$)")
plt.plot(ks, np.imag(log_beta_vals), "--", label=r"Im(log $\beta$)")
plt.xlabel(r"$k$")
plt.ylabel("Imaginary part")
plt.title("Imaginary parts")
plt.grid(True)
plt.legend()

plt.tight_layout()

# ============================================================
# Compare magnitudes
# ============================================================

plt.figure(figsize=(8, 4))
plt.plot(ks, np.abs(np.exp(Phi_plus_vals)),
         label=r"$|e^{\Phi_+}|$")
plt.plot(ks, np.abs(np.exp(Phi_minus_vals)),
         label=r"$|e^{\Phi_-}|$")
plt.plot(ks, np.abs(alpha_vals), "--",
         label=r"$|\alpha|$")
plt.plot(ks, np.abs(beta_vals), "--",
         label=r"$|\beta|$")
plt.xlabel(r"$k$")
plt.ylabel("Magnitude")
plt.title("Magnitude comparison")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.show()

# ============================================================
# Print sample values
# ============================================================

print("\n================ SAMPLE VALUES ================\n")

for ktest in [0.3, 1.0, 2.0]:

    l = np.exp(-1j * ktest)
    x = (l - 1) / np.sqrt(2)
    y = l

    s = np.sin(ktest / 2.0) ** 2
    root = np.sqrt(s / (s + 2.0))

    tp = 0.5 * (1 + root)
    tm = 0.5 * (1 - root)

    hp = (
        xlogx(1 - tp)
        - xlogx(tp)
        - xlogx(1 - 2 * tp)
    )

    hm = (
        xlogx(1 - tm)
        - xlogx(tm)
        - xlogx(1 - 2 * tm)
    )

    logx = np.emath.log(x)
    logy = np.emath.log(y)

    Phi_p = hp + tp * logy + (1 - 2 * tp) * logx
    Phi_m = hm + tm * logy + (1 - 2 * tm) * logx

    disc = np.emath.sqrt(x * x + 4 * y)
    alpha = (x + disc) / 2
    beta = (x - disc) / 2

    print(f"k = {ktest:.2f}")
    print("-----------------------------------------------")
    print(f"t+           = {tp}")
    print(f"t-           = {tm}")
    print()
    print(f"Phi+         = {Phi_p}")
    print(f"log(alpha)   = {np.emath.log(alpha)}")
    print(f"Difference   = {Phi_p - np.emath.log(alpha)}")
    print()
    print(f"Phi-         = {Phi_m}")
    print(f"log(beta)    = {np.emath.log(beta)}")
    print(f"Difference   = {Phi_m - np.emath.log(beta)}")
    print("\n")









