import numpy as np
import matplotlib.pyplot as plt

n_steps = 4001
N = n_steps * 2 + 1

ks = np.linspace(-np.pi, np.pi, N, endpoint=False)
psi_p = np.array([1, 0], dtype=complex)
psi_m = np.array([0, 1], dtype=complex)

psi0 = psi_p
# psi0 = 1/np.sqrt(2) * (psi_p + 1j* psi_m) # symmetric initial state
psi_k = np.zeros((N, 2), dtype=complex)

# k-space evolution
for i, k in enumerate(ks):

    M_k = (1/np.sqrt(2)) * np.array([
        [np.exp(-1j * k), np.exp(-1j * k)],
        [1, -1]
    ])

    M_k_n = np.linalg.matrix_power(M_k, n_steps)
    psi_k[i] = M_k_n @ psi0

# inverse FFT
psi_x = np.fft.ifft(psi_k, axis=0)


psi_x_p = psi_x @ psi_p
psi_x_m = psi_x @ psi_m

# -----------------------
# 3 SUBPLOTS
# -----------------------
fig, ax = plt.subplots(1, 3, figsize=(8, 3), sharex=True)

# |+> component
ax[0].plot(np.abs(psi_x_p[:n_steps+2])**2)
ax[0].set_title(r"At $|+\rangle$")
ax[0].set_xlabel("Momentum p in units of q")
ax[0].set_ylabel("Probability")
print(np.sum(np.abs(psi_x_p[:n_steps+2])**2))

# |-> component (your second coin state)
ax[1].plot(np.abs(psi_x_m[:n_steps+2])**2)
ax[1].set_title(r"At $|-\rangle$")
ax[1].set_xlabel("Momentum p in units of q")
print(np.sum(np.abs(psi_x_m[:n_steps+2])**2))

# total probability
ax[2].plot(np.abs(psi_x_p[:n_steps+2])**2 + np.abs(psi_x_m[:n_steps+2])**2)
ax[2].set_title("Combined")
ax[2].set_xlabel("Momentum p in units of q")
print(np.sum(np.abs(psi_x_p[:n_steps+2])**2 + np.abs(psi_x_m[:n_steps+2])**2))

plt.tight_layout()
plt.show()


#############

