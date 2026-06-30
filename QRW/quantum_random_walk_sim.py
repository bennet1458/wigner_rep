import numpy as np
import matplotlib.pyplot as plt

# Parameters
n_steps = 2000


# Initial state: coin = |0>, position = 0
psi0 = np.zeros((2, n_steps+1), dtype=complex)
psi0[0, 0] = 1.0

# Hadamard coin
H = (1/np.sqrt(2)) * np.array([[1, 1],
                               [1, -1]])

def step(psi):
    # Apply coin
    psi = H @ psi

    # Apply shift
    new_psi = np.zeros_like(psi)

    for i in range(0, n_steps - 1):
        # coin 0 → stay
        new_psi[0, i] += psi[0, i]
        # coin 1 → move right
        new_psi[1, i + 1] += psi[1, i]

    return new_psi

# Run walk
psi = psi0
for _ in range(n_steps):
    psi = step(psi)


# Probability distribution
prob = np.abs(psi)**2


##################
# Plot
xs = np.arange(0, n_steps + 1)
# plt.plot(xs, prob[0])
plt.plot(xs, prob[1])
# plt.plot(xs, prob[0] + prob[1], color='black', linestyle='dashed', label='Total')
plt.title("Quantum Walk (steps: 0 or +1)")
plt.xlabel("Position")
plt.ylabel("Probability")
plt.show()

# # Probability distribution
# prob = np.sum(np.abs(psi)**2, axis=0)

# # Plot
# xs = np.arange(0, n_steps + 1)
# plt.plot(xs, prob)
# plt.title("Quantum Walk (steps: 0 or +1)")
# plt.xlabel("Position")
# plt.ylabel("Probability")
# plt.show()