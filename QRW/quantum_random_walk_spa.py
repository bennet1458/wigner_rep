import numpy as np
import matplotlib.pyplot as plt

n = 2000

eps = 0.0
p = np.arange(
    np.ceil(n/2*(1-1/np.sqrt(2)) + eps),
    np.floor(n/2*(1+1/np.sqrt(2)) - eps)+1, 1, dtype=int)
print(p)
print(n/2*(1-1/np.sqrt(2)))
print(n/2*(1+1/np.sqrt(2)))

psi_p = np.array([1, 0], dtype=complex)
psi_m = np.array([0, 1], dtype=complex)
psi_0 = psi_m
psi_0 = 1/np.sqrt(2) * (psi_p + 1j* psi_m) # symmetric initial state

k_1 = 2*np.arccos((1-2*p/n)/np.sqrt(1-(1-2*p/n)**2))
k_2 = 2*np.arccos(-(1-2*p/n)/np.sqrt(1-(1-2*p/n)**2))
k_1[k_1 > np.pi] -= 2*np.pi
k_2[k_2 > np.pi] -= 2*np.pi

k_ns = [k_1]#, k_2]


def sin2(k):
    return np.sin(k/2)

def sinsqrt(k):
    return np.sqrt(2-sin2(k)**2)

def exp(k):
    return np.exp(-1j*k)

def cosfac(k):
    return np.emath.sqrt(2*exp(k)*(np.cos(k)+3))

def pi_plus(k):
    factor = 1/cosfac(k)
    a = factor * (exp(k)+cosfac(k)+1)/2
    b = factor * exp(k)
    c = factor * 1
    d = factor * (-exp(k)+cosfac(k)-1)/2
    return  np.array([[a,b],[c,d]])

def pi_minus(k):
    factor = 1/cosfac(k)
    a = factor * (-exp(k)+cosfac(k)-1)/2
    b = factor * -exp(k)
    c = factor * -1
    d = factor * (exp(k)+cosfac(k)+1)/2
    return np.array([[a,b],[c,d]])

def thetadd_plus(k):
    return -sin2(k)/(4*sinsqrt(k)**3)

def thetadd_minus(k):
    return -thetadd_plus(k)

def factor():
    return np.emath.sqrt(1j/(2*np.pi*n))+p*0

def lambda_plus(k):
    return 0.5*((np.exp(-1j*k)-1)/np.sqrt(2)
                + np.emath.sqrt((1-np.exp(-1j*k))**2/2
                                + 4*np.exp(-1j*k)))

def lambda_minus(k):
    return 0.5*((np.exp(-1j*k)-1)/np.sqrt(2)
                - np.emath.sqrt((1-np.exp(-1j*k))**2/2
                                + 4*np.exp(-1j*k)))

# def theta_plus(k):
#     return np.unwrap(np.angle(lambda_plus(k)))

# def theta_minus(k):
#     return np.unwrap(np.angle(lambda_minus(k)))

def theta_plus(k):
    return -k/2 - np.arctan(sin2(k)/sinsqrt(k))

def theta_minus(k):
    return np.pi - k/2 + np.arctan(sin2(k)/sinsqrt(k))

def exp_arg_plus(k):
    return np.exp(1j*(n*theta_plus(k)+k*p))

def exp_arg_minus(k):
    return np.exp(1j*(n*theta_minus(k)+k*p))

psi_n = np.zeros((2, len(p)), dtype=complex)


for k_n in k_ns:
    psi_n_plus = (factor()[np.newaxis, :]
                  * 1/np.emath.sqrt(thetadd_plus(k_n)[np.newaxis, :])
                  * exp_arg_plus(k_n)[np.newaxis, :]
                #   * np.sum(pi_plus(k_n)*psi_0[np.newaxis, :, np.newaxis], axis=1)
                  * pi_plus(k_n)[:, 1, :]
                  )
    psi_n_minus = (factor()[np.newaxis, :]
                   * 1/np.emath.sqrt(thetadd_minus(k_n)[np.newaxis, :])
                   * exp_arg_minus(k_n)[np.newaxis, :]
                #    * np.sum(pi_minus(k_n)*psi_0[np.newaxis, :, np.newaxis], axis=1)
                   * pi_minus(k_n)[:, 1, :]
                   )
    psi_n += psi_n_plus + psi_n_minus



psi_n_p = psi_n[0]
psi_n_m = psi_n[1]


# -----------------------
# 3 SUBPLOTS
# -----------------------
fig, ax = plt.subplots(1, 3, figsize=(8, 3), sharex=True)

# |+> component
ax[0].plot(p, np.abs(psi_n_p)**2)
ax[0].set_title(r"At $|+\rangle$")
ax[0].set_xlabel("Momentum p in units of q")
ax[0].set_ylabel("Probability")
print(np.sum(np.abs(psi_n_p)**2))

# |-> component (your second coin state)
ax[1].plot(p, np.abs(psi_n_m)**2)
ax[1].set_title(r"At $|-\rangle$")
ax[1].set_xlabel("Momentum p in units of q")
print(np.sum(np.abs(psi_n_m)**2))

# total probability
ax[2].plot(p, np.abs(psi_n_p)**2 + np.abs(psi_n_m)**2)
ax[2].set_title("Combined")
ax[2].set_xlabel("Momentum p in units of q")
print(np.sum(np.abs(psi_n_p)**2 + np.abs(psi_n_m)**2))

plt.tight_layout()
plt.show()




