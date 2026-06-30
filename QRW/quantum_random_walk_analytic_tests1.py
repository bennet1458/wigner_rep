import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp
# from scipy.integrate import trapz
from math import comb

n = 500
# N = n * 2 + 1
N =  512

ks = np.linspace(-np.pi, np.pi, N, endpoint=False)
psi_p = np.array([1, 0], dtype=complex)
psi_m = np.array([0, 1], dtype=complex)

psi0 = psi_m
# psi0 = 1/np.sqrt(2) * (psi_p + 1j* psi_m) # symmetric initial state



def matrix_power():
    
    psi_k = np.zeros((N, 2), dtype=complex)
    for i, k in enumerate(ks):
        M_k = (1/np.sqrt(2)) * np.array([
            [np.exp(-1j * k), np.exp(-1j * k)],
            [1, -1]
        ])

        M_k_n = np.linalg.matrix_power(M_k, n + 1)
        psi_k[i] = M_k_n @ psi0
    return psi_k

def fib_poly(n_, l):
    term = np.zeros(n_//2+1, dtype=complex)
    for k_ in range(n_//2+1):
        bin_factor = comb(n_-k_, k_) #sp.binom(n_-k_, k_)
        term[k_] = bin_factor * l**k_ * ((l-1)/(np.sqrt(2)))**(n_-2*k_)
    return np.sum(term)


def series_expansion():
    psi_k = np.zeros((N, 2), dtype=complex)
    for i, k in enumerate(ks):
        l = np.exp(-1j * k)
        fib_n = fib_poly(n, l)
        fib_n_plus_1 = fib_poly(n+1, l)
        M_k_n = 1/np.sqrt(2) * np.array([
            [np.sqrt(2) * fib_n_plus_1 + fib_n,
                l * fib_n],
            [fib_n, 
             np.sqrt(2) * fib_n_plus_1 - l * fib_n]
        ])
        psi_k[i] = M_k_n @ psi0
    return psi_k

def integral_approximation(n_, l):
    t = np.linspace(0.000001, 0.499999, 10000)
    x = (l-1)/(np.sqrt(2))
    y = l
    S = np.sqrt(n_/(2 * np.pi)) * np.sqrt((1 - t)/(t*(1-2*t)))
    h = np.log(1-t) * (1-t) - np.log(t) * t - np.log(1-2*t) * (1-2*t) 
    
    with np.errstate(divide='ignore', invalid='ignore'):
        Phi = h + np.emath.log(y) * t + np.emath.log(x) * (1-2*t)
        integrand = np.exp(n_ * Phi) * S
    
    # Clean up any NaN or infinite values
    integrand = np.nan_to_num(integrand, nan=0.0, posinf=0.0, neginf=0.0)
    return np.trapezoid(integrand, t)




# def stationary_phase_approx(n_, k):
#     l = np.exp(-1j * k)
#     x = (l-1)/(np.sqrt(2))
#     y = l
    
#     with np.errstate(divide='ignore', invalid='ignore'):
#         t_0 = 0.5 * (1 - np.sqrt(np.sin(k/2)**2/(np.sin(k/2)**2+2)))

#         if abs(1-2*t_0) < 1e-10:
#             return 0.0j

#         h_0 = np.log(1-t_0) * (1-t_0) - np.log(t_0) * t_0 - np.log(1-2*t_0) * (1-2*t_0)
#         Phi_0 = h_0 + np.emath.log(y) * t_0 + np.emath.log(x) * (1-2*t_0)
#         # print(Phi_0)
#         Phi_d2_0 = 1/(1-t_0) - 1/t_0 - 4/(1-2*t_0)
#         S_0 = np.sqrt(n_/(2 * np.pi)) * np.sqrt((1 - t_0)/(t_0*(1-2*t_0)))
#         fac = np.sqrt(2 * np.pi * n_ /np.abs(Phi_d2_0))
#     return S_0 * fac * np.exp(n_*Phi_0-1j*np.pi/4)

def stationary_phase_approx(n_, k):
    l = np.exp(-1j * k)
    x = (l-1)/(np.sqrt(2))
    y = l
    
    with np.errstate(divide='ignore', invalid='ignore'):
        t_0p = 0.5 * (1 + np.sqrt(np.sin(k/2)**2/(np.sin(k/2)**2+2)))
        t_0m = 0.5 * (1 - np.sqrt(np.sin(k/2)**2/(np.sin(k/2)**2+2)))

        if abs(1-2*t_0m) < 1e-10:
            return 0.0j
        
        if abs(1-2*t_0p) < 1e-10:
            return 0.0j

        h_0p = np.emath.log(1-t_0p) * (1-t_0p) - np.emath.log(t_0p) * t_0p - np.emath.log(1-2*t_0p) * (1-2*t_0p)
        
        h_0m = np.emath.log(1-t_0m) * (1-t_0m) - np.emath.log(t_0m) * t_0m - np.emath.log(1-2*t_0m) * (1-2*t_0m)
        
        Phi_0p = h_0p + np.emath.log(y) * t_0p + np.emath.log(x) * (1-2*t_0p)
        Phi_0m = h_0m + np.emath.log(y) * t_0m + np.emath.log(x) * (1-2*t_0m)
        # print(Phi_0p)
        # print(Phi_0)
        # Phi_d2_0 = 1/(1-t_0) - 1/t_0 - 4/(1-2*t_0)
        # S_0 = np.sqrt(n_/(2 * np.pi)) * np.sqrt((1 - t_0)/(t_0*(1-2*t_0)))
        # fac = np.sqrt(2 * np.pi * n_ /np.abs(Phi_d2_0))
        result =  1e146 * np.exp(n_*Phi_0p+1j*np.pi/4) + np.exp(n_*Phi_0m+1j*np.pi/4) 
        # print(result)
    return result


def integral_expansion():
    psi_k = np.zeros((N, 2), dtype=complex)
    for i, k in enumerate(ks):
        l = np.exp(-1j * k)
        fib_n = stationary_phase_approx(n, k)
        fib_n_plus_1 = stationary_phase_approx(n+1, k)
        M_k_n = 1/np.sqrt(2) * np.array([
            [np.sqrt(2) * fib_n_plus_1 + fib_n,
                l * fib_n],
            [fib_n, 
             np.sqrt(2) * fib_n_plus_1 - l * fib_n]
        ])
        psi_k[i] = M_k_n @ psi0
    return psi_k

psi_k_mp = matrix_power()
psi_k_se = series_expansion()
# psi_k_ie = integral_expansion()

# inverse FFT
psi_x_mp = np.fft.ifft(psi_k_mp, axis=0)
psi_x_se = np.fft.ifft(psi_k_se, axis=0)
# psi_x_ie = np.fft.ifft(psi_k_ie, axis=0)


psi_x_mp_p = psi_x_mp @ psi_p
psi_x_se_p = psi_x_se @ psi_p
# psi_x_ie_p = psi_x_ie @ psi_p

psi_x_mp_m = psi_x_mp @ psi_m
psi_x_se_m = psi_x_se @ psi_m
# psi_x_ie_m = psi_x_ie @ psi_m

# -----------------------
# 3 SUBPLOTS
# -----------------------
fig, ax = plt.subplots(1, 3, figsize=(8, 3), sharex=True)

# |+> component
ax[0].plot(np.abs(psi_x_mp_p[:n+2])**2)
ax[0].plot(np.abs(psi_x_se_p[:n+2])**2, "--")
# ax[0].plot(np.abs(psi_x_ie_p[:n+2])**2, ":")
ax[0].set_title(r"At $|+\rangle$")
ax[0].set_xlabel("Momentum p in units of q")
ax[0].set_ylabel("Probability")

# |-> component (your second coin state)
ax[1].plot(np.abs(psi_x_mp_m[:n+2])**2)
ax[1].plot(np.abs(psi_x_se_m[:n+2])**2, "--")
# ax[1].plot(np.abs(psi_x_ie_m[:n+2])**2, ":")
ax[1].set_title(r"At $|-\rangle$")
ax[1].set_xlabel("Momentum p in units of q")

# total probability
ax[2].plot(np.abs(psi_x_mp_p[:n+2])**2 + np.abs(psi_x_mp_m[:n+2])**2)
ax[2].plot(np.abs(psi_x_se_p[:n+2])**2 + np.abs(psi_x_se_m[:n+2])**2, "--")
# ax[2].plot(np.abs((psi_x_ie_p[:n+2])**2 + np.abs(psi_x_ie_m[:n+2])**2)/np.sum(np.abs((psi_x_ie_p[:n+2])**2 + np.abs(psi_x_ie_m[:n+2])**2)), ":")
# ax[2].plot(np.abs((psi_x_ie_p[:n+2])**2 + np.abs(psi_x_ie_m[:n+2])**2), ":")
ax[2].set_title("Combined")
ax[2].set_xlabel("Momentum p in units of q")

plt.tight_layout()
plt.show()


#############
