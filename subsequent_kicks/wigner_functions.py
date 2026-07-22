import numpy as np

def W0(sigma_x, sigma_p, hbar):

    def W(x, p):
        return (
            1/(np.pi*hbar)
            * np.exp(
                -x**2/(2*sigma_x**2)
                -p**2/(2*sigma_p**2)
            )
        )
    return W

def kick(W, q, phi, hbar):
    def W_kicked(x, p):
        return 1/2*(
            W(x, p - q)
            - 2 * np.cos(q/hbar * x + phi) * W(x, p - q/2)
            + W(x, p)
        )
    return W_kicked

def time_evolution(W, t, m):
    def W_evolved(x, p):
        return W(x - p*t/m, p)
    return W_evolved

def harmonic_evolution(W, t, m, omega):
    c = np.cos(omega*t)
    s = np.sin(omega*t)

    def W_new(x, p):
        return W(
            c*x - s*p/(m*omega),
            c*p + s*m*omega*x
        )

    return W_new

def transform(W):
    def transformed(x, p):
        return np.transpose(W(x, p))
    return transformed


def decoherence(W, x, p, t_i, Lambda, m, hbar):
    # grid spacing
    dx = x[1] - x[0]
    dp = p[1] - p[0]

    Nx = len(x)
    Np = len(p)

    # Fourier frequencies
    k_x = 2*np.pi * np.fft.fftfreq(Nx, d=dx)
    k_p = 2*np.pi * np.fft.fftfreq(Np, d=dp)

    K_x = k_x[np.newaxis, :]
    K_p = k_p[:, np.newaxis]

    factor = np.exp(
        -hbar**2 * Lambda / 2 *
        (
            (2*t_i**3)/(3*m**2) * K_x**2
            + (t_i**2/m) * K_x * K_p
            + t_i * K_p**2
        )
    )

    def W_decohered(x, p):
        # apply decoherence kernel in Fourier space
        values = W(x, p)

        return np.fft.ifft2(
            factor * np.fft.fft2(values)
        ).real

    return W_decohered


###### marginals ##########


# def x_marginal(W, p):
#     """Integrate 2D Wigner function over p axis (sum over p for each x)"""
#     dp = p[1] - p[0]  # spacing in p
#     return np.sum(W, axis=0) * dp
def x_marginal(W, p):
    """Integrate 2D Wigner function over p axis (sum over p for each x)"""
    dp = p[1] - p[0]  # spacing in p
    return np.sum(W, axis=1) * dp


# def smoothing(y, w=3):
#     c = np.cumsum(np.insert(y, 0, 0))
#     return (c[w:] - c[:-w]) / w

def smoothing(y, w=3):
    if w == 0:
        return y
    ys = y.copy()

    for _ in range(w):
        ys[1:-1] = (
            ys[:-2] + 2*ys[1:-1] + ys[2:]
        ) * 0.25

    return ys

def modulation_depth(y, w=3): #w should be low for larger q
    y = np.asarray(y)

    # smoothing 
    y = smoothing(y, w)

    # dominant peak
    p = np.argmax(y)
    peak = y[p]

    # slope
    dy = np.diff(y)

    # minima locations:
    # slope changes from negative to positive
    minima = np.where((dy[:-1] < 0) & (dy[1:] > 0))[0] + 1

    # nearest valleys
    left = minima[minima < p]
    right = minima[minima > p]

    if len(left) == 0 or len(right) == 0:
        return 0.0
    
    len_y = len(y)

    # if left[-1] < 0.25*len_y or right[0] > 0.75*len_y:
    #     return 0.0

    lv = y[left[-1]]
    rv = y[right[0]]

    valley = 0.5 * (lv + rv)

    denom = peak + valley
    if denom == 0:
        return 0.0

    M = (peak - valley) / denom

    # if M > 0.9999:
    #     return 0.0

    if np.isnan(M):
        return 0.0
    
    if M > 1.0:
        return 1.0

    return M

