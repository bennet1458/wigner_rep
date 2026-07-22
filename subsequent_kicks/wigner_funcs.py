import numpy as np


def W0(x, p, sigma_x, sigma_p, hbar):
    return (
            1/(np.pi*hbar)
            * np.exp(
                -x**2/(2*sigma_x**2)
                -p**2/(2*sigma_p**2)
            )
        )
 


# def kick(W, x, p, dp, q, phi, hbar):
#     shift = int(round(q/dp))
#     shift_half = int(round(q/(2*dp)))

#     return 1*(np.roll(W, shift, axis=1)
#                 - 2 * np.cos(q/hbar * x + phi) * np.roll(W, shift_half, axis=1)
#                 + W
#             )



def kick(W, x, p, dp, q, phi, hbar):
    shift = int(round(q/dp))
    shift_half = int(round(q/(2*dp)))

    def shift_array(arr, s):
        shifted = np.zeros_like(arr)
        if s > 0:
            shifted[:, s:] = arr[:, :-s]
        elif s < 0:
            shifted[:, :s] = arr[:, -s:]
        else:
            shifted = arr.copy()
        return shifted

    return (shift_array(W, shift)
                + 2 * np.cos(q/hbar * x + phi) * shift_array(W, shift_half)
                + W
            )


def time_evolution(W, x, p, t, m, dx):
    W_new = np.empty_like(W)
    p = p[0, :]
    for j, pj in enumerate(p):
        # print((pj * t / m) / dx)
        shift = int(np.round((pj * t / m) / dx))  # shift in x-grid points
        W_new[:, j] = np.roll(W[:, j], shift)

    return W_new



def x_marginal(W, p):
    """Integrate 2D Wigner function over p axis (sum over p for each x)"""
    dp = p[1] - p[0]  # spacing in p
    return np.sum(W, axis=1) * dp