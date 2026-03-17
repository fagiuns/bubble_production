import numpy as np


def dEdk_constv(k, m, R, gamma):

    # Compute dE/dk for constant velocity bubble for given k, m, R, gamma.

    # Works with either:
    # - scalar k
    # - numpy array k

    term1 = (
        (gamma**2 - 1)
        * (k**3 + k * m**2 * gamma**2)**4
        * (
            -2 * k * (gamma**2 - 1) * np.cos(k * R)
            + R * (k**2 + m**2 * gamma**2) * np.sin(k * R)
        )**2
    )

    term2 = (
        (k**2 + m**2)
        * gamma**2
        * (k**2 + m**2 * gamma**2)**4
        * (
            k * R * (k**2 + m**2 * gamma**2) * np.cos(k * R)
            + (-m**2 * gamma**2 + k**2 * (-3 + 2 * gamma**2)) * np.sin(k * R)
        )**2
    )

    numerator = gamma**2 * 4 * (term1 + term2)
    denominator = k**4 * (m**2 * gamma**2 + k**2)**8

    return numerator / denominator
