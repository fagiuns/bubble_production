import numpy as np

# Function E(m, R, gamma) for energy emitted by constant velocity bubble


def E_constv(m, R, gamma, J_0):

    u = m * R * gamma

    A = -3 * (
        -7
        + 9 * gamma**2
        + 2 * m**2 * R**2 * gamma**2 * (-1 + 2 * gamma**2)
        + 2 * m * R * gamma * (-3 + 5 * gamma**2)
    )

    B = (
        4 * m**3 * R**3 * gamma**3
        + 6 * m**2 * R**2 * gamma**2 * (-2 + gamma**2)
        - 24 * m * R * gamma * (-1 + gamma**2)
        + 3 * (-8 + 12 * gamma**2 - 3 * gamma**4 + gamma**6)
    )

    return np.pi * J_0**2 * (B + np.exp(-2 * u) * A) / (6 * m**5 * gamma**3)


def E_speed_of_lightT(m, R, T):

    return (np.pi * T**2 * (4 * R**3 + 6 * R**2 * T + 6 * R * T**2 + 3 * T**3)) / (6 + 6 * m**2 * T**2)
