import numpy as np
from scipy.integrate import quad

# Functions for bubble production from accelerated bubble for time tau


def RedE_acc(k, m, R, tau):

    # Compute real part of complex integral

    def integrand(t):
        omega = np.sqrt(k**2 + m**2)
        s = np.sqrt(R**2 + t**2)
        return np.cos(omega * t) * np.exp(-t / tau) * (np.sin(k * s) - k * s * np.cos(k * s))

    t_max = 4 * tau
    val = 0.0

    for t0, t1 in zip(np.linspace(0, t_max, 6)[:-1], np.linspace(0, t_max, 6)[1:]):
        v, _ = quad(integrand, t0, t1, limit=1000, epsabs=1e-6, epsrel=1e-4)
        val += v

    return val


def ImdE_acc(k, m, R, tau):

    # Compute imaginary part of complex integral

    def integrand(t):
        omega = np.sqrt(k**2 + m**2)
        s = np.sqrt(R**2 + t**2)
        return np.sin(omega * t) * np.exp(-t / tau) * (np.sin(k * s) - k * s * np.cos(k * s))

    t_max = 4 * tau
    val = 0.0

    for t0, t1 in zip(np.linspace(0, t_max, 6)[:-1], np.linspace(0, t_max, 6)[1:]):
        v, _ = quad(integrand, t0, t1, limit=1000, epsabs=1e-6, epsrel=1e-4)
        val += v

    return val


def dEdk_acc(k, m, R, tau):

    # dE/dk for bubble accelerated for tau (change J_0 if needed)

    if k == 0:
        return 0.0

    Re = RedE_acc(k, m, R, tau)
    Im = ImdE_acc(k, m, R, tau)
    J_0 = 1

    return (4 * J_0**2 * (Re**2 + Im**2)) / k**4


def E_acc(m, R, tau):

    # Energy for accelerated bubble at time tau

    def integrand(k):
        return dEdk_acc(k, m, R, tau)

    k_max = 4 * m**2 * tau
    val, _ = quad(integrand, 0, k_max, limit=400, epsabs=1e-5, epsrel=1e-3)

    return val
