import numpy as np
from scipy.integrate import quad

# Functions for TOTAL bubble production that goes from constant acceleration to constant velocity


def RedE(k, m, R, tau):

    # Compute real part of complex integral

    kp2 = k**2
    mp2 = m**2
    omega = np.sqrt(kp2 + mp2)
    rho2 = R**2 + tau**2
    rho = np.sqrt(rho2)
    u = tau / rho

    # Constant velocity part

    A = (np.sin(k * rho) * (kp2 * u * (kp2 * u**2 - omega**2) * rho * np.cos(omega * tau) -
                            omega * (omega**2 - 3 * kp2 * u**2) * np.sin(omega * tau)) +
         k * np.cos(k * rho) * (2 * kp2 * u**3 * np.cos(omega * tau) +
                                omega * (omega**2 - kp2 * u**2) * rho * np.sin(omega * tau)))/((omega**2 - kp2 * u**2)**2)

    # Numerical Integral for constant acceleration

    def integrand(t):
        s = np.sqrt(R**2 + t**2)
        return np.cos(omega * t) * np.exp(-t / tau) * (np.sin(k * s) - k * s * np.cos(k * s))

    t_max = 4 * tau
    val = 0.0

    for t0, t1 in zip(np.linspace(0, t_max, 6)[:-1], np.linspace(0, t_max, 6)[1:]):
        v, _ = quad(integrand, t0, t1, limit=1000, epsabs=1e-6, epsrel=1e-4)
        val += v

    return val + A


def ImdE(k, m, R, tau):

    # Compute real part of complex integral

    kp2 = k**2
    mp2 = m**2
    omega = np.sqrt(kp2 + mp2)
    rho2 = R**2 + tau**2
    rho = np.sqrt(rho2)
    u = tau / rho

    # Constant velocity part

    B = (np.sin(k * rho) * (kp2 * u * (kp2 * u**2 - omega**2) * rho * np.sin(omega * tau) +
                            omega * (omega**2 - 3 * kp2 * u**2) * np.cos(omega * tau)) +
         k * np.cos(k * rho) * (2 * kp2 * u**3 * np.sin(omega * tau) -
                                omega * (omega**2 - kp2 * u**2) * rho * np.cos(omega * tau)))/((omega**2 - kp2 * u**2)**2)

    # Numerical Integral for constant acceleration

    def integrand(t):
        s = np.sqrt(R**2 + t**2)
        return np.sin(omega * t) * np.exp(-t / tau) * (np.sin(k * s) - k * s * np.cos(k * s))

    t_max = 4 * tau
    val = 0.0

    for t0, t1 in zip(np.linspace(0, t_max, 6)[:-1], np.linspace(0, t_max, 6)[1:]):
        v, _ = quad(integrand, t0, t1, limit=1000, epsabs=1e-6, epsrel=1e-4)
        val += v

    return val + B


def dEdk_acc_to_constv(k, m, R, tau):

    # dE/dk (change J_0 if needed)

    if k == 0:
        return 0.0

    Re = RedE(k, m, R, tau)
    Im = ImdE(k, m, R, tau)
    J_0 = 1/(10**16 * R**4)

    return (4 * J_0**2 * (Re**2 + Im**2)) / k**4


def E_acc_to_constv(m, R, tau):

    # Total Energy

    def integrand(k):
        return dEdk_acc_to_constv(k, m, R, tau)

    k_max = 4 * m * np.sqrt(R**2 + tau**2) / R
    val, _ = quad(integrand, 0, k_max, limit=400, epsabs=1e-5, epsrel=1e-3)

    return val
