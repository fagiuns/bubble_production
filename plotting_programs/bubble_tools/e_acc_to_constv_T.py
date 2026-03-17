import numpy as np
from scipy.integrate import quad

# Functions for bubble production that goes from constant acceleration to constant velocity
# MEASURED AT A CERTAIN TIME T


def RedE_T(k, m, R, T, tau):

    # Compute real part of complex integral

    kp2 = k**2
    mp2 = m**2
    omega = np.sqrt(kp2 + mp2)
    rho2 = R**2 + tau**2
    rho = np.sqrt(rho2)
    sup = np.exp(-tau / T)

    # Constant velocity part
    # ---------- First big bracket ----------
    A1 = (
        -np.cos(omega * tau) * (
            k * (
                (1 + (kp2 + mp2)*T**2)**3 * rho2**4
                + k**6 * T**6 * tau**6 * (R**2 + tau*(-2*T + tau))
                - k**4 * T**4 * tau**4 * rho2 * (
                    -4*T*(-1 + (kp2 + mp2)*T**2)*tau
                    + (-3 + (kp2 + mp2)*T**2)*rho2
                )
                - k**2 * T**2 * tau**2 * rho2**2 * (
                    2*T*(1 + (kp2 + mp2)*T**2*(-6 + (kp2 + mp2)*T**2))*tau
                    + (-3 + (kp2 + mp2)*T**2)*(1 + (kp2 + mp2)*T**2)*rho2
                )
            ) * np.cos(k * rho)
            - rho * (
                k**8 * T**7 * tau**7
                + (1 + (kp2 + mp2)*T**2)**3 * rho2**3
                + 3*k**6*T**5*tau**5 * (
                    rho2 + T*tau - (kp2 + mp2)*T**2*rho2
                )
                + k**2*T*(1 + (kp2 + mp2)*T**2)*tau*rho2**2 * (
                    R**2 + 5*T*tau
                    - 7*(kp2 + mp2)*T**3*tau
                    + tau**2
                    - (kp2 + mp2)**2 * T**4 * rho2
                )
                + k**4*T**3*tau**3*rho2 * (
                    7*T*tau
                    + 3*(kp2 + mp2)*T**3*tau
                    + 3*rho2
                    - 2*(kp2 + mp2)*T**2*rho2
                    + 3*(kp2 + mp2)**2 * T**4 * rho2
                )
            ) * np.sin(k * rho)
        )
    )

    # ---------- Second big bracket ----------
    A2 = (
        -omega * T * rho * np.sin(omega * tau) * (
            k * rho * (
                k**6*T**6*tau**6
                - (1 + (kp2 + mp2)*T**2)**3 * rho2**3
                + k**4*T**4*tau**4 * (
                    8*T*tau
                    + (1 - 3*(kp2 + mp2)*T**2)*rho2
                )
                + k**2*T**2*tau**2 * rho2 * (
                    -8*T*(-1 + (kp2 + mp2)*T**2)*tau
                    - (1 - 3*(kp2 + mp2)*T**2)*(1 + (kp2 + mp2)*T**2)*rho2
                )
            ) * np.cos(k * rho)
            + (
                (1 + (kp2 + mp2)*T**2)**3 * rho2**3
                + k**6*T**5*tau**5 * (2*R**2 + tau*(-3*T + 2*tau))
                + k**4*T**3*tau**3*rho2 * (
                    3*T*tau
                    + 7*(kp2 + mp2)*T**3*tau
                    + 4*rho2
                    - 4*(kp2 + mp2)*T**2*rho2
                )
                + k**2*T*(1 + (kp2 + mp2)*T**2)*tau*rho2**2 * (
                    T*(7 - 5*(kp2 + mp2)*T**2)*tau
                    + 2*(1 + (kp2 + mp2)*T**2)*rho2
                )
            ) * np.sin(k * rho)
        )
    )

    # ---------- Denominator ----------
    denom = (
        T**7 * rho2**(7/2)
        * (
            (kp2 + mp2 + 1/T**2)**2
            + (k**4 * tau**4)/rho2**2
            + (2*k**2*(-kp2 - mp2 + 1/T**2)*tau**2)/rho2
        )**2
    )

    # Numerical Integral for constant acceleration

    def integrand(t):
        s = np.sqrt(R**2 + t**2)
        return np.cos(omega * t) * np.exp(-t / T) * (np.sin(k * s) - k * s * np.cos(k * s))

    t_max = min(4 * T, tau)
    val = 0.0

    for t0, t1 in zip(np.linspace(0, t_max, 6)[:-1], np.linspace(0, t_max, 6)[1:]):
        v, _ = quad(integrand, t0, t1, limit=1000, epsabs=1e-6, epsrel=1e-4)
        val += v

    return val + sup*((A1 + A2) / denom)


def ImdE_T(k, m, R, T, tau):

    # Compute imaginary part of complex integral

    kp2 = k**2
    mp2 = m**2
    omega = np.sqrt(kp2 + mp2)
    rho2 = R**2 + tau**2
    rho = np.sqrt(rho2)
    sup = np.exp(-tau / T)

    # Constant velocity part
    # ---------- First big bracket ----------
    A1 = (
        -np.sin(omega * tau) * (
            k * (
                (1 + (kp2 + mp2)*T**2)**3 * rho2**4
                + k**6 * T**6 * tau**6 * (R**2 + tau*(-2*T + tau))
                - k**4 * T**4 * tau**4 * rho2 * (
                    -4*T*(-1 + (kp2 + mp2)*T**2)*tau
                    + (-3 + (kp2 + mp2)*T**2)*rho2
                )
                - k**2 * T**2 * tau**2 * rho2**2 * (
                    2*T*(1 + (kp2 + mp2)*T**2*(-6 + (kp2 + mp2)*T**2))*tau
                    + (-3 + (kp2 + mp2)*T**2)*(1 + (kp2 + mp2)*T**2)*rho2
                )
            ) * np.cos(k * rho)
            - rho * (
                k**8 * T**7 * tau**7
                + (1 + (kp2 + mp2)*T**2)**3 * rho2**3
                + 3*k**6*T**5*tau**5 * (
                    rho2 + T*tau - (kp2 + mp2)*T**2*rho2
                )
                + k**2*T*(1 + (kp2 + mp2)*T**2)*tau*rho2**2 * (
                    R**2 + 5*T*tau
                    - 7*(kp2 + mp2)*T**3*tau
                    + tau**2
                    - (kp2 + mp2)**2 * T**4 * rho2
                )
                + k**4*T**3*tau**3*rho2 * (
                    7*T*tau
                    + 3*(kp2 + mp2)*T**3*tau
                    + 3*rho2
                    - 2*(kp2 + mp2)*T**2*rho2
                    + 3*(kp2 + mp2)**2 * T**4 * rho2
                )
            ) * np.sin(k * rho)
        )
    )

    # ---------- Second big bracket ----------
    A2 = (
        omega * T * rho * np.cos(omega * tau) * (
            k * rho * (
                k**6*T**6*tau**6
                - (1 + (kp2 + mp2)*T**2)**3 * rho2**3
                + k**4*T**4*tau**4 * (
                    8*T*tau
                    + (1 - 3*(kp2 + mp2)*T**2)*rho2
                )
                + k**2*T**2*tau**2 * rho2 * (
                    -8*T*(-1 + (kp2 + mp2)*T**2)*tau
                    - (1 - 3*(kp2 + mp2)*T**2)*(1 + (kp2 + mp2)*T**2)*rho2
                )
            ) * np.cos(k * rho)
            + (
                (1 + (kp2 + mp2)*T**2)**3 * rho2**3
                + k**6*T**5*tau**5 * (2*R**2 + tau*(-3*T + 2*tau))
                + k**4*T**3*tau**3*rho2 * (
                    3*T*tau
                    + 7*(kp2 + mp2)*T**3*tau
                    + 4*rho2
                    - 4*(kp2 + mp2)*T**2*rho2
                )
                + k**2*T*(1 + (kp2 + mp2)*T**2)*tau*rho2**2 * (
                    T*(7 - 5*(kp2 + mp2)*T**2)*tau
                    + 2*(1 + (kp2 + mp2)*T**2)*rho2
                )
            ) * np.sin(k * rho)
        )
    )

    # ---------- Denominator ----------
    denom = (
        T**7 * rho2**(7/2)
        * (
            (kp2 + mp2 + 1/T**2)**2
            + (k**4 * tau**4)/rho2**2
            + (2*k**2*(-kp2 - mp2 + 1/T**2)*tau**2)/rho2
        )**2
    )

    # Numerical Integral for constant acceleration

    def integrand(t):
        s = np.sqrt(R**2 + t**2)
        return np.sin(omega * t) * np.exp(-t / T) * (np.sin(k * s) - k * s * np.cos(k * s))

    t_max = min(4 * T, tau)
    val = 0.0

    for t0, t1 in zip(np.linspace(0, t_max, 6)[:-1], np.linspace(0, t_max, 6)[1:]):
        v, _ = quad(integrand, t0, t1, limit=1000, epsabs=1e-6, epsrel=1e-4)
        val += v

    return val + sup * ((A1 + A2) / denom)


def dEdk_acc_to_constv_T(k, m, R, T, tau):

    # dE/dk at finite T (change J_0 if needed)

    if k == 0:
        return 0.0

    Re = RedE_T(k, m, R, T, tau)
    Im = ImdE_T(k, m, R, T, tau)
    J_0 = 1

    return (4 * J_0**2 * (Re**2 + Im**2)) / k**4


def E_acc_to_constv_T(m, R, T, tau):

    # Energy at finite T

    def integrand(k):
        return dEdk_acc_to_constv_T(k, m, R, T, tau)

    k_max = 4 * m**2 * T
    val, _ = quad(integrand, 0, k_max, limit=400, epsabs=1e-5, epsrel=1e-3)

    return val
