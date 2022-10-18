"""
This python file contain the main functions frequently used in the other scripts, especially differential equation
systems.
This module is imported at the beginning of each script.
"""

### libraries
import numpy as np
from numpy import exp
from scipy.integrate import odeint


def g_(k, rho):
    """
    :param k: cost of male vs female
    :param rho: proportion of male in the population
    :return: proportion of female produced in the progeny
    """
    g = (1 - exp(-k * rho)) / (1 + exp(-k * rho))
    return g


def no_mimicry(n, t, param_dict):
    """
    Differential equations system which can be used for:
                - one species only (n[2] = 0, cb = 0, s = 0)
                - two species without sympatry and without mimicry (cb = 0)
                - two sympatric species without mimicry
    :param n: array containing number of females and males [F1,M1,F2,M2]
    :param t: time
    :param param_dict: dictionary for all parameters
    :return: dF1/dt, dM1/dt, dF2/dt, dM2/dt
    """
    rho1 = np.divide(n[1], (n[0] + n[1]), out=np.zeros_like(n[1]), where=(n[0] + n[1]) > 0) # male proportion in the population for species 1
    rho2 = np.divide(n[3], (n[2] + n[3]), out=np.zeros_like(n[3]), where=(n[2] + n[3]) > 0) # male proportion in the population for species 2

    b = param_dict['b']
    d = param_dict['d']
    p = param_dict['p']
    l1 = param_dict['l1']
    k1 = param_dict['k1']
    l2 = param_dict['l2']
    k2 = param_dict['k2']
    cw = param_dict['cw']
    cb = param_dict['cb']
    K = param_dict['K']
    a = param_dict['a']
    B = param_dict['B']

    # F1, M1, F2, M2
    return (np.array([
        n[0] * b * g_(rho1, k1) - d * n[0] - n[0] * p * (1 - a * l1) / (1 + l1 * n[0] * (1 - B * rho1)) - (
                    cw * n[0] + cb * n[2]) * n[0] / K,
        n[0] * b * (1 - g_(rho1, k1)) - d * n[1] - n[1] * p / (1 + l1 * n[0] * (1 - B * rho1)),
        n[2] * b * g_(rho2, k2) - d * n[2] - n[2] * p * (1 - a * l2) / (1 + l2 * n[2] * (1 - B * rho2)) - (
                    cw * n[2] + cb * n[0]) * n[2] / K,
        n[2] * b * (1 - g_(rho2, k2)) - d * n[3] - n[3] * p / (1 + l2 * n[2] * (1 - B * rho2))
    ], dtype='float64'))


def mimicry(n, t, param_dict):
    """
    Differential equations system to be used for two mimetic species
    :param n: array containing number of females and males [F1,M1,F2,M2]
    :param t: time
    :param param_dict: dictionary for all parameters
    :return: dF1/dt, dM1/dt, dF2/dt, dM2/dt
    """
    rho1 = np.divide(n[1], (n[0] + n[1]), out=np.zeros_like(n[1]), where=(n[0] + n[1]) > 0) # male proportion in the population for species 1
    rho2 = np.divide(n[3], (n[2] + n[3]), out=np.zeros_like(n[3]), where=(n[2] + n[3]) > 0) # male proportion in the population for species 2
    rho3 = np.divide((n[1] + n[3]), (n[0] + n[1] + n[2] + n[3]), out=np.zeros_like(n[3]),
                     where=(n[0] + n[1] + n[2] + n[3]) > 0) # male proportion in the mimetic community

    b = param_dict['b']
    d = param_dict['d']
    p = param_dict['p']
    l1 = param_dict['l1']
    k1 = param_dict['k1']
    l2 = param_dict['l2']
    k2 = param_dict['k2']
    cw = param_dict['cw']
    cb = param_dict['cb']
    K = param_dict['K']
    a = param_dict['a']
    B = param_dict['B']

    # F1, M1, F2, M2
    return (np.array([
        n[0] * b * g_(rho1, k1) - d * n[0] - n[0] * p * (1 - a * l1) / (
                    1 + (l1 * n[0] + l2 * n[2]) * (1 - B * rho3)) - (cw * n[0] + cb * n[2]) * n[0] / K,
        n[0] * b * (1 - g_(rho1, k1)) - d * n[1] - n[1] * p / (1 + (l1 * n[0] + l2 * n[2]) * (1 - B * rho3)),
        n[2] * b * g_(rho2, k2) - d * n[2] - n[2] * p * (1 - a * l2) / (
                    1 + (l1 * n[0] + l2 * n[2]) * (1 - B * rho3)) - (cw * n[2] + cb * n[0]) * n[2] / K,
        n[2] * b * (1 - g_(rho2, k2)) - d * n[3] - n[3] * p / (1 + (l1 * n[0] + l2 * n[2]) * (1 - B * rho3))
    ], dtype='float64'))


def dslm(n, t, param_dict):
    """
    Differential equations system for the case of DSLM.
    The first species is monomorphic and the second species is dimorphic.
    :param n: array containing number of females and males [F1,M1,F2,M2]
    :param t: time
    :param param_dict: dictionary for all parameters
    :return: dF1/dt, dM1/dt, dF2/dt, dM2/dt
    """
    rho1 = np.divide(n[1], (n[0] + n[1]), out=np.zeros_like(n[1]), where=(n[0] + n[1]) > 0) # male proportion in the population for species 1
    rho2 = np.divide(n[3], (n[2] + n[3]), out=np.zeros_like(n[3]), where=(n[2] + n[3]) > 0) # male proportion in the population for species 2
    rho3 = np.divide((n[1] + n[3]), (n[1] + n[0] + n[3]), out=np.zeros_like(n[3]), where=(n[1] + n[0] + n[3]) > 0) # male proportion in the mimetic community

    b = param_dict['b']
    d = param_dict['d']
    p = param_dict['p']
    l1 = param_dict['l1']
    k1 = param_dict['k1']
    l2 = param_dict['l2']
    k2 = param_dict['k2']
    cw = param_dict['cw']
    cb = param_dict['cb']
    K = param_dict['K']
    a = param_dict['a']
    B = param_dict['B']


    # F1, M1, F2, M2
    return (np.array([
        n[0] * b * g_(rho1, k1) - d * n[0] - n[0] * p * (1 - a * l1) / (
                1 + l1 * n[0] * (1 - B * rho3)) - (cw * n[0] + cb * n[2]) * n[0] / K,

        n[0] * b * (1 - g_(rho1, k1)) - d * n[1] - n[1] * p / (1 + l1 * n[0] * (1 - B * rho3)),

        n[2] * b * g_(rho2, k2) - d * n[2] - n[2] * p * (1 - a * l2) / (
                1 + l2 * n[2]) - (cw * n[2] + cb * n[0]) * n[2] / K,

        n[2] * b * (1 - g_(rho2, k2)) - d * n[3] - n[3] * p / (1 + l1 * n[0] * (1 - B * rho3))
    ], dtype='float64'))


def solver(func, AB, SR, ab, sr, b, d, p, l1, k1, l2, k2, cw, cb, K, a, B):
    """
    :param func: function to use (no_mimicry, mimicry or dslm)
    :param AB: total initial abundance of the species 1 population (F1+M1)
    :param SR: initial proportion of male for the species 1
    :param ab: total initial abundance of the species 2 population (F2+M2)
    :param sr: initial proportion of male for the species 2
    :param b: birth rate
    :param d: death rate (excluding predation)
    :param p: predation death rate
    :param l1: defence level of F1
    :param k1: relative investment in producing M1 vs F1
    :param l2: defence level of F2
    :param k2: relative investment in producing M2 vs F2
    :param cw: intraspecific competition
    :param cb: interspecific competition
    :param K: carrying capacity link to resources
    :param a: intensity of direct avantage to females due to their painful sting
    :param B: intensity of male cost on protection brought by Müllerian mimicry
    :return: [persistence of sp1 (0/1), persistence of sp2 (0/1), coexistence (0/1), F1, M1, F2, M2]
    """
    TIME_INT = np.linspace(0, 50, 500)

    cond_ini = np.array([AB * (1 - SR), AB * SR,
                         ab * (1 - sr), ab * sr], dtype='float64')

    first_state = cond_ini
    iteration = 0
    exit = 0

    param_list = {'b': b,
                  'd': d,
                  'p': p,
                  'l1': l1,
                  'k1': k1,
                  'l2': l2,
                  'k2': k2,
                  'cw': cw,
                  'cb': cb,
                  'K': K,
                  'a': a,
                  'B': B}

    while exit == 0:
        sol = odeint(func, first_state, TIME_INT, args=(param_list,))
        second_state = sol[-1, :]
        if iteration == 100:
            break
        elif np.any(np.abs(second_state - first_state) > 0.0001):
            first_state = second_state
            iteration += 1
        else:
            exit = 1

    if (sol[-1, 1] + sol[-1, 0]) > 0.001:
        eq_sp1 = 1
    else:
        eq_sp1 = 0

    if (sol[-1, 3] + sol[-1, 2]) > 0.001:
        eq_sp2 = 1
    else:
        eq_sp2 = 0

    if (eq_sp1 == 1) & (eq_sp2 == 1):
        coexistence = 1
    else:
        coexistence = 0

    return [eq_sp1, eq_sp2, coexistence, sol[-1, 0], sol[-1, 1], sol[-1, 2], sol[-1, 3]]


if __name__ == '__main__':
    # examples
    print(solver(no_mimicry, 1000, 0.5, 0, 0.5, 1, 0.2, 0.3, 0.05, 3, 0, 3, 1, 0.3, 1000, 5, 0.8))  # no sympatry
    print(solver(no_mimicry, 1000, 0.5, 1000, 0.5, 1, 0.2, 0.3, 0.05, 3, 0, 3, 1, 0.3, 1000, 5, 0.8))  # no mimicry
    print(solver(mimicry, 1000, 0.5, 1000, 0.5, 1, 0.2, 0.3, 0.05, 3, 0, 3, 1, 0.3, 1000, 5, 0.8))  # mimicry
    print(solver(dslm, 1000, 0.5, 1000, 0.5, 1, 0.2, 0.3, 0.05, 3, 0, 3, 1, 0.3, 1000, 5, 0.8))  # mimicry with DSLM
