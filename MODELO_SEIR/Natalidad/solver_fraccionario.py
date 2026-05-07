# -*- coding: utf-8 -*-
"""
Solver fraccionario Diethelm predictor‑corrector – SIR y SEIR con demografía.
Incidencia: diferencia diaria de susceptibles (S(t-1)-S(t)).
"""

import numpy as np
from math import gamma

_cache_potencias = {}

def precalcular_potencias(alpha, n_max):
    """Precalcula k^alpha y k^(alpha+1) para k=0..n_max+1."""
    key = (alpha, n_max)
    if key not in _cache_potencias:
        pow_alpha = np.zeros(n_max + 2)
        pow_alpha_plus1 = np.zeros(n_max + 2)
        for k in range(1, n_max + 2):
            pow_alpha[k] = k ** alpha
            pow_alpha_plus1[k] = k ** (alpha + 1)
        _cache_potencias[key] = (pow_alpha, pow_alpha_plus1)
    return _cache_potencias[key]


def proyectar_positivo(y, N_total=None):
    """
    Proyecta a valores >= 0.
    Si N_total se provee, reescala para conservar la suma (modelos con N fija).
    Si es None (modelos con demografía), solo trunca en 0.
    """
    y_proj = np.maximum(y, 0.0)
    if N_total is not None:
        total = np.sum(y_proj)
        if total > 1e-12:
            y_proj = y_proj * (N_total / total)
        else:
            y_proj = np.array([N_total] + [0.0]*(len(y)-1))
    return y_proj


def resolver_fraccionario(funcion_rhs, alpha, y0, T, h,
                          t_observacion=None, conservar_N=True):
    """
    Resuelve sistema fraccionario con Diethelm predictor‑corrector.

    conservar_N : bool
        Si True (SIR, SEIR sin demografía) usa proyección que mantiene N total.
        Si False (SEIR con demografía) solo trunca valores negativos.
    """
    N_pasos = int(round(T / h))
    t_fino = np.linspace(0, T, N_pasos + 1)
    dim = len(y0)
    N_total = np.sum(y0) if conservar_N else None

    gamma1 = gamma(alpha + 1)
    gamma2 = gamma(alpha + 2)
    h_alpha = h ** alpha

    pow_alpha, pow_alpha_plus1 = precalcular_potencias(alpha, N_pasos + 2)

    y = np.zeros((N_pasos + 1, dim))
    f_vals = np.zeros((N_pasos + 1, dim))

    y[0] = y0
    f_vals[0] = funcion_rhs(0.0, y[0])

    for n in range(1, N_pasos + 1):
        # Predictor
        b_coeffs = np.array([pow_alpha[n - j] - pow_alpha[n - j - 1]
                             for j in range(n)])
        y_pred = y[0] + (h_alpha / gamma1) * np.dot(b_coeffs, f_vals[:n])
        y_pred = proyectar_positivo(y_pred, N_total)
        f_pred = funcion_rhs(t_fino[n], y_pred)

        # Corrector
        a_coeffs = np.zeros(n + 1)
        if n >= 1:
            a_coeffs[0] = pow_alpha_plus1[n-1] - (n - 1 - alpha) * pow_alpha[n]
        for j in range(1, n):
            a_coeffs[j] = (pow_alpha_plus1[n - j + 1] +
                          pow_alpha_plus1[n - j - 1] -
                          2 * pow_alpha_plus1[n - j])
        a_coeffs[n] = 1.0

        y_corr = y[0] + (h_alpha / gamma2) * (
            np.dot(a_coeffs[:n], f_vals[:n]) + a_coeffs[n] * f_pred)
        y[n] = proyectar_positivo(y_corr, N_total)
        f_vals[n] = funcion_rhs(t_fino[n], y[n])

    resultado = {'t_fino': t_fino, 'y_fino': y}

    if t_observacion is not None:
        indices = np.clip(np.round(t_observacion / h).astype(int), 0, N_pasos)
        resultado['y_obs'] = y[indices]
        resultado['t_obs'] = t_observacion

    return resultado


# Lados derechos de los modelos

def sir_rhs(t, y, beta, gamma, N):
    """SIR clásico (N constante)."""
    S, I, R = y
    return np.array([-beta * S * I / N,
                      beta * S * I / N - gamma * I,
                      gamma * I])


def seir_rhs(t, y, beta, gamma, kappa, N):
    """SEIR sin demografía (N constante)."""
    S, E, I, R = y
    dS = -beta * S * I / N
    dE =  beta * S * I / N - kappa * E
    dI =  kappa * E - gamma * I
    dR =  gamma * I
    return np.array([dS, dE, dI, dR])


def seirv_rhs(t, y, beta, sigma, gamma, mu, Lambda):
    """
    SEIR con demografía (población variable).
    N(t) = S+E+I+R se calcula internamente.
    sigma : tasa de progresión E → I 
    Lambda: tasa de reclutamiento (nacimientos)
    mu    : tasa de mortalidad per cápita
    """
    S, E, I, R = y
    N = S + E + I + R
    dS = Lambda - beta * S * I / N - mu * S
    dE = beta * S * I / N - (sigma + mu) * E
    dI = sigma * E - (gamma + mu) * I
    dR = gamma * I - mu * R
    return np.array([dS, dE, dI, dR])


# ------------------------------------------------------------
# Funciones de conveniencia para resolver cada modelo
# ------------------------------------------------------------

def resolver_sir(beta, gamma, N, I0, alpha, T, h, t_observacion=None):
    """SIR fraccionario con población constante N."""
    y0 = np.array([N - I0, I0, 0.0])
    return resolver_fraccionario(
        lambda t, y: sir_rhs(t, y, beta, gamma, N),
        alpha, y0, T, h, t_observacion,
        conservar_N=True
    )


def resolver_seir(beta, gamma, kappa, N, I0, E0, alpha, T, h, t_observacion=None):
    """SEIR fraccionario sin demografía (N constante)."""
    y0 = np.array([N - I0 - E0, E0, I0, 0.0])
    return resolver_fraccionario(
        lambda t, y: seir_rhs(t, y, beta, gamma, kappa, N),
        alpha, y0, T, h, t_observacion,
        conservar_N=True
    )


def resolver_seirv(beta, sigma, gamma, mu, Lambda, alpha,
                   N0, I0, E0, T, h, t_observacion=None):
    """
    SEIR fraccionario con demografía (población variable).
    N0 : población total inicial (S0+E0+I0, con R0=0)
    """
    S0 = N0 - I0 - E0
    y0 = np.array([S0, E0, I0, 0.0])
    return resolver_fraccionario(
        lambda t, y: seirv_rhs(t, y, beta, sigma, gamma, mu, Lambda),
        alpha, y0, T, h, t_observacion,
        conservar_N=False      # ← población variable
    )


# Incidencia: diferencia de susceptibles 


def calcular_incidencia_diffS(resultado):
    """
    Calcula la incidencia diaria como S(t-1) - S(t).
    Supone que 'y_obs' contiene los compartimentos en tiempos 0,1,...,T.
    Retorna array de longitud T (días 1..T).
    """
    if 'y_obs' not in resultado:
        raise ValueError("El diccionario debe contener 'y_obs' (usar t_observacion).")
    S = resultado['y_obs'][:, 0]          # S en cada tiempo de observación
    inc = np.maximum(S[:-1] - S[1:], 1e-9)
    return inc