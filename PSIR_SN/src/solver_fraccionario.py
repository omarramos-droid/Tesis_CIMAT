# -*- coding: utf-8 -*-
"""

@author: omar.ramos

Solver fraccionario Diethelm predictor-corrector.
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


def proyectar_positivo(y, N_total):
    """Proyecta a valores >= 0 conservando N total."""
    y_proj = np.maximum(y, 0.0)
    total = np.sum(y_proj)
    if total > 1e-12:
        y_proj = y_proj * (N_total / total)
    else:
        y_proj = np.array([N_total, 0.0, 0.0])
    return y_proj


def resolver_fraccionario(funcion_rhs, alpha, y0, T, h, t_observacion=None):
    """Resuelve sistema fraccionario con Diethelm predictor-corrector."""
    N_pasos = int(round(T / h))
    t_fino = np.linspace(0, T, N_pasos + 1)
    dim = len(y0)
    N_total = np.sum(y0)
    
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


def sir_rhs(t, y, beta, gamma, N):
    """Lado derecho del modelo SIR."""
    S, I, R = y
    return np.array([-beta * S * I / N,
                      beta * S * I / N - gamma * I,
                      gamma * I])


def resolver_sir(beta, gamma, N, I0, alpha, T, h, t_observacion=None):
    """FunciÃ³n de conveniencia para resolver SIR fraccionario."""
    y0 = np.array([N - I0, I0, 0.0])
    return resolver_fraccionario(
        lambda t, y: sir_rhs(t, y, beta, gamma, N),
        alpha, y0, T, h, t_observacion
    )