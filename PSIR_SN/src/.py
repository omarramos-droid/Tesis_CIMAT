# -*- coding: utf-8 -*-
"""
Created on Tue May  5 15:59:44 2026

@author: dell
"""

# src/postprocess.py

import numpy as np
import sys, os
sys.path.append(os.path.abspath("../scripts"))
from solver_fraccionario import *
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf


NOMBRES = ['beta', 'gamma', 'alpha', 'phi', 'I0']

def extraer_posterior_combinado(fit, nombres=NOMBRES):
    sv = fit.stan_variables()
    return np.column_stack([sv[var] for var in nombres])

def extraer_posterior_por_cadena(fit, nombres=NOMBRES):
    df = fit.draws_pd(vars=nombres)

    n_chains = fit.chains
    n_draws_total = len(df)
    n_draws_per_chain = n_draws_total // n_chains

    cadenas_idx = np.repeat(np.arange(1, n_chains + 1), n_draws_per_chain)
    draws_idx   = np.tile(np.arange(1, n_draws_per_chain + 1), n_chains)

    df.index = pd.MultiIndex.from_arrays(
        [cadenas_idx, draws_idx], names=['chain', 'draw']
    )

    cadenas = []
    for c in range(1, n_chains + 1):
        cadenas.append(df.xs(c, level='chain').to_numpy())

    return cadenas

# --------- GRÁFICAS ---------

def guardar_trazas(cadenas, outdir):
    os.makedirs(outdir, exist_ok=True)

    n_param = len(NOMBRES)
    fig, axes = plt.subplots(n_param, 1, figsize=(12, 3*n_param), sharex=True)

    for i, ax in enumerate(axes):
        for cad in cadenas:
            ax.plot(cad[:, i], alpha=0.6, lw=0.5)
        ax.set_ylabel(NOMBRES[i])

    axes[-1].set_xlabel('Iteración')
    plt.tight_layout()
    plt.savefig(f"{outdir}/trazas.png")
    plt.close()

def guardar_autocorrelacion(posterior, outdir):
    os.makedirs(outdir, exist_ok=True)

    fig, axes = plt.subplots(len(NOMBRES), 1, figsize=(10, 3*len(NOMBRES)))
    for i, ax in enumerate(axes):
        plot_acf(posterior[:, i], lags=50, ax=ax)

    plt.tight_layout()
    plt.savefig(f"{outdir}/acf.png")
    plt.close()
    

def graficar_ajuste_incidencia(fit, y_obs, t_obs, N_total, outdir,
                               h_solver=0.01, nsamples=20):
    posterior = extraer_posterior_combinado(fit)
    idx = np.random.choice(posterior.shape[0], size=nsamples, replace=False)
    dias = t_obs[1:]
    plt.figure(figsize=(12, 6))
    for ind in idx:
        beta, gamma, alpha, phi, I0 = posterior[ind, :]
        try:
            res = resolver_sir(beta=beta, gamma=gamma, N=N_total, I0=float(I0),
                               alpha=alpha, T=t_obs[-1], h=h_solver,
                               t_observacion=t_obs)
            S_obs = res['y_obs'][:, 0]
            inc = np.maximum(S_obs[:-1] - S_obs[1:], 1e-8)
            plt.plot(dias, inc, color='blue', alpha=0.15, linewidth=0.8)
        except:
            continue
    # Curva media
    theta_media = np.mean(posterior, axis=0)
    try:
        res_media = resolver_sir(beta=theta_media[0], gamma=theta_media[1],
                                 N=N_total, I0=float(theta_media[4]),
                                 alpha=theta_media[2], T=t_obs[-1], h=h_solver,
                                 t_observacion=t_obs)
        S_media = res_media['y_obs'][:, 0]
        inc_media = np.maximum(S_media[:-1] - S_media[1:], 1e-8)
        plt.plot(dias, inc_media, 'r-', linewidth=2.5, label='Media posterior')
    except:
        pass
    plt.scatter(dias, y_obs, color='black', s=20, label='Datos observados', zorder=5)
    plt.xlabel('Día')
    plt.ylabel('Casos nuevos')
    plt.title(f'Ajuste – {nsamples} curvas posteriores')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{outdir}/ajuste_incidencia.png")
    plt.close()

def graficar_autocorrelacion(posterior, outdir,lags=50):
    n_param = len(NOMBRES)
    fig, axes = plt.subplots(n_param, 1, figsize=(10, 3*n_param))
    if n_param == 1: axes = [axes]
    for i, ax in enumerate(axes):
        plot_acf(posterior[:, i], lags=lags, ax=ax, title=f'Autocorrelaci�n {NOMBRES[i]}')
    plt.tight_layout()
    plt.savefig(f"{outdir}/autocorrelacion.png")
    plt.close()


# --------- RESUMEN ---------

def resumen_numerico(posterior):
    res = {}
    for i, nom in enumerate(NOMBRES):
        vals = posterior[:, i]
        res[nom] = [
            np.mean(vals),
            np.percentile(vals, 2.5),
            np.percentile(vals, 97.5)
        ]

    df = pd.DataFrame(res, index=['mean','2.5%','97.5%']).T
    return df
