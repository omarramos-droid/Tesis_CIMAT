# -*- coding: utf-8 -*-
"""
Created on Tue May  5 15:59:44 2026

@author: dell
"""

# src/postprocess.py

import scipy.stats as stats  
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

# --------- GRÃFICAS ---------

def guardar_trazas(cadenas, outdir):
    os.makedirs(outdir, exist_ok=True)

    n_param = len(NOMBRES)
    fig, axes = plt.subplots(n_param, 1, figsize=(12, 3*n_param), sharex=True)

    for i, ax in enumerate(axes):
        for cad in cadenas:
            ax.plot(cad[:, i], alpha=0.6, lw=0.5)
        ax.set_ylabel(NOMBRES[i])

    axes[-1].set_xlabel('IteraciÃ³n')
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
    plt.xlabel('DÃ­a')
    plt.ylabel('Casos nuevos')
    plt.title(f'Ajuste â€“ {nsamples} curvas posteriores')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{outdir}/ajuste_incidencia.png")
    plt.close()

def graficar_dispersiones(posterior,outdir):
    n = len(NOMBRES)
    fig, axes = plt.subplots(n, n, figsize=(12, 12))
    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if i == j:
                ax.hist(posterior[:, i], bins=40, color='gray', alpha=0.7)
            else:
                ax.scatter(posterior[:, j], posterior[:, i],
                           alpha=0.3, s=5, color='blue', edgecolors='none')
            if i == n-1:
                ax.set_xlabel(NOMBRES[j])
            else:
                ax.set_xticklabels([])
            if j == 0:
                ax.set_ylabel(NOMBRES[i])
            else:
                ax.set_yticklabels([])
    plt.suptitle('Matriz de dispersión', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{outdir}/MatriD.png")
    plt.close()

def guardar_prior_vs_posterior(fit, outdir):
    os.makedirs(outdir, exist_ok=True)
    
    # Extraer muestras
    beta_post = fit.stan_variable('beta')
    gamma_post = fit.stan_variable('gamma')
    I0_post = fit.stan_variable('I0')
    alpha_post = fit.stan_variable('alpha')
    phi_post = fit.stan_variable('phi')

    # Configuración de diccionarios para iterar (Nombre: [Muestras, Función Prior, Título])
    params = {
    # Stan: beta ~ lognormal(log(1.7), 0.3)
    "beta":  [beta_post,  lambda x: stats.lognorm.pdf(x, 0.3, scale=1.7), r"$\beta$"],
    
    # Stan: gamma ~ lognormal(log(0.8), 0.2)
    "gamma": [gamma_post, lambda x: stats.lognorm.pdf(x, 0.2, scale=0.8), r"$\gamma$"],
    
    # Stan: I0 ~ lognormal(log(6), 2)
    "I0":    [I0_post,    lambda x: stats.lognorm.pdf(x, 2, scale=6),    "$I_0$"],
    
    # Stan: alpha ~ beta(2, 2)
    "alpha": [alpha_post, lambda x: stats.beta.pdf(x, 2, 2),              r"$\alpha$"],
    
    # Stan: phi ~ lognormal(log(30), 2)
    "phi":   [phi_post,   lambda x: stats.lognorm.pdf(x, 2, scale=30),   r"$\phi$"]
}


    n = len(params)
    rows = (n + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(14, 4 * rows))
    axes = axes.flatten()

    for i, (name, info) in enumerate(params.items()):
        ax = axes[i]
        post_samples, prior_pdf, titulo = info
        
        # Histograma posterior
        ax.hist(post_samples, bins=60, density=True, alpha=0.6, label="Posterior", color='skyblue')
        
        # Curva Prior
        x_min, x_max = ax.get_xlim()
        x = np.linspace(x_min, x_max, 500)
        ax.plot(x, prior_pdf(x), lw=2.5, color='crimson', label="Prior")
        
        ax.set_title(titulo)
        ax.legend()

    # Eliminar eje sobrante si es impar
    if len(axes) > n:
        fig.delaxes(axes[-1])

    plt.tight_layout()
    plt.savefig(f"{outdir}/prior_vs_posterior.png")
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

