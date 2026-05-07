# -*- coding: utf-8 -*-
"""
Post‑procesamiento para modelo SEIR fraccionario con demografía.
Adaptado para los parámetros del modelo Stan.
"""

import scipy.stats as stats
import numpy as np
import sys
import os

sys.path.append(os.path.abspath("../scripts"))
from solver_fraccionario import resolver_seirv  # tu solver SEIR

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf


# Parámetros estimados 
NOMBRES = ['beta', 'sigma', 'gamma', 'Lambda', 'mu', 'alpha', 'phi', 'I0']


def extraer_posterior_combinado(fit, nombres=NOMBRES):
    """Devuelve matriz (muestras_total, num_params) con todas las cadenas concatenadas."""
    sv = fit.stan_variables()
    return np.column_stack([sv[var] for var in nombres])


def extraer_posterior_por_cadena(fit, nombres=NOMBRES):
    """Devuelve lista de arrays, una por cadena, con las muestras."""
    df = fit.draws_pd(vars=nombres)
    n_chains = fit.chains
    n_draws_total = len(df)
    n_draws_per_chain = n_draws_total // n_chains

    cadenas_idx = np.repeat(np.arange(1, n_chains + 1), n_draws_per_chain)
    draws_idx = np.tile(np.arange(1, n_draws_per_chain + 1), n_chains)
    df.index = pd.MultiIndex.from_arrays([cadenas_idx, draws_idx], names=['chain', 'draw'])

    cadenas = []
    for c in range(1, n_chains + 1):
        cadenas.append(df.xs(c, level='chain').to_numpy())
    return cadenas



# Gráficas 
def guardar_trazas(cadenas, outdir):
    os.makedirs(outdir, exist_ok=True)
    n_param = len(NOMBRES)
    fig, axes = plt.subplots(n_param, 1, figsize=(12, 3 * n_param), sharex=True)

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
    fig, axes = plt.subplots(len(NOMBRES), 1, figsize=(10, 3 * len(NOMBRES)))
    for i, ax in enumerate(axes):
        plot_acf(posterior[:, i], lags=50, ax=ax)
    plt.tight_layout()
    plt.savefig(f"{outdir}/acf.png")
    plt.close()


def graficar_ajuste_incidencia(fit, y_obs, t_obs, N0, E0, outdir,
                               h_solver=0.01, nsamples=20):
    """
    Dibuja curvas de incidencia posterior 
    y las compara con los datos observados.

    Parámetros:
        fit       : objeto CmdStanPy
        y_obs     : array con casos observados por día (longitud T)
        t_obs     : array de tiempos (incluye t=0, longitud T+1)
        N0        : población total inicial (escalar)
        E0        : expuestos iniciales (fijo, ej. 0)
        outdir    : directorio para guardar la figura
        h_solver  : paso de integración 
        nsamples  : número de curvas posteriores a dibujar
    """
    posterior = extraer_posterior_combinado(fit)  # (muestras, 8)
    idx = np.random.choice(posterior.shape[0], size=nsamples, replace=False)

    dias = t_obs[1:]  # tiempos donde se observan casos (día 1,2,...)
    plt.figure(figsize=(12, 6))

    for ind in idx:
        beta, sigma, gamma, Lambda, mu, alpha, phi, I0 = posterior[ind, :]
        try:
            # Resolver el modelo fraccionario SEIR
            res = resolver_seirv(
                beta=beta, sigma=sigma, gamma=gamma,
                mu=mu, Lambda=Lambda, alpha=alpha,
                N0=N0, I0=I0, E0=E0,
                T=t_obs[-1], h=h_solver, t_observacion=t_obs
            )
            # Se espera que la solución incluya S (susceptibles) en la primera columna
            # Ajusta según lo que realmente devuelva tu resolver_seirv
            S_obs = res['y_obs'][:, 0]   # asumiendo que es un diccionario con clave 'y_obs'
            inc = np.maximum(S_obs[:-1] - S_obs[1:], 1e-8)
            plt.plot(dias, inc, color='blue', alpha=0.15, linewidth=0.8)
        except Exception as e:
            print(f"Error con muestra {ind}: {e}")
            continue

    # Curva media posterior
    theta_media = np.mean(posterior, axis=0)
    beta_m, sigma_m, gamma_m, Lambda_m, mu_m, alpha_m, phi_m, I0_m = theta_media
    try:
        res_media = resolver_seirv(
            beta=beta_m, sigma=sigma_m, gamma=gamma_m,
            mu=mu_m, Lambda=Lambda_m, alpha=alpha_m,
            N0=N0, I0=I0_m, E0=E0,
            T=t_obs[-1], h=h_solver, t_observacion=t_obs
        )
        S_media = res_media['y_obs'][:, 0]
        inc_media = np.maximum(S_media[:-1] - S_media[1:], 1e-8)
        plt.plot(dias, inc_media, 'r-', linewidth=2.5, label='Media posterior')
    except Exception as e:
        print(f"Error al graficar media: {e}")

    plt.scatter(dias, y_obs, color='black', s=20, label='Datos observados', zorder=5)
    plt.xlabel('Día')
    plt.ylabel('Casos nuevos')
    plt.title(f'Ajuste – {nsamples} curvas posteriores')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{outdir}/ajuste_incidencia.png")
    plt.close()


def graficar_dispersiones(posterior, outdir):
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
            if i == n - 1:
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
    """
    Compara la distribución a posteriori con las densidades a priori
    para cada parámetro (usando las mismas distribuciones que en el Stan).
    """
    os.makedirs(outdir, exist_ok=True)

    # Extraer muestras posteriores
    beta_post = fit.stan_variable('beta')
    sigma_post = fit.stan_variable('sigma')
    gamma_post = fit.stan_variable('gamma')
    Lambda_post = fit.stan_variable('Lambda')
    mu_post = fit.stan_variable('mu')
    I0_post = fit.stan_variable('I0')
    alpha_post = fit.stan_variable('alpha')
    phi_post = fit.stan_variable('phi')

    # Diccionario: nombre -> (muestras, función de densidad prior, etiqueta)
    params = {
        "beta":   (beta_post,    lambda x: stats.lognorm.pdf(x, 0.3, scale=np.exp(np.log(1.5))),  r"$\beta$"),
        "sigma":  (sigma_post,   lambda x: stats.lognorm.pdf(x, 0.3, scale=np.exp(np.log(1.2))),  r"$\sigma$"),
        "gamma":  (gamma_post,   lambda x: stats.lognorm.pdf(x, 0.2, scale=np.exp(np.log(0.8))),  r"$\gamma$"),
        "Lambda": (Lambda_post,  lambda x: stats.lognorm.pdf(x, 0.5, scale=np.exp(np.log(30))),   r"$\Lambda$"),
        "mu":     (mu_post,      lambda x: stats.lognorm.pdf(x, 0.5, scale=np.exp(np.log(0.001))), r"$\mu$"),
        "I0":     (I0_post,      lambda x: stats.lognorm.pdf(x, 2.0, scale=np.exp(np.log(15))),   r"$I_0$"),
        "alpha":  (alpha_post,   lambda x: stats.beta.pdf(x, 2, 2),                               r"$\alpha$"),
        "phi":    (phi_post,     lambda x: stats.lognorm.pdf(x, 2.0, scale=np.exp(np.log(30))),   r"$\phi$")
    }

    n = len(params)
    rows = (n + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(14, 4 * rows))
    axes = axes.flatten()

    for i, (name, (post_samples, prior_pdf, label)) in enumerate(params.items()):
        ax = axes[i]
        # Histograma posterior
        ax.hist(post_samples, bins=60, density=True, alpha=0.6, label="Posterior", color='skyblue')
        # Curva prior
        x_min, x_max = ax.get_xlim()
        x = np.linspace(x_min, x_max, 500)
        ax.plot(x, prior_pdf(x), lw=2.5, color='crimson', label="Prior")
        ax.set_title(label)
        ax.legend()

    if len(axes) > n:
        fig.delaxes(axes[-1])

    plt.tight_layout()
    plt.savefig(f"{outdir}/prior_vs_posterior.png")
    plt.close()


def resumen_numerico(posterior):
    """Devuelve DataFrame con media e intervalos creíbles al 95%."""
    res = {}
    for i, nom in enumerate(NOMBRES):
        vals = posterior[:, i]
        res[nom] = [np.mean(vals), np.percentile(vals, 2.5), np.percentile(vals, 97.5)]
    df = pd.DataFrame(res, index=['mean', '2.5%', '97.5%']).T
    return df