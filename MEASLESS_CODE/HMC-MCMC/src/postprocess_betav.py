# postprocess_betav.py
# Propósito: 
#Funciones de post-procesamiento para el modelo SEIR fraccionario
#            con beta(t) variable. Incluye extracción de posteriores,
#            gráficos de diagnóstico, ajuste, perfil temporal, evolución de
#            beta y distribución predictiva.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
import scipy.stats as stats
import math
import os


#  PARÁMETROS
# Lista de parámetros
NOMBRES_PRIOR_POST = ["beta0", "delta", "I0", "alpha", "p", "theta", "gamma", "kappa"]

NOMBRES_COMPLETOS = NOMBRES_PRIOR_POST  # Solo los 8 parámetros

# Alias para compatibilidad
NOMBRES = NOMBRES_COMPLETOS


# SOLVERS  (SEIR FRACCIONARIO EN PYTHON)
# Estas funciones replican el solver del modelo Stan en Python, útiles para
# depuración o para generar soluciones con parámetros fijos (no se usan en
# el flujo principal de análisis de la cadena MCMC).

def seir_rhs_beta_var(t, Y, beta0, delta, gamma, kappa, N):
    """
    Ecuaciones diferenciales del modelo SEIR con beta(t) = beta0 * exp(-delta * t).
    """
    S, E, I, R = Y
    beta_t = beta0 * np.exp(-delta * t)
    dS = -beta_t * S * I / N
    dE =  beta_t * S * I / N - kappa * E
    dI =  kappa * E - gamma * I
    dR =  gamma * I
    return np.array([dS, dE, dI, dR])


def solve_seir_frac_beta_var(alpha, beta0, delta, gamma, kappa, N, I0, E0, t_eval, h=0.05):
    """
    Resuelve el sistema SEIR fraccionario con el método Predictor-Corrector.
    Devuelve las trayectorias de S, E, I, R en los tiempos especificados.
    """
    T = t_eval[-1]
    n_steps = int(round(T / h))
    t_fine = np.linspace(0, T, n_steps + 1)

    dim = 4
    y = np.zeros((n_steps + 1, dim))
    fvals = np.zeros((n_steps + 1, dim))

    # Condiciones iniciales
    S0 = N - I0 - E0
    y[0] = np.array([S0, E0, I0, 0.0])
    beta_t0 = beta0 * np.exp(-delta * 0.0)
    fvals[0] = seir_rhs_beta_var(0.0, y[0], beta0, delta, gamma, kappa, N)

    G1 = math.gamma(alpha + 1)
    G2 = math.gamma(alpha + 2)
    h_alpha = h ** alpha

    for n in range(1, n_steps + 1):
        t_current = t_fine[n]

        # Predictor
        b = np.array([(n - j) ** alpha - (n - j - 1) ** alpha for j in range(n)])
        y_pred = y[0] + (h_alpha / G1) * np.dot(b, fvals[:n])
        y_pred = np.maximum(y_pred, 0)
        f_pred = seir_rhs_beta_var(t_current, y_pred, beta0, delta, gamma, kappa, N)

        # Corrector
        a = np.zeros(n + 1)
        a[0] = (n - 1) ** (alpha + 1) - (n - 1 - alpha) * n ** alpha
        for j in range(1, n):
            a[j] = ((n - j + 1) ** (alpha + 1) + (n - j - 1) ** (alpha + 1)
                    - 2 * (n - j) ** (alpha + 1))
        a[n] = 1.0

        y[n] = y[0] + (h_alpha / G2) * (np.dot(a[:n], fvals[:n]) + a[n] * f_pred)
        y[n] = np.maximum(y[n], 0)
        fvals[n] = seir_rhs_beta_var(t_current, y[n], beta0, delta, gamma, kappa, N)

    idx = np.clip(np.round(t_eval / h).astype(int), 0, n_steps)
    return y[idx]


# eXTRACCIÓN DE POSTERIORES (MUESTRAS DE STAN)
# Estas funciones extraen las muestras de la cadena MCMC y las organizan
# en arrays de NumPy para su posterior análisis.

def extraer_posterior_combinado(fit, nombres=None):
    """
    Extrae las muestras de los parámetros especificados y las combina
    en una matriz de tamaño (n_draws, n_params).
    """
    if nombres is None:
        nombres = NOMBRES_PRIOR_POST
    sv = fit.stan_variables()
    return np.column_stack([sv[var] for var in nombres])


def extraer_posterior_por_cadena(fit, nombres=None):
    """
    Extrae las muestras separadas por cadena. Útil para gráficos de trazas.
    Devuelve una lista de arrays, uno por cadena.
    """
    if nombres is None:
        nombres = NOMBRES_PRIOR_POST

    df = fit.draws_pd(vars=nombres)
    n_chains = fit.chains

    # Si el índice es MultiIndex (chain, draw) se usa directamente.
    if isinstance(df.index, pd.MultiIndex):
        cadenas = []
        for c in range(1, n_chains + 1):
            cadenas.append(df.xs(c, level='chain').to_numpy())
    else:
        # Si es un índice plano, se divide en partes iguales.
        n_draws_total = len(df)
        n_draws_per_chain = n_draws_total // n_chains
        cadenas = []
        for c in range(n_chains):
            inicio = c * n_draws_per_chain
            fin = inicio + n_draws_per_chain
            cadenas.append(df.iloc[inicio:fin].to_numpy())

    return cadenas


# CÁLCULO DE MAP MARGINALES -MODA DE LA POSTERIOR
# Se usa estimación de densidad por Kernel (KDE) para encontrar la moda
# de la distribución marginal de cada parámetro.

def calc_map_marginal(samples, name):
    """
    Calcula la moda de la distribución marginal usando KDE.
    Para 'alpha' se transforma con logit para respetar su soporte (0,1).
    Para el resto se usa log para estabilizar la varianza.
    """
    if name == 'alpha':
        eps = 1e-10
        s = np.clip(samples, eps, 1 - eps)
        s_trans = np.log(s / (1 - s))
        kde = stats.gaussian_kde(s_trans)
        x_trans = np.linspace(min(s_trans), max(s_trans), 1000)
        dens = kde(x_trans)
        map_trans = x_trans[np.argmax(dens)]
        return 1 / (1 + np.exp(-map_trans))
    else:
        eps = 1e-10
        s_log = np.log(np.clip(samples, eps, None))
        kde = stats.gaussian_kde(s_log)
        x_log = np.linspace(min(s_log), max(s_log), 1000)
        dens = kde(x_log)
        map_log = x_log[np.argmax(dens)]
        return np.exp(map_log)


def obtener_maps_marginales(posterior, nombres):
    """
    Calcula los MAP marginales para todos los parámetros.
    Devuelve un diccionario {nombre: valor}.
    """
    maps = {}
    for i, nombre in enumerate(nombres):
        maps[nombre] = calc_map_marginal(posterior[:, i], nombre)
    return maps


# GRÁFICOS DE DIAGNÓSTICO (TRAZAS Y AUTOCORRELACIÓN)

def guardar_trazas(cadenas, outdir, nombres=None):
    """
    Genera un gráfico con las trazas de las cadenas para cada parámetro.
    """
    if nombres is None:
        nombres = NOMBRES_PRIOR_POST
    os.makedirs(outdir, exist_ok=True)
    n_param = len(nombres)
    fig, axes = plt.subplots(n_param, 1, figsize=(12, 3 * n_param), sharex=True)
    if n_param == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        for cad in cadenas:
            ax.plot(cad[:, i], alpha=0.6, lw=0.5)
        ax.set_ylabel(nombres[i])
    axes[-1].set_xlabel('Iteración')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'trazas.png'))
    plt.close()


def guardar_autocorrelacion(posterior, outdir, nombres=None):
    """
    Genera gráficos de autocorrelación para cada parámetro.
    """
    if nombres is None:
        nombres = NOMBRES_PRIOR_POST
    os.makedirs(outdir, exist_ok=True)
    n_param = posterior.shape[1]
    fig, axes = plt.subplots(n_param, 1, figsize=(10, 3 * n_param))
    if n_param == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        plot_acf(posterior[:, i], lags=50, ax=ax)
        ax.set_ylabel(nombres[i] if i < len(nombres) else f'Var{i}')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'acf.png'))
    plt.close()


#  GRÁFICOS DE AJUSTE Y PERFIL TEMPORAL

def graficar_ajuste_desde_stan(fit, y_obs, outdir, nsamples=50, alpha_ci=0.95):
    """
    Grafica el ajuste del modelo a los datos de entrenamiento.
    Muestra la media posterior de μ, su IC y las trayectorias individuales.
    Solo utiliza las primeras T semanas (entrenamiento).
    """
    mu_samples = fit.stan_variable('mu')
    T_train = len(y_obs)
    mu_samples = mu_samples[:, :T_train]  # solo primeras T
    semanas = np.arange(1, T_train + 1)

    mu_map = np.mean(mu_samples, axis=0)
    lower_ci = np.percentile(mu_samples, 100 * (1 - alpha_ci) / 2, axis=0)
    upper_ci = np.percentile(mu_samples, 100 * (1 + alpha_ci) / 2, axis=0)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.fill_between(semanas, lower_ci, upper_ci,
                    color='gray', alpha=0.12,
                    label=f'IC {int(100*alpha_ci)}%')

    idx = np.random.choice(mu_samples.shape[0],
                           size=min(nsamples, mu_samples.shape[0]),
                           replace=False)
    for i in idx:
        ax.plot(semanas, mu_samples[i, :], color='gray', alpha=0.15,
                linewidth=0.5, zorder=1)

    ax.plot(semanas, mu_map, 'k-', linewidth=2.0,
            label='Media posterior', zorder=3)

    ax.scatter(semanas, y_obs, color='black', s=25,
               zorder=5, label='Datos observados',
               edgecolors='white', linewidth=0.5)

    ax.set_xlabel('Semana', fontsize=11, style='italic')
    ax.set_ylabel('Casos reportados', fontsize=11, style='italic')
    ax.set_title('Ajuste del modelo SEIR fraccionario con $\\beta(t)$',
                 fontsize=12, fontweight='normal', style='italic')
    ax.set_xlim(0.5, len(y_obs) + 0.5)
    y_max = max(np.max(y_obs), np.max(upper_ci)) * 1.08
    ax.set_ylim(-y_max * 0.02, y_max)
    ax.grid(True, alpha=0.15, linestyle='-', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    legend = ax.legend(loc='upper right', frameon=False,
                       fontsize=9, handlelength=1.5)
    for text in legend.get_texts():
        text.set_style('italic')

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'ajuste_incidencia_beta_var.png'),
                dpi=300, facecolor='white', bbox_inches='tight')
    plt.close()


def graficar_perfil_epidemia(fit, y_obs, outdir):
    """
    Perfil temporal con residuos estandarizados (solo entrenamiento).
    """
    mu_samples = fit.stan_variable('mu')
    T_train = len(y_obs)
    mu_samples_train = mu_samples[:, :T_train]
    semanas = np.arange(1, T_train + 1)

    mu_map = np.mean(mu_samples_train, axis=0)
    lower_95 = np.percentile(mu_samples_train, 2.5, axis=0)
    upper_95 = np.percentile(mu_samples_train, 97.5, axis=0)
    lower_50 = np.percentile(mu_samples_train, 25, axis=0)
    upper_50 = np.percentile(mu_samples_train, 75, axis=0)

    residuals = y_obs - mu_map

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7),
                                   gridspec_kw={'height_ratios': [3, 1]})

    ax1.fill_between(semanas, lower_95, upper_95, color='gray', alpha=0.1)
    ax1.fill_between(semanas, lower_50, upper_50, color='gray', alpha=0.15)
    ax1.plot(semanas, mu_map, 'k-', linewidth=2, label='Media posterior')
    ax1.scatter(semanas, y_obs, color='black', s=30, zorder=5, label='Observado')

    ax1.set_ylabel('Casos reportados', fontsize=11, style='italic')
    ax1.set_title('Perfil temporal - Modelo con $\\beta(t)$',
                  fontsize=12, fontweight='normal', style='italic')
    ax1.grid(True, alpha=0.15, linewidth=0.5)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.legend(frameon=False, fontsize=9)

    ax2.bar(semanas, residuals, color='gray', alpha=0.3,
            edgecolor='black', linewidth=0.3)
    ax2.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
    ax2.set_xlabel('Semana', fontsize=11, style='italic')
    ax2.set_ylabel('Residuos', fontsize=11, style='italic')
    ax2.grid(True, alpha=0.15, linewidth=0.5)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'perfil_epidemia_beta_var.png'),
                dpi=300, facecolor='white', bbox_inches='tight')
    plt.close()


def graficar_evolucion_beta(fit, outdir):
    """
    Grafica la evolución temporal de beta(t) = beta0 * exp(-delta * t)
    con bandas de credibilidad para todo el período (T_total).
    """
    os.makedirs(outdir, exist_ok=True)

    beta0_samples = fit.stan_variable('beta0')
    delta_samples = fit.stan_variable('delta')
    beta_week_samples = fit.stan_variable('beta_week')

    T_total = beta_week_samples.shape[1]
    semanas = np.arange(0, T_total)

    beta_mean = np.mean(beta_week_samples, axis=0)
    beta_lower = np.percentile(beta_week_samples, 2.5, axis=0)
    beta_upper = np.percentile(beta_week_samples, 97.5, axis=0)

    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.fill_between(semanas, beta_lower, beta_upper,
                     color='gray', alpha=0.15, label='IC 95%')
    ax1.plot(semanas, beta_mean, 'k-', linewidth=2.2,
             label=r'Media posterior $\beta(t)$')

    beta0_mean = np.mean(beta0_samples)
    beta0_ci = np.percentile(beta0_samples, [2.5, 97.5])
    delta_mean = np.mean(delta_samples)
    delta_ci = np.percentile(delta_samples, [2.5, 97.5])

    ax1.set_xlabel('Semana', fontsize=11, style='italic')
    ax1.set_ylabel(r'$\beta(t)$', fontsize=12, style='italic')
    ax1.set_title(r'Evolución temporal: $\beta(t) = \beta_0 \, e^{-\delta t}$',
                  fontsize=13, fontweight='normal', style='italic')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(True, alpha=0.12, linestyle='-', linewidth=0.4)
    ax1.tick_params(labelsize=9)

    leg = ax1.legend(loc='upper right', frameon=False, fontsize=9,
                     handlelength=1.5)
    if leg:
        for text in leg.get_texts():
            text.set_style('italic')

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'evolucion_beta_t.png'),
                dpi=300, facecolor='white', bbox_inches='tight')
    plt.close()

    print("\n" + "=" * 60)
    print("EVOLUCIÓN DE β(t)")
    print("=" * 60)
    print(f"β₀  = {beta0_mean:.3f}  [{beta0_ci[0]:.3f}, {beta0_ci[1]:.3f}]")
    print(f"δ   = {delta_mean:.4f}  [{delta_ci[0]:.4f}, {delta_ci[1]:.4f}]")
    print(f"β(0)  = {beta_mean[0]:.3f}")
    print(f"β(T-1) = {beta_mean[-1]:.3f}")


# DISTRIBUCIÓN PREDICTIVA (PPC)

def extraer_predictiva(fit, T_train, T_new=None):
    """
    Extrae las muestras de la distribución predictiva posterior (y_rep).
    Devuelve un diccionario con las muestras completas y separadas
    por período (entrenamiento y futuro).
    """
    y_rep = fit.stan_variable('y_rep')
    n_samples = y_rep.shape[0]
    T_total = y_rep.shape[1]

    result = {
        'y_rep': y_rep,
        'T_train': T_train,
        'T_total': T_total,
        'n_samples': n_samples
    }

    if T_train > 0:
        result['y_rep_train'] = y_rep[:, :T_train]

    if T_new is not None and T_new > 0:
        if T_train + T_new <= T_total:
            result['y_rep_future'] = y_rep[:, T_train:T_train + T_new]
        else:
            result['y_rep_future'] = y_rep[:, T_train:]

    return result


def intervalos_predictivos(y_rep, alpha=0.95):
    """
    Calcula intervalos predictivos (PI) para la variable y_rep.
    Devuelve media, mediana, desviación y percentiles.
    """
    if y_rep.ndim == 1:
        y_rep = y_rep.reshape(-1, 1)

    lower = 100 * (1 - alpha) / 2
    upper = 100 * (1 + alpha) / 2

    return {
        'mean': np.mean(y_rep, axis=0),
        'median': np.median(y_rep, axis=0),
        'std': np.std(y_rep, axis=0),
        f'lower_{int(alpha*100)}': np.percentile(y_rep, lower, axis=0),
        f'upper_{int(alpha*100)}': np.percentile(y_rep, upper, axis=0),
        'lower_50': np.percentile(y_rep, 25, axis=0),
        'upper_50': np.percentile(y_rep, 75, axis=0),
        'lower_80': np.percentile(y_rep, 10, axis=0),
        'upper_80': np.percentile(y_rep, 90, axis=0)
    }


def graficar_predictiva_completa(fit, y_train, y_test=None, outdir=None, alpha=0.95):
    """
    Gráfico simplificado de la distribución predictiva:
    - Ajuste μ con IC (banda gris)
    - Datos observados en círculos (negro = entrenamiento, rojo = validación)
    - Media predictiva (de y_rep) en X rojas
    """
    if outdir is None:
        outdir = 'figures'
    os.makedirs(outdir, exist_ok=True)

    T_train = len(y_train)

    mu = fit.stan_variable('mu')
    y_rep = fit.stan_variable('y_rep')

    T_total_real = mu.shape[1]
    T_new_real = T_total_real - T_train

    # Ajustar y_test al tamaño real
    if y_test is not None:
        if len(y_test) > T_new_real:
            y_test = y_test[:T_new_real]
        elif len(y_test) < T_new_real:
            y_test = np.concatenate([y_test, [np.nan]*(T_new_real - len(y_test))])
    else:
        y_test = np.full(T_new_real, np.nan)

    T_new = len(y_test)
    T_total = T_train + T_new

    # Recortar si es necesario
    if T_total_real > T_total:
        mu = mu[:, :T_total]
        y_rep = y_rep[:, :T_total]
    elif T_total_real < T_total:
        T_total = T_total_real
        T_new = T_total - T_train
        y_test = y_test[:T_new] if len(y_test) > T_new else y_test

    # Estadísticas de μ
    stats_mu = intervalos_predictivos(mu, alpha=alpha)
    mu_mean = stats_mu['mean']
    mu_lower = stats_mu[f'lower_{int(alpha*100)}']
    mu_upper = stats_mu[f'upper_{int(alpha*100)}']

    # Media predictiva (de y_rep)
    pred_mean = np.mean(y_rep, axis=0)

    semanas = np.arange(1, T_total + 1)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Banda de credibilidad para μ
    ax.fill_between(semanas, mu_lower, mu_upper,
                    color='gray', alpha=0.25, label=f'IC {int(alpha*100)}% para μ')

    # Curva de μ
    ax.plot(semanas, mu_mean, 'k-', lw=2, label='Media posterior μ (ajuste)')

    # Datos de entrenamiento
    ax.scatter(semanas[:T_train], y_train[:T_train],
               color='black', s=50, label='Entrenamiento (observado)',
               zorder=5, edgecolors='white', linewidth=0.8)

    # Datos de validación
    if not np.all(np.isnan(y_test)):
        ax.scatter(semanas[T_train:], y_test,
                   color='red', s=50, marker='o', facecolors='none',
                   linewidth=1.5, label='Validación (observado)', zorder=5)

    # Media predictiva (X rojas)
    ax.scatter(semanas, pred_mean,
               color='red', s=60, marker='x', linewidth=2,
               label='Media predictiva (y_rep)', zorder=6)

    ax.axvline(x=T_train + 0.5, color='red', linestyle='--', alpha=0.6,
               label='Fin entrenamiento')

    ax.set_xlabel('Semana', fontsize=12)
    ax.set_ylabel('Casos reportados', fontsize=12)
    ax.set_title('Ajuste y predicción del modelo SEIR fraccionario con β(t) variable',
                 fontsize=13, fontweight='normal')
    ax.legend(loc='upper right', frameon=False, fontsize=10)
    ax.grid(True, alpha=0.15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'predictiva_simple.png'), dpi=300,
                facecolor='white', bbox_inches='tight')
    plt.close()
    print("✅ Gráfico predictivo guardado como 'predictiva_simple.png'")


def analisis_predictivo(fit, y_train, y_test=None, outdir=None, alpha=0.95):
    """
    Función unificada para generar el gráfico predictivo.
    Actualmente solo genera el gráfico; no devuelve métricas.
    """
    if outdir is None:
        outdir = 'figures'
    os.makedirs(outdir, exist_ok=True)

    graficar_predictiva_completa(fit, y_train, y_test, outdir, alpha)
    return None


# PRIOR VS POSTERIOR (MAP MARGINALES)

def guardar_prior_vs_posterior_beta_var(fit, outdir):
    """
    Compara la distribución prior con la posterior para cada parámetro.
    Usa el MAP marginal como estimador puntual.
    """
    os.makedirs(outdir, exist_ok=True)

    posterior = extraer_posterior_combinado(fit, nombres=NOMBRES_PRIOR_POST)
    maps_marginales = obtener_maps_marginales(posterior, NOMBRES_PRIOR_POST)

    # Imprimir MAP marginales
    print("=" * 70)
    print("MAP MARGINALES (Modelo Beta Variable)")
    print("=" * 70)
    print(f"{'Parámetro':<10} {'MAP marginal':<18}")
    print("-" * 70)
    for nombre, valor in maps_marginales.items():
        print(f"{nombre:<10} {valor:<18.6f}")

    # Definir funciones de densidad para los priors
    def prior_beta0(x):
        return stats.lognorm.pdf(x, s=0.5, scale=np.exp(np.log(2.0)))

    def prior_delta(x):
        return stats.lognorm.pdf(x, s=0.5, scale=np.exp(np.log(0.09)))

    def prior_I0(x):
        return stats.lognorm.pdf(x, s=1.0, scale=np.exp(np.log(100)))

    def prior_alpha(x):
        return stats.beta.pdf(x, a=1, b=1)

    def prior_p(x):
        return stats.lognorm.pdf(x, s=0.25, scale=np.exp(np.log(0.5)))

    def prior_theta(x):
        return stats.lognorm.pdf(x, s=0.5, scale=np.exp(np.log(16)))

    def prior_gamma(x):
        return stats.lognorm.pdf(x, s=0.05, scale=np.exp(np.log(0.875)))

    def prior_kappa(x):
        return stats.lognorm.pdf(x, s=0.05, scale=np.exp(np.log(0.5)))

    priors = [prior_beta0, prior_delta, prior_I0, prior_alpha,
              prior_p, prior_theta, prior_gamma, prior_kappa]

    rangos_manuales = {
        'beta0': (0.1, 8.0),
        'delta': (0.0, 0.4),
        'I0':    (0.0, 1500.0),
        'alpha': (0.0, 1.0),
        'p':     (0.0, 1.1),
        'theta': (0.0, 60.0),
        'gamma': (0.0, 1.5),
        'kappa': (0.0, 1.2)
    }

    labels = [r'$\beta_0$', r'$\delta$', r'$I_0$', r'$\alpha$',
              r'$p$', r'$\theta$', r'$\gamma$', r'$\kappa$']
    descs = ['Tasa de transmisión inicial', 'Tasa de decaimiento',
             'Infectados iniciales', 'Orden fraccionario',
             'Tasa de detección', 'Sobredispersión',
             'Tasa de recuperación', 'Tasa de latencia']

    nombres_params = ['beta0', 'delta', 'I0', 'alpha', 'p', 'theta', 'gamma', 'kappa']

    n_params = 8
    cols = 2
    rows = 4

    fig, axes = plt.subplots(rows, cols, figsize=(14, 18))
    axes = axes.flatten()

    for i in range(n_params):
        ax = axes[i]
        name = nombres_params[i]
        x_min, x_max = rangos_manuales[name]

        post = posterior[:, i]
        map_marg = maps_marginales[name]
        media_val = np.mean(post)
        mediana_val = np.median(post)
        ci_lower = np.percentile(post, 2.5)
        ci_upper = np.percentile(post, 97.5)

        x = np.linspace(x_min, x_max, 500)

        ax.hist(post, bins=50, density=True, alpha=0.30,
                color='gray', edgecolor='white', linewidth=0.3,
                range=(x_min, x_max))

        prior_vals = priors[i](x)
        ax.plot(x, prior_vals, 'k--', linewidth=1.5, alpha=0.7, label='Prior')

        ax.axvline(map_marg, color='black', linestyle='-',
                   linewidth=2.0, alpha=0.9, label='MAP marginal')
        ax.axvline(media_val, color='gray', linestyle='-.',
                   linewidth=0.8, alpha=0.5, label='Media')

        info_text = (
            f'MAP:       {map_marg:.3g}\n'
            f'Media:     {media_val:.3g}\n'
            f'Mediana:   {mediana_val:.3g}\n'
            f'IC 95%:    [{ci_lower:.3g}, {ci_upper:.3g}]'
        )

        ax.annotate(info_text,
                    xy=(0.98, 0.97), xycoords='axes fraction',
                    fontsize=6.5, style='italic',
                    ha='right', va='top', family='monospace',
                    bbox=dict(boxstyle='round,pad=0.3',
                              facecolor='white', alpha=0.85,
                              edgecolor='gray', linewidth=0.5))

        ax.set_xlabel(labels[i], fontsize=11, style='italic')
        ax.set_ylabel('Densidad', fontsize=9, style='italic')
        ax.set_title(descs[i], fontsize=10, fontweight='normal', style='italic')
        ax.set_xlim(x_min, x_max)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.5)
        ax.spines['bottom'].set_linewidth(0.5)
        ax.tick_params(labelsize=8, width=0.5)
        ax.grid(True, alpha=0.1, linestyle='-', linewidth=0.3)

        leg = ax.legend(loc='upper left', frameon=False, fontsize=7,
                        handlelength=1.5)
        if leg:
            for text in leg.get_texts():
                text.set_style('italic')

    plt.suptitle('Actualización Bayesiana: Modelo con $\\beta(t)=\\beta_0 e^{-\\delta t}$\n'
                 'MAP marginal (—), Media (-·-)',
                 fontsize=13, style='italic', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'prior_vs_posterior_beta_var.png'),
                dpi=300, facecolor='white', bbox_inches='tight')
    plt.close()

    return maps_marginales


#  RESUMEN NUMÉRICO (TABLA DE ESTADÍSTICOS)

def resumen_numerico(posterior, nombres=None):
    """
    Genera una tabla con media y percentiles 2.5% y 97.5% para cada parámetro.
    """
    if nombres is None:
        nombres = [f'Param_{i}' for i in range(posterior.shape[1])]

    res = {}
    for i, nom in enumerate(nombres):
        vals = posterior[:, i]
        res[nom] = [np.mean(vals), np.percentile(vals, 2.5), np.percentile(vals, 97.5)]

    df = pd.DataFrame(res, index=['mean', '2.5%', '97.5%']).T
    return df