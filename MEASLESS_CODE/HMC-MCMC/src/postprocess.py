# postprocess.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
import scipy.stats as stats
import math
import os

# Nombres de parámetros 
NOMBRES = ["beta", "I0", "alpha", "p", "theta", "gamma", "kappa", "R0"]

# ── Solver SEIR fraccionario ──────────────────────────────
def seir_rhs(t, Y, beta, gamma, kappa, N):
    S, E, I, R = Y
    dS = -beta * S * I / N
    dE =  beta * S * I / N - kappa * E
    dI =  kappa * E - gamma * I
    dR =  gamma * I
    return np.array([dS, dE, dI, dR])

def solve_seir_frac(alpha, beta, gamma, kappa, N, I0, E0, t_eval, h=0.05):
    """Resuelve SEIR fraccionario y devuelve S, E, I, R en t_eval."""
    T = t_eval[-1]
    n_steps = int(round(T / h))
    t_fine = np.linspace(0, T, n_steps + 1)

    dim = 4
    y = np.zeros((n_steps + 1, dim))
    fvals = np.zeros((n_steps + 1, dim))

    S0 = N - I0 - E0
    y[0] = np.array([S0, E0, I0, 0.0])
    fvals[0] = seir_rhs(0.0, y[0], beta, gamma, kappa, N)

    G1 = math.gamma(alpha + 1)
    G2 = math.gamma(alpha + 2)
    h_alpha = h ** alpha

    for n in range(1, n_steps + 1):
        b = np.array([(n - j) ** alpha - (n - j - 1) ** alpha for j in range(n)])
        y_pred = y[0] + (h_alpha / G1) * np.dot(b, fvals[:n])
        y_pred = np.maximum(y_pred, 0)
        f_pred = seir_rhs(t_fine[n], y_pred, beta, gamma, kappa, N)

        a = np.zeros(n + 1)
        a[0] = (n - 1) ** (alpha + 1) - (n - 1 - alpha) * n ** alpha
        for j in range(1, n):
            a[j] = ((n - j + 1) ** (alpha + 1) + (n - j - 1) ** (alpha + 1)
                    - 2 * (n - j) ** (alpha + 1))
        a[n] = 1.0

        y[n] = y[0] + (h_alpha / G2) * (np.dot(a[:n], fvals[:n]) + a[n] * f_pred)
        y[n] = np.maximum(y[n], 0)
        fvals[n] = seir_rhs(t_fine[n], y[n], beta, gamma, kappa, N)

    idx = np.clip(np.round(t_eval / h).astype(int), 0, n_steps)
    return y[idx]

# ── Extracción de posteriores ──────────────────────────────
def extraer_posterior_combinado(fit, nombres=NOMBRES):
    sv = fit.stan_variables()
    return np.column_stack([sv[var] for var in nombres])

def extraer_posterior_por_cadena(fit, nombres=NOMBRES):
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

# ── Cálculo de MAP marginales ──────────────────────────────
def calc_map_marginal(samples, name):
    """Calcula la moda de la distribución marginal vía KDE"""
    if name == 'alpha':
        eps = 1e-10
        s = np.clip(samples, eps, 1-eps)
        s_trans = np.log(s / (1 - s))
        kde = stats.gaussian_kde(s_trans)
        x_trans = np.linspace(min(s_trans), max(s_trans), 1000)
        dens = kde(x_trans)
        map_trans = x_trans[np.argmax(dens)]
        return 1 / (1 + np.exp(-map_trans))
    else:
        s_log = np.log(samples)
        kde = stats.gaussian_kde(s_log)
        x_log = np.linspace(min(s_log), max(s_log), 1000)
        dens = kde(x_log)
        map_log = x_log[np.argmax(dens)]
        return np.exp(map_log)

def obtener_maps_marginales(posterior):
    """Calcula los MAP marginales para todos los parámetros"""
    maps = {}
    for i, nombre in enumerate(NOMBRES[:7]):  # Excluir R0 que es derivado
        maps[nombre] = calc_map_marginal(posterior[:, i], nombre)
    return maps

# ── Gráficos ─────────────────────────────────────────────
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
    plt.savefig(os.path.join(outdir, 'trazas.png'))
    plt.close()

def guardar_autocorrelacion(posterior, outdir):
    os.makedirs(outdir, exist_ok=True)
    fig, axes = plt.subplots(len(NOMBRES), 1, figsize=(10, 3 * len(NOMBRES)))
    for i, ax in enumerate(axes):
        plot_acf(posterior[:, i], lags=50, ax=ax)
        ax.set_ylabel(NOMBRES[i])
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'acf.png'))
    plt.close()

def graficar_ajuste_desde_stan(fit, y_obs, outdir, nsamples=50, alpha_ci=0.95):
    """
    Estilo minimalista: líneas grises muy tenues para muestras,
    línea negra gruesa para la media posterior, puntos negros para datos.
    """
    mu_samples = fit.stan_variable('mu')
    semanas = np.arange(1, len(y_obs) + 1)
    
    # Calcular MAP (máximo a posteriori) - media en este caso
    mu_map = np.mean(mu_samples, axis=0)
    
    # Calcular bandas de credibilidad
    lower_ci = np.percentile(mu_samples, 100 * (1 - alpha_ci) / 2, axis=0)
    upper_ci = np.percentile(mu_samples, 100 * (1 + alpha_ci) / 2, axis=0)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Capa 1: Bandas de credibilidad (gris muy sutil)
    ax.fill_between(semanas, lower_ci, upper_ci, 
                     color='gray', alpha=0.12, 
                     label=f'IC {int(100*alpha_ci)}%')
    
    # Capa 2: Muestras individuales (gris muy claro, apenas visibles)
    idx = np.random.choice(mu_samples.shape[0], size=min(nsamples, mu_samples.shape[0]), 
                          replace=False)
    for i in idx:
        ax.plot(semanas, mu_samples[i, :], color='gray', alpha=0.15, 
                linewidth=0.5, zorder=1)
    
    # Capa 3: MAP (negro, línea principal)
    ax.plot(semanas, mu_map, 'k-', linewidth=2.0, 
            label='Media posterior', zorder=3)
    
    # Capa 4: Datos observados (puntos negros)
    ax.scatter(semanas, y_obs, color='black', s=25, 
              zorder=5, label='Datos observados', 
              edgecolors='white', linewidth=0.5)
    
    # Estilo minimalista
    ax.set_xlabel('Semana', fontsize=11, style='italic')
    ax.set_ylabel('Casos reportados', fontsize=11, style='italic')
    ax.set_title('Ajuste del modelo SEIR fraccionario', 
                 fontsize=12, fontweight='normal', style='italic')
    
    # Límites ajustados
    ax.set_xlim(0.5, len(y_obs) + 0.5)
    y_max = max(np.max(y_obs), np.max(upper_ci)) * 1.08
    ax.set_ylim(-y_max * 0.02, y_max)
    
    # Grid muy sutil
    ax.grid(True, alpha=0.15, linestyle='-', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Leyenda minimalista
    legend = ax.legend(loc='upper right', frameon=False, 
                      fontsize=9, handlelength=1.5)
    for text in legend.get_texts():
        text.set_style('italic')
    
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'ajuste_incidencia.png'), 
                dpi=300, facecolor='white', bbox_inches='tight')
    plt.close()

def guardar_prior_vs_posterior(fit, outdir):
    """
    Comparación prior-posterior con MAP marginales.
    Estilo minimalista.
    """
    os.makedirs(outdir, exist_ok=True)
    posterior = extraer_posterior_combinado(fit)
    
    # Calcular MAP marginales
    maps_marginales = obtener_maps_marginales(posterior)
    
    # Imprimir MAP marginales
    print("=" * 70)
    print("MAP MARGINALES")
    print("=" * 70)
    print(f"{'Parámetro':<10} {'MAP marginal':<18}")
    print("-" * 70)
    for nombre, valor in maps_marginales.items():
        print(f"{nombre:<10} {valor:<18.6f}")
    
    # FUNCIONES DE PRIOR ACTUALIZADAS
    def prior_beta(x):
        # beta ~ LogNormal(log(10), 0.5) -> Median is 10
        return stats.lognorm.pdf(x, s=0.5, scale=np.exp(np.log(10)))

    def prior_I0(x):
        # I0 ~ LogNormal(log(500), 1.5) -> Median is 500
        return stats.lognorm.pdf(x, s=1.5, scale=np.exp(np.log(500)))

    def prior_alpha(x):
        # alpha ~ Beta(1, 1) = Uniforme
        return stats.beta.pdf(x, a=1, b=1)

    def prior_p(x):
        # p ~ LogNormal(log(0.1), 0.5) -> Median is 0.1
        return stats.lognorm.pdf(x, s=0.5, scale=np.exp(np.log(0.1)))

    def prior_theta(x):
        # theta ~ LogNormal(log(30), 0.5) -> Median is 30
        return stats.lognorm.pdf(x, s=0.5, scale=np.exp(np.log(30)))

    def prior_gamma(x):
        # gamma ~ LogNormal(log(0.8), 0.1) -> Median is 0.8
        return stats.lognorm.pdf(x, s=0.1, scale=np.exp(np.log(0.8)))

    def prior_kappa(x):
        # kappa ~ LogNormal(log(0.4), 0.2) -> Median is 0.4
        return stats.lognorm.pdf(x, s=0.2, scale=np.exp(np.log(0.4)))
    
    # RANGOS MANUALES
    rangos_manuales = {
        'beta':  (0.1, 15.0),      
        'I0':    (0.0, 10000.0),   
        'alpha': (0.0, 1.0),       
        'p':     (0.0, 0.3),     
        'theta': (0.0, 100.0),      
        'gamma': (0.0, 1.5),       
        'kappa': (0.0, 1.5)        
    }
    
    # DICCIONARIO DE PARÁMETROS
    params_dict = {
        'beta':  (posterior[:, 0], prior_beta,  maps_marginales['beta'],  r'$\beta$',  'Tasa de transmisión'),
        'I0':    (posterior[:, 1], prior_I0,    maps_marginales['I0'],    r'$I_0$',    'Infectados iniciales'),
        'alpha': (posterior[:, 2], prior_alpha, maps_marginales['alpha'], r'$\alpha$',  'Orden fraccionario'),
        'p':     (posterior[:, 3], prior_p,     maps_marginales['p'],     r'$p$',       'Tasa de detección'),
        'theta': (posterior[:, 4], prior_theta, maps_marginales['theta'], r'$\theta$',  'Sobredispersión'),
        'gamma': (posterior[:, 5], prior_gamma, maps_marginales['gamma'], r'$\gamma$',  'Tasa de recuperación'),
        'kappa': (posterior[:, 6], prior_kappa, maps_marginales['kappa'], r'$\kappa$',  'Tasa de latencia')
    }
    
    # CREAR FIGURA
    n_params = len(params_dict)
    cols = 2
    rows = (n_params + 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(14, 4.5 * rows))
    axes = axes.flatten()
    
    for i, (name, (post, prior_pdf, map_marg, label, desc)) in enumerate(params_dict.items()):
        ax = axes[i]
        
        x_min, x_max = rangos_manuales[name]
        
        media_val = np.mean(post)
        mediana_val = np.median(post)
        ci_lower = np.percentile(post, 2.5)
        ci_upper = np.percentile(post, 97.5)
        
        x = np.linspace(x_min, x_max, 500)
        
        # Histograma del posterior
        ax.hist(post, bins=50, density=True, alpha=0.30, 
                color='gray', edgecolor='white', linewidth=0.3,
                range=(x_min, x_max))
        
        # Prior
        prior_vals = prior_pdf(x)
        ax.plot(x, prior_vals, 'k--', linewidth=1.5, alpha=0.7, label='Prior')
        
        # MAP marginal
        ax.axvline(map_marg, color='black', linestyle='-', 
                  linewidth=2.0, alpha=0.9, label='MAP marginal')
        
        # Media
        ax.axvline(media_val, color='gray', linestyle='-.', 
                  linewidth=0.8, alpha=0.5, label='Media')
        
        # Anotaciones
        info_text = (
            f'MAP:       {map_marg:.3g}\n'
            f'Media:     {media_val:.3g}\n'
            f'Mediana:   {mediana_val:.3g}\n'
            f'IC 95%:    [{ci_lower:.3g}, {ci_upper:.3g}]'
        )
        
        ax.annotate(info_text, 
                   xy=(0.98, 0.97), xycoords='axes fraction',
                   fontsize=6.5, style='italic',
                   ha='right', va='top',
                   family='monospace',
                   bbox=dict(boxstyle='round,pad=0.3', 
                           facecolor='white', alpha=0.85, 
                           edgecolor='gray', linewidth=0.5))
        
        ax.set_xlabel(label, fontsize=11, style='italic')
        ax.set_ylabel('Densidad', fontsize=9, style='italic')
        ax.set_title(desc, fontsize=10, fontweight='normal', style='italic')
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
    
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.suptitle('Actualización Bayesiana: Prior → Posterior\n'
                 'MAP marginal (—), Media (-·-)',
                fontsize=13, style='italic', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'prior_vs_posterior.png'), 
                dpi=300, facecolor='white', bbox_inches='tight')
    plt.close()

    print("\n" + "=" * 70)
    print("RANGOS MANUALES UTILIZADOS EN LOS GRÁFICOS")
    print("=" * 70)
    for nombre, (xmin, xmax) in rangos_manuales.items():
        print(f"  {nombre:<10} [{xmin:<10.4f}, {xmax:<10.4f}]")
    
    return maps_marginales

def graficar_perfil_epidemia(fit, y_obs, outdir):
    """
    Perfil temporal de la epidemia con estilo minimalista.
    Compara trayectoria MAP con datos observados.
    """
    mu_samples = fit.stan_variable('mu')
    semanas = np.arange(1, len(y_obs) + 1)
    
    # Calcular MAP y bandas
    mu_map = np.mean(mu_samples, axis=0)
    lower_95 = np.percentile(mu_samples, 2.5, axis=0)
    upper_95 = np.percentile(mu_samples, 97.5, axis=0)
    lower_50 = np.percentile(mu_samples, 25, axis=0)
    upper_50 = np.percentile(mu_samples, 75, axis=0)
    
    # Calcular residuos
    residuals = y_obs - mu_map
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), 
                                    gridspec_kw={'height_ratios': [3, 1]})
    
    # Panel principal: ajuste
    ax1.fill_between(semanas, lower_95, upper_95, 
                     color='gray', alpha=0.1)
    ax1.fill_between(semanas, lower_50, upper_50, 
                     color='gray', alpha=0.15)
    ax1.plot(semanas, mu_map, 'k-', linewidth=2, label='Trayectoria MAP')
    ax1.scatter(semanas, y_obs, color='black', s=30, 
               zorder=5, label='Observado')
    
    ax1.set_ylabel('Casos reportados', fontsize=11, style='italic')
    ax1.set_title('Perfil temporal de la epidemia', 
                 fontsize=12, fontweight='normal', style='italic')
    ax1.grid(True, alpha=0.15, linewidth=0.5)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.legend(frameon=False, fontsize=9).get_texts()[0].set_style('italic')
    
    # Panel inferior: residuos
    ax2.bar(semanas, residuals, color='gray', alpha=0.3, 
           edgecolor='black', linewidth=0.3)
    ax2.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
    ax2.set_xlabel('Semana', fontsize=11, style='italic')
    ax2.set_ylabel('Residuos', fontsize=11, style='italic')
    ax2.grid(True, alpha=0.15, linewidth=0.5)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'perfil_epidemia.png'), 
                dpi=300, facecolor='white', bbox_inches='tight')
    plt.close()


def resumen_numerico(posterior):
    res = {}
    for i, nom in enumerate(NOMBRES):
        vals = posterior[:, i]
        res[nom] = [np.mean(vals), np.percentile(vals, 2.5), np.percentile(vals, 97.5)]
    df = pd.DataFrame(res, index=['mean', '2.5%', '97.5%']).T
    return df