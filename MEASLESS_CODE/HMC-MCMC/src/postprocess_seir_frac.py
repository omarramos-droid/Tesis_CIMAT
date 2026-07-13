# postprocess_betav.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
import scipy.stats as stats
import math
import os

# ============================================================
# NOMBRES DE PARÁMETROS
# ============================================================
NOMBRES_COMPLETOS = ["beta0", "delta", "I0", "alpha", "p", "theta", "gamma", "kappa", 
                     "R0_initial", "R0_final", "beta_mean"]

# Solo los 8 parámetros principales (para gráficos prior/posterior)
NOMBRES_PRIOR_POST = ["beta0", "delta", "I0", "alpha", "p", "theta", "gamma", "kappa"]

# Para compatibilidad con funciones que usan NOMBRES
NOMBRES = NOMBRES_COMPLETOS

# ============================================================
# SOLVER SEIR FRACCIONARIO CON BETA VARIABLE
# ============================================================
def seir_rhs_beta_var(t, Y, beta0, delta, gamma, kappa, N):
    """RHS del SEIR con beta(t) = beta0 * exp(-delta * t)"""
    S, E, I, R = Y
    beta_t = beta0 * np.exp(-delta * t)
    dS = -beta_t * S * I / N
    dE =  beta_t * S * I / N - kappa * E
    dI =  kappa * E - gamma * I
    dR =  gamma * I
    return np.array([dS, dE, dI, dR])

def solve_seir_frac_beta_var(alpha, beta0, delta, gamma, kappa, N, I0, E0, t_eval, h=0.05):
    """
    Resuelve SEIR fraccionario con beta(t) = beta0 * exp(-delta * t).
    Devuelve S, E, I, R en los tiempos t_eval.
    """
    T = t_eval[-1]
    n_steps = int(round(T / h))
    t_fine = np.linspace(0, T, n_steps + 1)

    dim = 4
    y = np.zeros((n_steps + 1, dim))
    fvals = np.zeros((n_steps + 1, dim))

    S0 = N - I0 - E0
    y[0] = np.array([S0, E0, I0, 0.0])
    
    # beta en t=0
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

# También mantenemos el solver antiguo para compatibilidad
def seir_rhs(t, Y, beta, gamma, kappa, N):
    S, E, I, R = Y
    dS = -beta * S * I / N
    dE =  beta * S * I / N - kappa * E
    dI =  kappa * E - gamma * I
    dR =  gamma * I
    return np.array([dS, dE, dI, dR])

def solve_seir_frac(alpha, beta, gamma, kappa, N, I0, E0, t_eval, h=0.05):
    """Resuelve SEIR fraccionario (beta constante)."""
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

# ============================================================
# EXTRACCIÓN DE POSTERIORES
# ============================================================
def extraer_posterior_combinado(fit, nombres=None):
    if nombres is None:
        nombres = NOMBRES_PRIOR_POST
    sv = fit.stan_variables()
    return np.column_stack([sv[var] for var in nombres])

def extraer_posterior_por_cadena(fit, nombres=None):
    """Extrae posterior separado por cadenas - versión robusta."""
    if nombres is None:
        nombres = NOMBRES_PRIOR_POST
    
    df = fit.draws_pd(vars=nombres)
    n_chains = fit.chains
    
    # Verificar si el índice es MultiIndex
    if isinstance(df.index, pd.MultiIndex):
        # Caso: MultiIndex (chain, draw)
        cadenas = []
        for c in range(1, n_chains + 1):
            cadenas.append(df.xs(c, level='chain').to_numpy())
    else:
        # Caso: índice plano - dividir manualmente
        n_draws_total = len(df)
        n_draws_per_chain = n_draws_total // n_chains
        cadenas = []
        for c in range(n_chains):
            inicio = c * n_draws_per_chain
            fin = inicio + n_draws_per_chain
            cadenas.append(df.iloc[inicio:fin].to_numpy())
    
    return cadenas

# ============================================================
# GRÁFICOS DE DIAGNÓSTICO
# ============================================================
def guardar_trazas(cadenas, outdir, nombres=None):
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

# ============================================================
# AJUSTE A DATOS
# ============================================================
def graficar_ajuste_desde_stan(fit, y_obs, outdir, nsamples=50, alpha_ci=0.95):
    """
    Estilo minimalista: líneas grises, media en negro, datos como puntos.
    """
    mu_samples = fit.stan_variable('mu')
    semanas = np.arange(1, len(y_obs) + 1)
    
    mu_map = np.mean(mu_samples, axis=0)
    lower_ci = np.percentile(mu_samples, 100 * (1 - alpha_ci) / 2, axis=0)
    upper_ci = np.percentile(mu_samples, 100 * (1 + alpha_ci) / 2, axis=0)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.fill_between(semanas, lower_ci, upper_ci, 
                     color='gray', alpha=0.12, 
                     label=f'IC {int(100*alpha_ci)}%')
    
    idx = np.random.choice(mu_samples.shape[0], size=min(nsamples, mu_samples.shape[0]), 
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

# ============================================================
# PRIOR VS POSTERIOR (BETA VARIABLE)
# ============================================================
def guardar_prior_vs_posterior_beta_var(fit, outdir):
    """
    Comparación prior-posterior para el modelo con beta(t) variable.
    """
    os.makedirs(outdir, exist_ok=True)
    
    posterior = extraer_posterior_combinado(fit, nombres=NOMBRES_PRIOR_POST)
    
    # MAP conjunto
    try:
        lp = fit.stan_variable('lp__')
        idx_map = np.argmax(lp)
        using_lp = True
        print("Usando lp__ para MAP conjunto")
    except (ValueError, KeyError):
        from scipy.stats import gaussian_kde
        posterior_trans = posterior.copy()
        eps = 1e-10
        alpha_safe = np.clip(posterior[:, 3], eps, 1-eps)
        posterior_trans[:, 3] = np.log(alpha_safe / (1 - alpha_safe))
        for j in [0, 1, 2, 4, 5, 6, 7]:
            posterior_trans[:, j] = np.log(np.clip(posterior[:, j], eps, None))
        kde = gaussian_kde(posterior_trans.T)
        dens = kde(posterior_trans.T)
        idx_map = np.argmax(dens)
        using_lp = False
        print("Usando KDE multivariada para MAP conjunto")
    
    map_vals = [posterior[idx_map, j] for j in range(8)]
    
    # MAP marginales
    from scipy.stats import gaussian_kde
    
    def calc_map_marginal(samples, name):
        if name == 'alpha':
            eps = 1e-10
            s = np.clip(samples, eps, 1-eps)
            s_trans = np.log(s / (1 - s))
            kde = gaussian_kde(s_trans)
            x_trans = np.linspace(min(s_trans), max(s_trans), 1000)
            dens = kde(x_trans)
            map_trans = x_trans[np.argmax(dens)]
            return 1 / (1 + np.exp(-map_trans))
        else:
            eps = 1e-10
            s_log = np.log(np.clip(samples, eps, None))
            kde = gaussian_kde(s_log)
            x_log = np.linspace(min(s_log), max(s_log), 1000)
            dens = kde(x_log)
            map_log = x_log[np.argmax(dens)]
            return np.exp(map_log)
    
    map_marg = [calc_map_marginal(posterior[:, j], 
                ['beta0','delta','I0','alpha','p','theta','gamma','kappa'][j]) 
                for j in range(8)]
    
    print("=" * 70)
    print("MAP CONJUNTO vs MAP MARGINALES (Modelo Beta Variable)")
    print("=" * 70)
    nombres_params = ['beta0', 'delta', 'I0', 'alpha', 'p', 'theta', 'gamma', 'kappa']
    for i, nombre in enumerate(nombres_params):
        diff = abs(map_vals[i] - map_marg[i]) / map_marg[i] * 100 if map_marg[i] != 0 else 0
        print(f"{nombre:<10} MAP conj: {map_vals[i]:<15.6f} MAP marg: {map_marg[i]:<15.6f} Diff: {diff:.2f}%")
    
    # Funciones de prior
    def prior_beta0(x):
        return stats.lognorm.pdf(x, s=0.5, scale=np.exp(np.log(5)))
    def prior_delta(x):
        return stats.lognorm.pdf(x, s=0.8, scale=np.exp(np.log(0.05)))
    def prior_I0(x):
        return stats.lognorm.pdf(x, s=1.0, scale=np.exp(np.log(2000)))
    def prior_alpha(x):
        mu, sigma = 0.4, 0.9
        eps = 1e-15
        x_safe = np.clip(x, eps, 1-eps)
        logit_x = np.log(x_safe / (1.0 - x_safe))
        return stats.norm.pdf(logit_x, loc=mu, scale=sigma) / (x_safe * (1.0 - x_safe))
    def prior_p(x):
        return stats.lognorm.pdf(x, s=0.5, scale=np.exp(np.log(0.01)))
    def prior_theta(x):
        return stats.lognorm.pdf(x, s=0.5, scale=np.exp(np.log(20)))
    def prior_gamma(x):
        return stats.lognorm.pdf(x, s=0.1, scale=np.exp(np.log(0.8)))
    def prior_kappa(x):
        return stats.lognorm.pdf(x, s=0.2, scale=np.exp(np.log(0.4)))
    
    priors = [prior_beta0, prior_delta, prior_I0, prior_alpha, 
              prior_p, prior_theta, prior_gamma, prior_kappa]
    
    rangos_manuales = {
        'beta0': (0.1, 15.0),
        'delta': (0.0, 0.3),
        'I0':    (0.0, 20000.0),
        'alpha': (0.0, 1.0),
        'p':     (0.0, 0.03),
        'theta': (0.0, 80.0),
        'gamma': (0.0, 1.5),
        'kappa': (0.0, 1.0)
    }
    
    labels = [r'$\beta_0$', r'$\delta$', r'$I_0$', r'$\alpha$', 
              r'$p$', r'$\theta$', r'$\gamma$', r'$\kappa$']
    descs = ['Tasa de transmisión inicial', 'Tasa de decaimiento',
             'Infectados iniciales', 'Orden fraccionario',
             'Tasa de subreporte', 'Sobredispersión',
             'Tasa de recuperación', 'Tasa de latencia']
    
    n_params = 8
    cols = 2
    rows = 4
    
    fig, axes = plt.subplots(rows, cols, figsize=(14, 20))
    axes = axes.flatten()
    
    for i in range(n_params):
        ax = axes[i]
        name = nombres_params[i]
        x_min, x_max = rangos_manuales[name]
        
        post = posterior[:, i]
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
        ax.axvline(map_vals[i], color='black', linestyle='-', 
                  linewidth=2.0, alpha=0.9, label='MAP conjunto')
        ax.axvline(map_marg[i], color='black', linestyle=':', 
                  linewidth=1.2, alpha=0.6, label='MAP marginal')
        ax.axvline(media_val, color='gray', linestyle='-.', 
                  linewidth=0.8, alpha=0.5, label='Media')
        
        info_text = (
            f'MAP conj:  {map_vals[i]:.3g}\n'
            f'MAP marg:  {map_marg[i]:.3g}\n'
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
                       handlelength=1.5, ncol=2)
        if leg:
            for text in leg.get_texts():
                text.set_style('italic')
    
    plt.suptitle('Actualización Bayesiana: Modelo con $\\beta(t)=\\beta_0 e^{-\\delta t}$\n'
                 'MAP conjunto (—), MAP marginal (···), Media (-·-)',
                fontsize=13, style='italic', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'prior_vs_posterior_beta_var.png'), 
                dpi=300, facecolor='white', bbox_inches='tight')
    plt.close()
    
    return map_vals, map_marg

# ============================================================
# PERFIL TEMPORAL
# ============================================================
def graficar_perfil_epidemia(fit, y_obs, outdir):
    """Perfil temporal con residuos."""
    mu_samples = fit.stan_variable('mu')
    semanas = np.arange(1, len(y_obs) + 1)
    
    mu_map = np.mean(mu_samples, axis=0)
    lower_95 = np.percentile(mu_samples, 2.5, axis=0)
    upper_95 = np.percentile(mu_samples, 97.5, axis=0)
    lower_50 = np.percentile(mu_samples, 25, axis=0)
    upper_50 = np.percentile(mu_samples, 75, axis=0)
    
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

# ============================================================
# RESUMEN NUMÉRICO
# ============================================================
def resumen_numerico(posterior, nombres=None):
    if nombres is None:
        nombres = [f'Param_{i}' for i in range(posterior.shape[1])]
    res = {}
    for i, nom in enumerate(nombres):
        vals = posterior[:, i]
        res[nom] = [np.mean(vals), np.percentile(vals, 2.5), np.percentile(vals, 97.5)]
    df = pd.DataFrame(res, index=['mean', '2.5%', '97.5%']).T
    return df