# -*- coding: utf-8 -*-
"""Simulación_Fraccionaria_SEIR - """

import numpy as np
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.integrate as spi
from lmfit import minimize, Parameters
from scipy.special import gammaln
from scipy.stats import chi2
import os
os.makedirs("outputs", exist_ok=True)

# Solver fraccionario  (Adams–Bashforth–Moulton)
def solver_fraccionario(f, alpha, y0, T, h):
    N = int(round(T / h))
    t = np.linspace(0, T, N + 1)
    dim = len(y0)
    y = np.zeros((N + 1, dim))
    fvals = np.zeros((N + 1, dim))

    y[0] = y0
    fvals[0] = f(t[0], y[0])

    gamma_alpha1 = math.gamma(alpha + 1)
    gamma_alpha2 = math.gamma(alpha + 2)
    h_alpha = h ** alpha

    for n in range(1, N + 1):
        # --- Predictor ---
        b = np.array([(n - j) ** alpha - (n - j - 1) ** alpha for j in range(n)])
        pred_sum = np.dot(b, fvals[:n])          # fvals[:n] tiene forma (n, dim)
        y_pred = y[0] + (h_alpha / gamma_alpha1) * pred_sum
        y_pred = np.maximum(y_pred, 0)
        f_pred = f(t[n], y_pred)

        # --- Corrector ---
        a = np.zeros(n + 1)
        a[0] = (n - 1) ** (alpha + 1) - (n - 1 - alpha) * n ** alpha
        for j in range(1, n):
            a[j] = ((n - j + 1) ** (alpha + 1) +
                    (n - j - 1) ** (alpha + 1) -
                    2 * (n - j) ** (alpha + 1))
        a[n] = 1.0

        corr_sum = np.dot(a[:n], fvals[:n]) + a[n] * f_pred
        y[n] = y[0] + (h_alpha / gamma_alpha2) * corr_sum
        y[n] = np.maximum(y[n], 0)
        fvals[n] = f(t[n], y[n])

    return t, y


# Modelo SEIR 

def seir_rhs(t, Y, beta, gamma, kappa, N):
    S, E, I, R = Y
    dS = -beta * S * I / N
    dE =  beta * S * I / N - kappa * E
    dI =  kappa * E - gamma * I
    dR =  gamma * I
    return np.array([dS, dE, dI, dR])


# Solver fraccionario para el SEIR 
def solve_seir_frac(params, alpha, t_data, h):
    """
    params: diccionario con 'beta','gamma','kappa','N','E0','I0'
    t_data: tiempos donde se pide la solución (p.ej. enteros)
    h: paso de integración (debe ser <= 1)
    """
    beta  = params['beta']
    gamma = params['gamma']
    kappa = params['kappa']
    N_pop = params['N']
    I0    = params['I0']
    E0    = params['E0']
    S0    = N_pop - I0 - E0
    y0    = [S0, E0, I0, 0.0]
    T     = t_data[-1]

    t_fine, sol_fine = solver_fraccionario(
        lambda t, Y: seir_rhs(t, Y, beta, gamma, kappa, N_pop),
        alpha, y0, T, h
    )
    sol_fine = np.maximum(sol_fine, 0)      # seguridad extra

    # Interpolar a los tiempos pedidos
    idx = np.clip(np.round(t_data / h).astype(int), 0, len(sol_fine)-1)
    return sol_fine[idx]


# Generación de datos sintéticos 

## Estimaci�n de I_0 en la geografia  y agregar la demografia.
## 
beta_true   = 1.7
gamma_true  = 0.8
kappa_true  = 2
N_true      = 4500
I0_true     = 5
E0_true     = 0
alpha_true  = 0.8
theta_true  = 30   # sobredispersión
t_max       = 20
h_gen       = 0.05    # paso fino para generar datos (no confundir con h de ajuste)
np.random.seed(12+1)

S0_true = N_true - I0_true - E0_true
y0_true = [S0_true, E0_true, I0_true, 0.0]
t_data  = np.arange(0, t_max + 1, 1)          # tiempos de observación (enteros)

# Resolver con el solver 
t_fine, sol_fine = solver_fraccionario(
    lambda t, Y: seir_rhs(t, Y, beta_true, gamma_true, kappa_true, N_true),
    alpha_true, y0_true, t_data[-1], h_gen
)

idx = np.round(t_data / h_gen).astype(int)
sol_true = sol_fine[np.round(t_data / h_gen).astype(int)]
E_true = sol_true[:, 1]

incidencia_true = kappa_true * (E_true[:-1] + E_true[1:]) / 2.0
incidencia_true = np.clip(incidencia_true, 1e-8, None)


# Ruido Binomial Negativo
mu = incidencia_true
p_nb = theta_true / (theta_true + mu)
y_obs = np.random.negative_binomial(theta_true, p_nb).astype(float)

t_inc = t_data[:-1]

plt.figure()
plt.plot(t_inc, incidencia_true, 'r-o')
plt.title('Incidencia verdaderas (sin ruido)')
plt.savefig("outputs/Simulascion_Indicencia.png", dpi=300)
plt.show()


# Gráficos 
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
axs[0].plot(t_fine, sol_fine[:,0], 'b-', label='Susceptibles')
axs[0].plot(t_fine, sol_fine[:,1], 'r-', label='Expuestos')
axs[0].plot(t_fine, sol_fine[:,2], 'y-', label='Infectados')
axs[0].plot(t_fine, sol_fine[:,3], 'g-', label='Recuperados')
axs[0].set_xlabel('Tiempo')
axs[0].set_ylabel('Individuos')
axs[0].set_title('SEIR fraccionario (sintético)')
axs[0].legend()
axs[0].grid(alpha=0.3)

axs[1].plot(t_inc, incidencia_true, 'b--', lw=2, label='Incidencia verdadera')
axs[1].plot(t_inc, y_obs, 'ko', ms=5, alpha=0.7, label='Observaciones BN')
axs[1].set_xlabel('Semanas')
axs[1].set_ylabel('Nuevos casos')
axs[1].set_title('Incidencia semanal')
axs[1].legend()
axs[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("outputs/Simulacion_SEIR.png", dpi=300)
plt.close()


# Función de verosimilitud (Binomial Negativa)


def neg_log_likelihood_nb(params, t_data, y_obs, h):
    alpha = params['alpha'].value
    theta = params['theta'].value
    p = {k: params[k].value for k in params if k not in ['alpha','theta']}
## if(alpha>0.98)
    try:
        sol = solve_seir_frac(p, alpha, t_data, h)
        E = sol[:, 1]
        kappa = p['kappa']
        inc = kappa * (E[:-1] + E[1:]) / 2.0
        inc = np.clip(inc, 1e-8, None)
        mu = inc
        y = y_obs

        # Log‑verosimilitud binomial negativa
        ll = (gammaln(y + theta) - gammaln(theta) - gammaln(y + 1) +
              theta * np.log(theta / (theta + mu)) +
              y * np.log(mu / (theta + mu)))
        return -np.sum(ll)
    except Exception:
        return 1e10   # valor grande pero finito, evita NaN


# Ajuste del modelo clásico (α=1 fijo)
def ajustar_modelo_classic(t_data, y_obs, h):
    params = Parameters()
    params.add('beta',  value=1.0, min=0.1, max=3.0)
    params.add('gamma', value=0.5, min=0.05, max=1.0)
    params.add('kappa', value=1.0, min=1.0, max=10.0)
    params.add('theta', value=25.0, min=0.5, max=50.0)
    params.add('E0',    value=E0_true, vary=False)
    params.add('I0',    value=I0_true, vary=False)
    params.add('N',     value=N_true,  vary=False)
    params.add('alpha', value=1.0, vary=False)
    result = minimize(neg_log_likelihood_nb, params,
                      args=(t_data, y_obs, h),
                      method='powell')   # más robusto que nelder
    return result


# Ajuste del modelo fraccionario (α libre)

def ajustar_modelo_frac(t_data, y_obs, h):
    params = Parameters()
    params.add('beta',  value=1.2, min=0.2, max=4.0)
    params.add('gamma', value=0.5, min=0.05, max=0.8)
    params.add('kappa', value=1, min=0.1, max=8.0)
    params.add('theta', value=10.0, min=0.5, max=50.0)
    params.add('alpha', value=0.7, min=0.2, max=0.99)
    params.add('E0',    value=E0_true, vary=False)
    params.add('I0',    value=I0_true, vary=False)
    params.add('N',     value=N_true,  vary=False)
    result = minimize(neg_log_likelihood_nb, params,
                      args=(t_data, y_obs, h),
                      method='powell')
    return result


# Prueba de hipótesis y gráficos finales


def likelihood_ratio_test(t_data, y_obs, h):
    print("Ajustando modelo clásico (α=1)...")
    res_classic = ajustar_modelo_classic(t_data, y_obs, h)
    print("Ajustando modelo fraccionario (α libre)...")
    res_frac = ajustar_modelo_frac(t_data, y_obs, h)

    logL_classic = -res_classic.fun
    logL_frac    = -res_frac.fun

    D = -2 * (logL_classic - logL_frac)
    p_value = 1 - chi2.cdf(D, df=1)

    k_classic = len([p for p in res_classic.params if res_classic.params[p].vary])
    k_frac    = len([p for p in res_frac.params if res_frac.params[p].vary])
    n = len(y_obs)

    aic_classic = -2 * logL_classic + 2 * k_classic
    aic_frac    = -2 * logL_frac    + 2 * k_frac
    bic_classic = -2 * logL_classic + k_classic * np.log(n)
    bic_frac    = -2 * logL_frac    + k_frac   * np.log(n)

    print("\n =Para el Ruido Binomial Negativo se presenta la log vero ")
    print(f"logL (clásico): {logL_classic:.2f}")
    print(f"logL (fracc):   {logL_frac:.2f}")
    print(f"D = {D:.4f}, p-value = {p_value:.4f}")
    print(f"AIC: clásico = {aic_classic:.2f}, fracc = {aic_frac:.2f}")
    print(f"BIC: clásico = {bic_classic:.2f}, fracc = {bic_frac:.2f}")

    # --- Gráfico de incidencia ajustada ---
    t_inc = t_data[:-1]

    # Clásico
    alpha_c = 1.0
    theta_c = res_classic.params['theta'].value
    p_c = {k: res_classic.params[k].value for k in res_classic.params if k not in ['alpha','theta']}
    sol_c = solve_seir_frac(p_c, alpha_c, t_data, h)
    inc_c = p_c['kappa'] * (sol_c[:,1][:-1] + sol_c[:,1][1:]) / 2.0
    inc_c = np.clip(inc_c, 1e-8, None)

    # Fraccionario
    alpha_f = res_frac.params['alpha'].value
    theta_f = res_frac.params['theta'].value
    p_f = {k: res_frac.params[k].value for k in res_frac.params if k not in ['alpha','theta']}
    sol_f = solve_seir_frac(p_f, alpha_f, t_data, h)
    inc_f = p_f['kappa'] * (sol_f[:,1][:-1] + sol_f[:,1][1:]) / 2.0
    inc_f = np.clip(inc_f, 1e-8, None)

    plt.figure(figsize=(10,5))
    plt.plot(t_inc, y_obs, 'ko', ms=4, alpha=0.7, label='Datos reales (BN)')
    plt.plot(t_inc, inc_c, 'b-', lw=2, label=f'Clásico (α=1, θ={theta_c:.2f})')
    plt.plot(t_inc, inc_f, 'r--', lw=2, label=f'Fracc (α={alpha_f:.3f}, θ={theta_f:.2f})')
    plt.plot(t_inc, incidencia_true, 'g--', lw=2, label='Verdadero (α=0.8)')
    plt.xlabel('Tiempo (semanas)')
    plt.ylabel('Nuevos casos')
    plt.title('Comparación de modelos')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(f"outputs/Ajuste_modelo_NB_h_{h}.png", dpi=300, bbox_inches='tight')
    plt.close()
    from lmfit.printfuncs import report_fit
    report_fit(res_frac)
    return res_classic, res_frac


# Ejecución principal
if __name__ == "__main__":
    import sys
    # El paso h se  pasa como argumento; si no, usamos 0.1
    h = float(sys.argv[1]) if len(sys.argv) > 1 else 0.1
    print(f"Ejecutando con paso de integración h = {h}")

    res_classic, res_frac = likelihood_ratio_test(t_data, y_obs, h)

    print("\n Parámetros estimados (modelo fraccionario) ---")
    for name, par in res_frac.params.items():
        print(f"{name}: {par.value:.5f}  (vary={par.vary})")