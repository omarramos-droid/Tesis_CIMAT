# -*- coding: utf-8 -*-
"""
Ajuste MLE para SEIR fraccionario con demografía.
Estima Lambda, beta, sigma, gamma, mu, alpha, I0 y phi (dispersión).
Compara modelo clásico (alpha=1) vs fraccionario (alpha libre) mediante LRT.
"""

import numpy as np
import math
import os
import json
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lmfit import minimize, Parameters
from scipy.special import gammaln
from scipy.stats import chi2

# Importar el solver unificado
from solver_fraccionario import *

# Argumentos

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=12+1)
parser.add_argument("--h", type=float, default=0.1)
args = parser.parse_args()

seed = args.seed
h = args.h

# Cargar datos y parámetros verdaderos

data_path = f"data/y_obs_seed_{seed}.npy"
t_path    = "data/t_data.npy"
params_path = f"data/params_true_seed_{seed}.json"

y_obs = np.load(data_path)
t_data = np.load(t_path)

with open(params_path) as f:
    params_true = json.load(f)

# Valores verdaderos 
N0_true = params_true.get("N0", 4500)          # población total inicial
I0_true = params_true.get("I0", 5)
E0_true = params_true.get("E0", 0)
alpha_true = params_true.get("alpha", 0.8)
Lambda_true = params_true.get("Lambda", 20.0)
mu_true = params_true.get("mu", 0.01)
sigma_true = params_true.get("sigma", 2.0)
beta_true = params_true.get("beta", 1.7)
gamma_true = params_true.get("gamma", 0.8)
phi_true = params_true.get("phi", 30)     # dispersión de la BN

# Función de log-verosimilitud (Binomial Negativa)


def neg_log_likelihood_nb(params, t_data, y_obs, h):
    # Extraer parámetros
    alpha = params['alpha'].value
    phi   = params['phi'].value
    beta  = params['beta'].value
    sigma = params['sigma'].value
    gamma = params['gamma'].value
    mu    = params['mu'].value
    Lambda = params['Lambda'].value
    I0    = int(round(params['I0'].value))
    E0    = 0                              
    N0    = params['N0'].value

    # Validaciones 
    if beta <= 0 or sigma <= 0 or gamma <= 0 or mu < 0 or Lambda < 0 or phi <= 0:
        return 1e10
    if I0 < 1 or I0 > N0:
        return 1e10
    if not (0 < alpha <= 1):
        return 1e10

    try:
        # Resolver el SEIR con demografía
        res = resolver_seirv(
            beta=beta, sigma=sigma, gamma=gamma, mu=mu, Lambda=Lambda,
            alpha=alpha, N0=N0, I0=I0, E0=E0, T=t_data[-1], h=h,
            t_observacion=t_data
        )
    except Exception:
        return 1e10

        # Incidencia como diferencia de susceptibles (consistente con Stan)
    inc = calcular_incidencia_diffS(res)
    inc = np.clip(inc, 1e-8, None)

    if len(inc) != len(y_obs):
        return 1e10

    # Log-verosimilitud binomial negativa
    ll = (gammaln(y_obs + phi) - gammaln(phi) - gammaln(y_obs + 1) +
          phi * np.log(phi / (phi + inc)) +
          y_obs * np.log(inc / (phi + inc)))
    return -np.sum(ll)


# Ajuste del modelo clásico (alpha = 1 fijo)
def ajustar_modelo_classic():
    params = Parameters()
    # Parámetros libres
    params.add('beta',   value=1.5, min=0.1, max=2.0)
    params.add('sigma',  value=1.5, min=0.1, max=2.0)
    params.add('gamma',  value=1, min=0.05, max=2.0)
    params.add('mu',     value=0.01, min=0.0001, max=0.3)
    params.add('Lambda', value=20.0, min=20.0, max=50.0)
    params.add('phi',    value=10.0, min=10, max=50.0)
    params.add('I0',     value=5, min=1, max=50)
    params.add('N0',     value=N0_true, vary=False)   # población inicial conocida
    params.add('E0',     value=E0_true, vary=False)   # Suceptibles
    params.add('alpha',  value=1.0, vary=False)       # fijo

    return minimize(neg_log_likelihood_nb, params,
                    args=(t_data, y_obs, h), method='powell')

# Ajuste del modelo fraccionario (alpha libre)
def ajustar_modelo_frac():
    params = Parameters()
    params.add('beta',   value=1.5, min=0.1, max=2.0)
    params.add('sigma',  value=2.0, min=0.1, max=2.0)
    params.add('gamma',  value=0.5, min=0.05, max=2.0)
    params.add('mu',     value=0.001, min=0.00001, max=0.2)
    params.add('Lambda', value=20.0, min=20.0, max=50.0)
    params.add('phi',    value=20.0, min=10, max=50.0)
    params.add('I0',     value=2, min=1, max=50)
    params.add('N0',     value=N0_true, vary=False)
    params.add('E0',     value=E0_true, vary=False)   # Suceptibles

    params.add('alpha',  value=0.5, min=0.2, max=0.99)   # libre

    return minimize(neg_log_likelihood_nb, params,
                    args=(t_data, y_obs, h), method='powell')

# Ejecutar y comparar

print("Ajustando modelo clásico (alpha=1)...")
res_c = ajustar_modelo_classic()
print("Ajustando modelo fraccionario (alpha libre)...")
res_f = ajustar_modelo_frac()

logL_c = -res_c.fun
logL_f = -res_f.fun

D = -2 * (logL_c - logL_f)
p_value = 1 - chi2.cdf(D, df=1)

print(f"logL clásico: {logL_c:.2f}")
print(f"logL fracc:   {logL_f:.2f}")
print(f"Estadístico D = {D:.4f}, p-value = {p_value:.4e}")

# Guardar resultados
os.makedirs("outputs", exist_ok=True)

results = {
    "seed": seed,
    "logL_classic": logL_c,
    "logL_frac": logL_f,
    "D": D,
    "p_value": p_value,
    # Estimaciones del modelo fraccionario
    "beta_est":  res_f.params["beta"].value,
    "sigma_est": res_f.params["sigma"].value,
    "gamma_est": res_f.params["gamma"].value,
    "mu_est":    res_f.params["mu"].value,
    "Lambda_est":res_f.params["Lambda"].value,
    "phi_est":   res_f.params["phi"].value,
    "I0_est":    res_f.params["I0"].value,
    "alpha_est": res_f.params["alpha"].value,
    "alpha_true": alpha_true
}
print("\nResultados fraccionario:", results)
print("Parámetros verdaderos:", params_true)

with open(f"outputs/mle_results_seed_{seed}.json", "w") as f:
    json.dump(results, f, indent=2)

# Gráfico de ajuste (SEIR con demografía)
t_inc = t_data[1:]

# Solución con modelo clásico
p_c = {k: res_c.params[k].value for k in ["beta","sigma","gamma","mu","Lambda","N0","I0"]}
res_c_sol = resolver_seirv(
    beta=p_c["beta"], sigma=p_c["sigma"], gamma=p_c["gamma"],
    mu=p_c["mu"], Lambda=p_c["Lambda"], alpha=1.0,
    N0=N0_true, I0=int(round(p_c["I0"])), E0=0, T=t_data[-1], h=h,
    t_observacion=t_data
)
inc_c = calcular_incidencia_diffS(res_c_sol)

# Solución con modelo fraccionario
p_f = {k: res_f.params[k].value for k in ["beta","sigma","gamma","mu","Lambda","N0","I0"]}
res_f_sol = resolver_seirv(
    beta=p_f["beta"], sigma=p_f["sigma"], gamma=p_f["gamma"],
    mu=p_f["mu"], Lambda=p_f["Lambda"], alpha=res_f.params["alpha"].value,
    N0=N0_true, I0=int(round(p_f["I0"])), E0=0, T=t_data[-1], h=h,
    t_observacion=t_data
)
inc_f = calcular_incidencia_diffS(res_f_sol)

plt.figure(figsize=(10,5))
plt.plot(t_inc, y_obs, "ko", label="Datos observados")
plt.plot(t_inc, inc_c, "b-", label=f"Clásico (α=1, φ={res_c.params['phi'].value:.1f})")
plt.plot(t_inc, inc_f, "r--", label=f"Fracc (α={res_f.params['alpha'].value:.3f}, φ={res_f.params['phi'].value:.1f})")
plt.xlabel("Tiempo")
plt.ylabel("Nuevos casos")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig(f"outputs/fit_seed_{seed}.png", dpi=150)
plt.close()
