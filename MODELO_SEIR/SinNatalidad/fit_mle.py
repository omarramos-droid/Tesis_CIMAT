# -*- coding: utf-8 -*-
"""Ajuste MLE para SIR fraccionario"""

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

# Argumentos

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--h", type=float, default=0.1)
args = parser.parse_args()

seed = args.seed
h = args.h

# Paths


data_path = f"data/y_obs_seed_{seed}.npy"
t_path    = "data/t_data.npy"
params_path = f"data/params_true_seed_{seed}.json"

# Cargar datos

y_obs = np.load(data_path)
t_data = np.load(t_path)

with open(params_path) as f:
    params_true = json.load(f)

N_true = params_true["N"]
I0_true = params_true["I0"]
alpha_true = params_true["alpha"]

# Solver 

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

        b = np.array([(n - j) ** alpha - (n - j - 1) ** alpha for j in range(n)])
        y_pred = y[0] + (h_alpha / gamma_alpha1) * np.dot(b, fvals[:n])
        y_pred = np.maximum(y_pred, 0)
        f_pred = f(t[n], y_pred)

        a = np.zeros(n + 1)
        a[0] = (n - 1) ** (alpha + 1) - (n - 1 - alpha) * n ** alpha

        for j in range(1, n):
            a[j] = ((n - j + 1) ** (alpha + 1) +
                    (n - j - 1) ** (alpha + 1) -
                    2 * (n - j) ** (alpha + 1))

        a[n] = 1.0

        y[n] = y[0] + (h_alpha / gamma_alpha2) * (
            np.dot(a[:n], fvals[:n]) + a[n] * f_pred
        )
        y[n] = np.maximum(y[n], 0)

        fvals[n] = f(t[n], y[n])

    return t, y

def sir_rhs(t, Y, beta, gamma, N):
    S, I, R = Y
    return np.array([
        -beta * S * I / N,
         beta * S * I / N - gamma * I,
         gamma * I
    ])

def solve_sir_frac(params, alpha, t_data, h):
    beta, gamma, N, I0 = params["beta"], params["gamma"], params["N"], params["I0"]
    y0 = np.array([N - I0, I0, 0.0])
    T = t_data[-1]

    t_fine, sol = solver_fraccionario(
        lambda t, Y: sir_rhs(t, Y, beta, gamma, N),
        alpha, y0, T, h
    )

    idx = np.clip(np.round(t_data / h).astype(int), 0, len(sol)-1)
    return sol[idx]

# Likelihood NB


def neg_log_likelihood_nb(params, t_data, y_obs, h):
    alpha = params['alpha'].value
    theta = params['theta'].value
    p = {k: params[k].value for k in params if k not in ['alpha', 'theta']}

    try:
        sol = solve_sir_frac(p, alpha, t_data, h)
        S = sol[:, 0]
        inc = np.clip(S[:-1] - S[1:], 1e-8, None)

        ll = (gammaln(y_obs + theta) - gammaln(theta) - gammaln(y_obs + 1) +
              theta * np.log(theta / (theta + inc)) +
              y_obs * np.log(inc / (theta + inc)))

        return -np.sum(ll)
    except:
        return 1e10

# Ajustes
def ajustar_modelo_classic():
    params = Parameters()
    params.add('beta',  value=1.0, min=0.1, max=3.0)
    params.add('gamma', value=0.5, min=0.05, max=1.1)
    params.add('theta', value=25.0, min=0.5, max=50.0)
    params.add('I0',    value=2, min=1, max=10)
    params.add('N',     value=N_true, vary=False)
    params.add('alpha', value=1.0, vary=False)

    return minimize(neg_log_likelihood_nb, params,
                    args=(t_data, y_obs, h), method='powell')

def ajustar_modelo_frac():
    params = Parameters()
    params.add('beta',  value=1.2, min=0.2, max=4.0)
    params.add('gamma', value=0.5, min=0.05, max=0.8)
    params.add('theta', value=10.0, min=0.5, max=50.0)
    params.add('alpha', value=0.7, min=0.2, max=0.99)
    params.add('I0',    value=I0_true, min=1, max=10)
    params.add('N',     value=N_true, vary=False)

    return minimize(neg_log_likelihood_nb, params,
                    args=(t_data, y_obs, h), method='powell')

# Ejecutar


print(f"Ajuste clasico ")

res_c = ajustar_modelo_classic()
print(f"Ajuste frac ")

res_f = ajustar_modelo_frac()


logL_c = -res_c.fun
logL_f = -res_f.fun

D = -2 * (logL_c - logL_f)
p_value = 1 - chi2.cdf(D, df=1)
print(f"Estadistico D={D}, con p_valor={p_value:.4e}")
# Guardar resultados

os.makedirs("outputs", exist_ok=True)

results = {
    "seed": seed,
    "logL_classic": logL_c,
    "logL_frac": logL_f,
    "D": D,
    "p_value": p_value,
    "alpha_est": res_f.params["alpha"].value,
    "beta_est": res_f.params["beta"].value,
    "gamma_est": res_f.params["gamma"].value,
    "theta_est": res_f.params["theta"].value,
    "I0_est": res_f.params["I0"].value,
    "alpha_true": alpha_true
}
print(results)
print(params_true)
with open(f"outputs/mle_results_seed_{seed}.json", "w") as f:
    json.dump(results, f, indent=2)

# Gráfico 
t_inc = t_data[:-1]

sol_c = solve_sir_frac(
    {k: res_c.params[k].value for k in res_c.params if k not in ["alpha","theta"]},
    1.0, t_data, h
)

sol_f = solve_sir_frac(
    {k: res_f.params[k].value for k in res_f.params if k not in ["alpha","theta"]},
    res_f.params["alpha"].value, t_data, h
)

inc_c = np.clip(sol_c[:-1,0] - sol_c[1:,0], 1e-8, None)
inc_f = np.clip(sol_f[:-1,0] - sol_f[1:,0], 1e-8, None)

plt.figure()
plt.plot(t_inc, y_obs, "ko", label="data")
plt.plot(t_inc, inc_c, label="classic")
plt.plot(t_inc, inc_f, label="frac")
plt.legend()
plt.savefig(f"outputs/fit_seed_{seed}.png", dpi=150)
plt.close()

