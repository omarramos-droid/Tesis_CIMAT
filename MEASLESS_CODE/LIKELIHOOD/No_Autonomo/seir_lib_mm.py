# seir_lib_mm.py
# Funciones para SEIR fraccionario y clásico con β(t) exponencial

import os
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.integrate import odeint
from scipy.special import gammaln
from scipy.interpolate import interp1d
from lmfit import minimize, Parameters
import multiprocessing as mp
from functools import partial

# CONSTANTES EPIDEMIOLÓGICAS (semanas⁻¹)
GAMMA = 0.875      # 1/periodo infeccioso (8 días → 0.875 semanas⁻¹)
KAPPA = 0.5        # 1/periodo incubación (14 días → 0.5 semanas⁻¹)
E0_FIJO = 0.0      # Exposición inicial fija en 0

# ---------- SOLVERS ----------
def seir_rhs_tvar(t, Y, beta0, delta, gamma, kappa, N):
    """RHS del SEIR con beta(t) = beta0 * exp(-delta * t)"""
    S, E, I, R = Y
    beta = beta0 * np.exp(-delta * t)
    dS = -beta * S * I / N
    dE =  beta * S * I / N - kappa * E
    dI =  kappa * E - gamma * I
    dR =  gamma * I
    return np.array([dS, dE, dI, dR])



def solver_fraccionario(f, alpha, y0, T, h=0.1):
    """Solver fraccionario genérico"""
    n_steps = int(round(T / h))
    t = np.linspace(0, T, n_steps + 1)
    dim = len(y0)
    y = np.zeros((n_steps + 1, dim))
    fvals = np.zeros((n_steps + 1, dim))
    y[0] = y0
    fvals[0] = f(t[0], y[0])
    gamma_alpha1 = math.gamma(alpha + 1)
    gamma_alpha2 = math.gamma(alpha + 2)
    h_alpha = h ** alpha

    for n in range(1, n_steps + 1):
        b = np.array([(n - j) ** alpha - (n - j - 1) ** alpha for j in range(n)])
        y_pred = y[0] + (h_alpha / gamma_alpha1) * np.dot(b, fvals[:n])
        y_pred = np.maximum(y_pred, 0)
        f_pred = f(t[n], y_pred)

        a = np.zeros(n + 1)
        a[0] = (n - 1) ** (alpha + 1) - (n - 1 - alpha) * n ** alpha
        for j in range(1, n):
            a[j] = ((n - j + 1) ** (alpha + 1) + (n - j - 1) ** (alpha + 1)
                    - 2 * (n - j) ** (alpha + 1))
        a[n] = 1.0
        y[n] = y[0] + (h_alpha / gamma_alpha2) * (np.dot(a[:n], fvals[:n]) + a[n] * f_pred)
        y[n] = np.maximum(y[n], 0)
        fvals[n] = f(t[n], y[n])

    return t, y

def solve_seir_frac_tvar(params, alpha, t_eval, h=0.1):
    """Resuelve SEIR fraccionario con beta variable"""
    beta0 = params["beta0"]
    delta = params["delta"]
    gamma = params["gamma"]
    kappa = params["kappa"]
    N = params["N"]
    I0 = params["I0"]
    E0 = E0_FIJO
    S0 = N - E0 - I0
    
    if S0 <= 0:
        raise ValueError(f"S0 debe ser positivo: S0={S0}")
    
    y0 = np.array([S0, E0, I0, 0.0])
    T = t_eval[-1]
    
    t_fine, sol = solver_fraccionario(
        lambda t, Y: seir_rhs_tvar(t, Y, beta0, delta, gamma, kappa, N),
        alpha, y0, T, h
    )
    interp = interp1d(t_fine, sol, axis=0, kind='linear',
                      bounds_error=False, fill_value=(sol[0], sol[-1]))
    return interp(t_eval)

def solve_seir_odeint_tvar(params, t_eval):
    """Resuelve SEIR clásico (ODEINT) con beta variable"""
    beta0 = params["beta0"]
    delta = params["delta"]
    gamma = params["gamma"]
    kappa = params["kappa"]
    N = params["N"]
    I0 = params["I0"]
    E0 = E0_FIJO
    S0 = N - E0 - I0
    
    if S0 <= 0:
        raise ValueError(f"S0 debe ser positivo: S0={S0}")
    
    y0 = [S0, E0, I0, 0.0]
    
    def deriv(y, t):
        S, E, I, R = y
        beta = beta0 * np.exp(-delta * t)
        dS = -beta * S * I / N
        dE =  beta * S * I / N - kappa * E
        dI =  kappa * E - gamma * I
        dR =  gamma * I
        return [dS, dE, dI, dR]
    
    return odeint(deriv, y0, t_eval)

# ---------- VEROSIMILITUD (Binomial Negativa) ----------
def negative_log_likelihood_frac_tvar(params, t_eval, y_obs, h=0.1):
    """Verosimilitud negativa para modelo fraccionario con beta variable"""
    log_beta0 = params["log_beta0"].value
    beta0 = np.exp(log_beta0)
    log_delta = params["log_delta"].value
    delta = np.exp(log_delta)
    log_theta = params["log_theta"].value
    theta = np.exp(log_theta)
    theta = max(theta, 1e-4)
    log_p = params["log_p"].value
    p_report = np.exp(log_p)
    log_I0 = params["log_I0"].value
    I0 = np.exp(log_I0)
    alpha = params["alpha"].value
    gamma = params["gamma"].value
    kappa = params["kappa"].value
    N = params["N"].value

    E0 = E0_FIJO
    S0 = N - E0 - I0
    if S0 <= 0:
        return 1e10

    p_seir = {
        "beta0": beta0, "delta": delta, "gamma": gamma, "kappa": kappa,
        "N": N, "I0": I0
    }
    try:
        sol = solve_seir_frac_tvar(p_seir, alpha, t_eval, h)
        S = sol[:, 0]
        incidence = np.maximum(S[:-1] - S[1:], 1e-8)
        mu = p_report * incidence
        y = y_obs[:len(mu)]
        mu = mu[:len(y)]
        mu = np.maximum(mu, 1e-8)

        ll = (gammaln(y + theta) - gammaln(theta) - gammaln(y + 1) +
              theta * np.log(theta / (theta + mu)) +
              y * np.log(mu / (theta + mu)))
        ll = np.where(np.isnan(ll), -1e10, ll)
        return -np.sum(ll)
    except Exception:
        return 1e10

def negative_log_likelihood_ode_tvar(params, t_eval, y_obs):
    """Verosimilitud negativa para modelo ODE con beta variable"""
    log_beta0 = params["log_beta0"].value
    beta0 = np.exp(log_beta0)
    log_delta = params["log_delta"].value
    delta = np.exp(log_delta)
    log_theta = params["log_theta"].value
    theta = np.exp(log_theta)
    theta = max(theta, 1e-4)
    log_p = params["log_p"].value
    p_report = np.exp(log_p)
    log_I0 = params["log_I0"].value
    I0 = np.exp(log_I0)
    gamma = params["gamma"].value
    kappa = params["kappa"].value
    N = params["N"].value

    E0 = E0_FIJO
    S0 = N - E0 - I0
    if S0 <= 0:
        return 1e10

    p_seir = {
        "beta0": beta0, "delta": delta, "gamma": gamma, "kappa": kappa,
        "N": N, "I0": I0
    }
    try:
        sol = solve_seir_odeint_tvar(p_seir, t_eval)
        S = sol[:, 0]
        incidence = np.maximum(S[:-1] - S[1:], 1e-8)
        mu = p_report * incidence
        y = y_obs[:len(mu)]
        mu = mu[:len(y)]
        mu = np.maximum(mu, 1e-8)
        ll = (gammaln(y + theta) - gammaln(theta) - gammaln(y + 1) +
              theta * np.log(theta / (theta + mu)) +
              y * np.log(mu / (theta + mu)))
        ll = np.where(np.isnan(ll), -1e10, ll)
        return -np.sum(ll)
    except Exception:
        return 1e10

# ---------- PROGRESO ----------
def monitor_progress(params, iteration, resid, *args, **kwargs):
    if iteration % 50 == 0:
        msg = [f"{name}={par.value:.5f}" for name, par in params.items() if par.vary]
        print(f"Iter {iteration:04d} | Loss={resid:.4f} | " + " | ".join(msg))

# ---------- AJUSTES ----------
def fit_fractional_model(N_TRUE, t_data, y_obs, H):
    """Ajusta modelo fraccionario con beta variable """
    params = Parameters()
    # Usar los valores óptimos de ODEINT como iniciales
    params.add("log_beta0", value=0.91671, min=np.log(0.5), max=np.log(10.0))
    params.add("log_delta", value=-2.28193, min=np.log(0.01), max=np.log(0.5))
    params.add("log_theta", value=2.54016, min=np.log(1), max=np.log(100.0))
    params.add("log_p", value=-0.41185, min=np.log(0.01), max=np.log(2.0))
    params.add("log_I0", value=5.24592, min=np.log(1.0), max=np.log(2000.0))
    params.add("alpha", value=0.95, min=0.1, max=1)  # α cercano a 1
    params.add("gamma", value=GAMMA, vary=False)
    params.add("kappa", value=KAPPA, vary=False)
    params.add("N", value=N_TRUE, vary=False)

    result = minimize(negative_log_likelihood_frac_tvar, params,
                      args=(t_data, y_obs, H),
                      method='lbfgsb',
                      iter_cb=monitor_progress,
                      options={'maxiter': 6000, 'ftol': 1e-12, 'gtol': 1e-9, 'eps': 1e-9})
    return result

def fit_fractional_alpha1_model(N_TRUE, t_data, y_obs, H):
    """Ajusta modelo fraccionario con alpha=1 """
    params = Parameters()
    # Usar los valores óptimos de ODEINT como iniciales
    params.add("log_beta0", value=0.91671, min=np.log(0.5), max=np.log(10.0))
    params.add("log_delta", value=-2.28193, min=np.log(0.01), max=np.log(0.5))
    params.add("log_theta", value=2.54016, min=np.log(1), max=np.log(100.0))
    params.add("log_p", value=-0.41185, min=np.log(0.01), max=np.log(2.0))
    params.add("log_I0", value=5.24592, min=np.log(1.0), max=np.log(2000.0))
    params.add("alpha", value=1.0, vary=False)
    params.add("gamma", value=GAMMA, vary=False)
    params.add("kappa", value=KAPPA, vary=False)
    params.add("N", value=N_TRUE, vary=False)

    result = minimize(negative_log_likelihood_frac_tvar, params,
                      args=(t_data, y_obs, H),
                      method='lbfgsb',
                      iter_cb=monitor_progress,
                      options={'maxiter': 6000, 'ftol': 1e-12, 'gtol': 1e-9, 'eps': 1e-9})
    return result

def fit_classic_model(N_TRUE, t_data, y_obs):
    """Ajusta modelo ODE clásico con beta variable"""
    params = Parameters()
    params.add("log_beta0", value=0.91671, min=np.log(0.5), max=np.log(10.0))
    params.add("log_delta", value=-2.28193, min=np.log(0.01), max=np.log(0.5))
    params.add("log_theta", value=2.54016, min=np.log(1), max=np.log(100.0))
    params.add("log_p", value=-0.41185, min=np.log(0.01), max=np.log(2.0))
    params.add("log_I0", value=5.24592, min=np.log(1.0), max=np.log(2000.0))
    params.add("gamma", value=GAMMA, vary=False)
    params.add("kappa", value=KAPPA, vary=False)
    params.add("N", value=N_TRUE, vary=False)

    result = minimize(negative_log_likelihood_ode_tvar, params,
                      args=(t_data, y_obs),
                      method='lbfgsb',
                      iter_cb=monitor_progress,
                      options={'maxiter': 6000, 'ftol': 1e-12, 'gtol': 1e-9, 'eps': 1e-9})
    return result

# ---------- GRÁFICOS ----------
def plot_comparison(frac_result, frac_alpha1_result, odeint_result, t_data, y_obs, H, output_dir, N_TRUE):
    """
    Compara predicciones de los tres modelos con beta variable.
    """
    def extract_params(res, alpha_fixed=None):
        raw = {k: res.params[k].value for k in res.params}
        I0 = np.exp(raw["log_I0"])
        E0 = E0_FIJO
        S0 = N_TRUE - E0 - I0
        p = {
            "beta0": np.exp(raw["log_beta0"]),
            "delta": np.exp(raw["log_delta"]),
            "theta": np.exp(raw["log_theta"]),
            "p": np.exp(raw["log_p"]),
            "I0": I0,
            "gamma": raw["gamma"],
            "kappa": raw["kappa"],
            "N": N_TRUE
        }
        if alpha_fixed is not None:
            p["alpha"] = alpha_fixed
        else:
            p["alpha"] = raw["alpha"]
        return p

    frac_params = extract_params(frac_result)
    alpha1_params = extract_params(frac_alpha1_result, alpha_fixed=1.0)
    ode_params = extract_params(odeint_result, alpha_fixed=1.0)

    sol_frac = solve_seir_frac_tvar(frac_params, frac_params["alpha"], t_data, H)
    inc_frac = np.maximum(sol_frac[:-1, 0] - sol_frac[1:, 0], 0)
    pred_frac = frac_params["p"] * inc_frac

    sol_alpha1 = solve_seir_frac_tvar(alpha1_params, 1.0, t_data, H)
    inc_alpha1 = np.maximum(sol_alpha1[:-1, 0] - sol_alpha1[1:, 0], 0)
    pred_alpha1 = alpha1_params["p"] * inc_alpha1

    sol_ode = solve_seir_odeint_tvar(ode_params, t_data)
    inc_ode = np.maximum(sol_ode[:-1, 0] - sol_ode[1:, 0], 0)
    pred_ode = ode_params["p"] * inc_ode

    n_pred = len(pred_frac)
    y_obs_plot = y_obs[:n_pred]

    plt.figure(figsize=(12, 7))
    plt.plot(y_obs_plot, 'ko', label='Datos observados')
    plt.plot(pred_frac, linewidth=3, label=f'Fraccional α={frac_params["alpha"]:.3f}')
    plt.plot(pred_alpha1, '--', linewidth=2, label='Fraccional α=1')
    plt.plot(pred_ode, ':', linewidth=3, label='ODEINT clásico')
    plt.xlabel('Semana')
    plt.ylabel('Casos reportados')
    plt.title('Comparación de modelos SEIR con β(t) exponencial')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparacion_modelos.png'), dpi=300)
    plt.close()

def save_relative_profile_plot(x, logL, mle, xlabel, filename, output_dir,
                               conf_level=0.95, cutoffs=None):
    """Guarda perfil de verosimilitud relativa"""
    max_logL = np.nanmax(logL)
    rel_lik = np.exp(logL - max_logL)

    plt.figure(figsize=(8, 5))
    plt.plot(x, rel_lik, 'k-', linewidth=1.5)
    plt.plot(x, rel_lik, 'ko', markersize=3)

    plt.axvline(mle, linestyle='--', color='k', linewidth=1,
                label=f'MLE = {mle:.4f}')

    if cutoffs is None:
        cutoffs = {0.95: 0.1465}
    for label, cutoff in cutoffs.items():
        plt.axhline(cutoff, linestyle=':', color='k', linewidth=0.8,
                    label=f'{int(label*100)}% CI (c={cutoff})')

    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel('Verosimilitud relativa', fontsize=12)
    plt.title(f'Perfil de verosimilitud relativa para {xlabel}', fontsize=12)
    plt.grid(False)
    plt.legend(loc='best', frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300, facecolor='white')
    plt.close()

def save_bivariate_contour(X, Y, logL_surf, mle_x, mle_y, xlabel, ylabel,
                           filename, output_dir, conf_levels=[0.90, 0.95, 0.99]):
    """Guarda contornos bivariados de verosimilitud"""
    max_logL = np.nanmax(logL_surf)
    rel_surf = np.exp(logL_surf - max_logL)
    delta_chi2 = {
        0.90: 4.605,
        0.95: 5.991,
        0.99: 9.210
    }

    levels = []
    level_labels = {}

    for cl in conf_levels:
        lev = np.exp(-delta_chi2[cl] / 2)
        levels.append(lev)
        level_labels[lev] = f"{int(cl*100)}%"

    levels = np.sort(levels)

    plt.figure(figsize=(8, 6))
    contour = plt.contour(X, Y, rel_surf, levels=levels, colors='k', linewidths=1.5)
    plt.clabel(contour, fmt=level_labels, inline=True, fontsize=10)
    plt.plot(mle_x, mle_y, 'ko', markersize=8)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f'Regiones de confianza conjunta para {xlabel} y {ylabel}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300, facecolor='white')
    plt.close()

# ---------- PERFILES DE VEROSIMILITUD ----------
def profile_univariate_frac_tvar(param_name, grid, mle_params, fixed_params, bounds,
                                 t_data, y_obs, H, verbose=False):
    """
    Perfil univariado para modelo fraccionario con beta variable.
    """
    logL_vals = []
    free_params = ['log_beta0', 'log_delta', 'log_theta', 'log_p', 'log_I0', 'alpha']
    current_vals = {k: mle_params[k] for k in free_params if k != param_name}

    for val in grid:
        params = Parameters()
        for k, v in fixed_params.items():
            params.add(k, value=v, vary=False)
        params.add(param_name, value=val, vary=False)

        for pname in free_params:
            if pname == param_name:
                continue
            b = bounds.get(pname, {"min": -np.inf, "max": np.inf})
            minv, maxv = b["min"], b["max"]
            init_val = current_vals.get(pname, mle_params[pname])
            params.add(pname, value=init_val, min=minv, max=maxv)

        try:
            res = minimize(negative_log_likelihood_frac_tvar, params,
                           args=(t_data, y_obs, H),
                           method='lbfgsb',
                           options={'maxiter': 2000, 'ftol': 1e-9, 'gtol': 1e-8})
            if not res.success:
                for pname in free_params:
                    if pname != param_name and pname in mle_params:
                        params[pname].value = mle_params[pname]
                res = minimize(negative_log_likelihood_frac_tvar, params,
                               args=(t_data, y_obs, H),
                               method='lbfgsb',
                               options={'maxiter': 2000, 'ftol': 1e-9, 'gtol': 1e-8})
                if not res.success and verbose:
                    print(f"Fallo en {param_name}={val:.5f}")
            if res.success:
                logL_vals.append(-res.fun)
                for pname in free_params:
                    if pname != param_name:
                        current_vals[pname] = res.params[pname].value
            else:
                logL_vals.append(np.nan)
        except Exception as e:
            if verbose:
                print(f"Error en {param_name}={val}: {e}")
            logL_vals.append(np.nan)

    return np.array(logL_vals)

def profile_univariate_ode_tvar(param_name, grid, mle_params, fixed_params, bounds,
                                t_data, y_obs, verbose=False):
    """
    Perfil univariado para modelo ODE con beta variable.
    """
    logL_vals = []
    free_params = ['log_beta0', 'log_delta', 'log_theta', 'log_p', 'log_I0']
    current_vals = {k: mle_params[k] for k in free_params if k != param_name}

    for val in grid:
        params = Parameters()
        for k, v in fixed_params.items():
            params.add(k, value=v, vary=False)
        params.add(param_name, value=val, vary=False)

        for pname in free_params:
            if pname == param_name:
                continue
            b = bounds.get(pname, {"min": -np.inf, "max": np.inf})
            minv, maxv = b["min"], b["max"]
            init_val = current_vals.get(pname, mle_params[pname])
            params.add(pname, value=init_val, min=minv, max=maxv)

        try:
            res = minimize(negative_log_likelihood_ode_tvar, params,
                           args=(t_data, y_obs),
                           method='lbfgsb',
                           options={'maxiter': 2000, 'ftol': 1e-9, 'gtol': 1e-8})
            if not res.success:
                for pname in free_params:
                    if pname != param_name and pname in mle_params:
                        params[pname].value = mle_params[pname]
                res = minimize(negative_log_likelihood_ode_tvar, params,
                               args=(t_data, y_obs),
                               method='lbfgsb',
                               options={'maxiter': 2000, 'ftol': 1e-9, 'gtol': 1e-8})
            if res.success:
                logL_vals.append(-res.fun)
                for pname in free_params:
                    if pname != param_name:
                        current_vals[pname] = res.params[pname].value
            else:
                logL_vals.append(np.nan)
        except Exception:
            logL_vals.append(np.nan)

    return np.array(logL_vals)

def bivariate_profile_frac_tvar(param1, grid1, param2, grid2, mle_params, fixed_params, bounds,
                                t_data, y_obs, H, verbose=False, parallel=False, n_cores=None):
    """
    Perfil bivariado para modelo fraccionario con beta variable.
    """
    free_params = ['log_beta0', 'log_delta', 'log_theta', 'log_p', 'log_I0', 'alpha']
    surface = np.full((len(grid1), len(grid2)), np.nan)

    if not parallel:
        total = len(grid1) * len(grid2)
        count = 0
        for i, v1 in enumerate(grid1):
            for j, v2 in enumerate(grid2):
                params = Parameters()
                for k, v in fixed_params.items():
                    params.add(k, value=v, vary=False)
                params.add(param1, value=v1, vary=False)
                params.add(param2, value=v2, vary=False)

                for pname in free_params:
                    if pname in [param1, param2]:
                        continue
                    b = bounds.get(pname, {"min": -np.inf, "max": np.inf})
                    minv, maxv = b["min"], b["max"]
                    init_val = mle_params.get(pname, 0.0)
                    params.add(pname, value=init_val, min=minv, max=maxv)

                try:
                    res = minimize(negative_log_likelihood_frac_tvar, params,
                                   args=(t_data, y_obs, H),
                                   method='lbfgsb',
                                   options={'maxiter': 800, 'ftol': 1e-8, 'gtol': 1e-6})
                    if not res.success:
                        for pname in free_params:
                            if pname not in [param1, param2]:
                                params[pname].value = mle_params[pname]
                        res = minimize(negative_log_likelihood_frac_tvar, params,
                                       args=(t_data, y_obs, H),
                                       method='lbfgsb',
                                       options={'maxiter': 800, 'ftol': 1e-8, 'gtol': 1e-6})
                    if res.success:
                        surface[i, j] = -res.fun
                except Exception as e:
                    if verbose:
                        print(f"Error en ({param1}={v1}, {param2}={v2}): {e}")
                count += 1
                if verbose and count % 50 == 0:
                    print(f"  Progreso bivariado: {count}/{total} puntos")
    else:
        # Versión paralela (simplificada)
        pass

    return surface

def bivariate_profile_ode_tvar(param1, grid1, param2, grid2, mle_params, fixed_params, bounds,
                               t_data, y_obs, verbose=False, parallel=False, n_cores=None):
    """
    Perfil bivariado para modelo ODE con beta variable.
    """
    free_params = ['log_beta0', 'log_delta', 'log_theta', 'log_p', 'log_I0']
    surface = np.full((len(grid1), len(grid2)), np.nan)

    if not parallel:
        total = len(grid1) * len(grid2)
        count = 0
        for i, v1 in enumerate(grid1):
            for j, v2 in enumerate(grid2):
                params = Parameters()
                for k, v in fixed_params.items():
                    params.add(k, value=v, vary=False)
                params.add(param1, value=v1, vary=False)
                params.add(param2, value=v2, vary=False)

                for pname in free_params:
                    if pname in [param1, param2]:
                        continue
                    b = bounds.get(pname, {"min": -np.inf, "max": np.inf})
                    minv, maxv = b["min"], b["max"]
                    init_val = mle_params.get(pname, 0.0)
                    params.add(pname, value=init_val, min=minv, max=maxv)

                try:
                    res = minimize(negative_log_likelihood_ode_tvar, params,
                                   args=(t_data, y_obs),
                                   method='lbfgsb',
                                   options={'maxiter': 800, 'ftol': 1e-8, 'gtol': 1e-6})
                    if not res.success:
                        for pname in free_params:
                            if pname not in [param1, param2]:
                                params[pname].value = mle_params[pname]
                        res = minimize(negative_log_likelihood_ode_tvar, params,
                                       args=(t_data, y_obs),
                                       method='lbfgsb',
                                       options={'maxiter': 800, 'ftol': 1e-8, 'gtol': 1e-6})
                    if res.success:
                        surface[i, j] = -res.fun
                except Exception:
                    pass
                count += 1
                if verbose and count % 50 == 0:
                    print(f"  Progreso ODE bivariado: {count}/{total} puntos")

    return surface

# ---------- FUNCIONES AUXILIARES ----------
def get_confidence_interval(x_grid, logL_profile, delta=1.92):
    """Calcula intervalo de confianza basado en perfil de verosimilitud"""
    max_logL = np.nanmax(logL_profile)
    threshold = max_logL - delta
    above = logL_profile >= threshold
    if not np.any(above):
        print("Advertencia: ningún punto supera el umbral.")
        return None, None
    indices = np.where(above)[0]
    left_idx = indices[0]
    right_idx = indices[-1]
    
    if left_idx > 0 and logL_profile[left_idx] < threshold:
        x1, x2 = x_grid[left_idx-1], x_grid[left_idx]
        y1, y2 = logL_profile[left_idx-1], logL_profile[left_idx]
        lower = x1 + (threshold - y1) * (x2 - x1) / (y2 - y1)
    else:
        lower = x_grid[left_idx]
    
    if right_idx < len(x_grid)-1 and logL_profile[right_idx] < threshold:
        x1, x2 = x_grid[right_idx], x_grid[right_idx+1]
        y1, y2 = logL_profile[right_idx], logL_profile[right_idx+1]
        upper = x1 + (threshold - y1) * (x2 - x1) / (y2 - y1)
    else:
        upper = x_grid[right_idx]
    
    return lower, upper

def extract_logL_and_k(result, param_names_free):
    """Extrae log-verosimilitud y número de parámetros"""
    logL = -result.fun
    k = len(param_names_free)
    return logL, k

def compute_aic_bic(logL, k, n_data):
    """Calcula AIC y BIC"""
    aic = 2*k - 2*logL
    bic = k*np.log(n_data) - 2*logL
    return aic, bic

def extract_params(res, alpha_fixed=None):
    """Extrae parámetros en escala real de un resultado de ajuste"""
    raw = {k: res.params[k].value for k in res.params}
    I0 = np.exp(raw["log_I0"])
    N = raw["N"]
    E0 = E0_FIJO
    S0 = N - E0 - I0
    p = {
        "beta0": np.exp(raw["log_beta0"]),
        "delta": np.exp(raw["log_delta"]),
        "theta": np.exp(raw["log_theta"]),
        "p": np.exp(raw["log_p"]),
        "I0": I0,
        "S0": S0,
        "E0": E0,
        "gamma": raw["gamma"],
        "kappa": raw["kappa"],
        "N": N
    }
    if alpha_fixed is not None:
        p["alpha"] = alpha_fixed
    else:
        p["alpha"] = raw["alpha"] if "alpha" in raw else 1.0
    return p

def print_params_real(result, model_name):
    """Imprime parámetros en escala real"""
    r = result.params
    I0 = np.exp(r['log_I0'].value)
    N = r['N'].value
    S0 = N - E0_FIJO - I0
    print(f"\n--- {model_name} ---")
    print(f"  β₀     = {np.exp(r['log_beta0'].value):.4f}")
    print(f"  δ      = {np.exp(r['log_delta'].value):.4f}")
    print(f"  θ      = {np.exp(r['log_theta'].value):.4f}")
    print(f"  p      = {np.exp(r['log_p'].value):.6f}")
    print(f"  I₀     = {I0:.1f}")
    print(f"  E₀     = {E0_FIJO:.1f} (fijo)")
    print(f"  S₀     = {S0:.0f}")
    if 'alpha' in r:
        print(f"  α      = {r['alpha'].value:.4f}")
    print(f"  γ      = {r['gamma'].value:.4f} (fijo)")
    print(f"  κ      = {r['kappa'].value:.4f} (fijo)")
    print(f"  N      = {r['N'].value:.0f} (fijo)")

# ---------- PRUEBA DE HIPÓTESIS ----------
def likelihood_ratio_test(frac_result, ord_result):
    """Test de razón de verosimilitud entre modelo fraccionario y ordinario"""
    logL_frac = -frac_result.fun
    logL_ord = -ord_result.fun
    df = 1  # alpha es el parámetro extra
    lrt_stat = 2 * (logL_frac - logL_ord)
    p_value = 1 - stats.chi2.cdf(lrt_stat, df)
    return {
        'logL_frac': logL_frac,
        'logL_ord': logL_ord,
        'lrt_stat': lrt_stat,
        'df': df,
        'p_value': p_value,
        'reject_H0': p_value < 0.05
    }

def compare_models_aic_bic(frac_result, ord_result, n_data):
    """Compara modelos usando AIC y BIC"""
    logL_frac = -frac_result.fun
    logL_ord = -ord_result.fun
    # Parámetros libres: fraccionario tiene 6 (log_beta0, log_delta, log_theta, log_p, log_I0, alpha)
    # Ordinario (α=1) tiene 5 (los mismos sin alpha)
    k_frac = 6
    k_ord = 5
    aic_frac = 2*k_frac - 2*logL_frac
    bic_frac = k_frac * np.log(n_data) - 2*logL_frac
    aic_ord = 2*k_ord - 2*logL_ord
    bic_ord = k_ord * np.log(n_data) - 2*logL_ord
    return {
        'frac': {'AIC': aic_frac, 'BIC': bic_frac, 'logL': logL_frac, 'k': k_frac},
        'ord':  {'AIC': aic_ord, 'BIC': bic_ord, 'logL': logL_ord, 'k': k_ord}
    }
def explore_likelihood_landscape(t_data, y_obs, H, N_TRUE, alpha_range=[0.1, 0.3, 0.5, 0.7, 0.9]):
    """
    Explora el paisaje de verosimilitud para diferentes valores de α
    """
    results = {}
    
    print("\n" + "="*70)
    print("EXPLORACIÓN DEL PAISAJE DE VEROSIMILITUD")
    print("="*70)
    print(f"{'α':<8} {'Loss final':<12} {'β₀':<10} {'δ':<10} {'p':<10} {'I₀':<10}")
    print("-"*70)
    
    for alpha_init in alpha_range:
        # Ajustar con este α inicial
        params = Parameters()
        params.add("log_beta0", value=np.log(2.58), min=np.log(0.5), max=np.log(10.0))
        params.add("log_delta", value=np.log(0.09), min=np.log(0.01), max=np.log(0.5))
        params.add("log_theta", value=np.log(13), min=np.log(1), max=np.log(100.0))
        params.add("log_p", value=np.log(0.63), min=np.log(0.01), max=np.log(2.0))
        params.add("log_I0", value=np.log(185), min=np.log(1.0), max=np.log(2000.0))
        params.add("alpha", value=alpha_init, min=0.1, max=1)
        params.add("gamma", value=GAMMA, vary=False)
        params.add("kappa", value=KAPPA, vary=False)
        params.add("N", value=N_TRUE, vary=False)
        
        result = minimize(negative_log_likelihood_frac_tvar, params,
                          args=(t_data, y_obs, H),
                          method='lbfgsb',
                          options={'maxiter': 3000, 'ftol': 1e-12, 'gtol': 1e-9})
        
        alpha_final = result.params['alpha'].value
        beta0 = np.exp(result.params['log_beta0'].value)
        delta = np.exp(result.params['log_delta'].value)
        p = np.exp(result.params['log_p'].value)
        I0 = np.exp(result.params['log_I0'].value)
        loss = result.fun
        
        print(f"{alpha_init:<8.2f} {loss:<12.4f} {beta0:<10.4f} {delta:<10.4f} {p:<10.6f} {I0:<10.1f}")
        
        results[alpha_init] = {
            'alpha_final': alpha_final,
            'beta0': beta0,
            'delta': delta,
            'p': p,
            'I0': I0,
            'loss': loss,
            'result': result
        }
    
    print("="*70)
    
    # Verificar si hay múltiples mínimos
    losses = [results[a]['loss'] for a in alpha_range]
    min_loss = min(losses)
    
    print("\nMÍNIMOS LOCALES ENCONTRADOS:")
    for a in alpha_range:
        diff = results[a]['loss'] - min_loss
        if diff < 0.1:
            print(f"  α_inicial={a:.2f} → α_final={results[a]['alpha_final']:.4f} (loss={results[a]['loss']:.4f}) ⭐ MÍNIMO")
        elif diff < 1.0:
            print(f"  α_inicial={a:.2f} → α_final={results[a]['alpha_final']:.4f} (loss={results[a]['loss']:.4f})")
        else:
            print(f"  α_inicial={a:.2f} → α_final={results[a]['alpha_final']:.4f} (loss={results[a]['loss']:.4f}) ❌ MÍNIMO LOCAL")
    
    return results

def diagnostic_solver_comparison(t_data, params, alpha_values=[ 0.9, 0.95, 0.99, 1.0]):
    """
    Compara el solver fraccionario con ODEINT para diferentes valores de alpha
    """
    beta0 = params["beta0"]
    delta = params["delta"]
    gamma = params["gamma"]
    kappa = params["kappa"]
    N = params["N"]
    I0 = params["I0"]
    E0 = E0_FIJO
    S0 = N - E0 - I0
    
    y0 = np.array([S0, E0, I0, 0.0])
    T = t_data[-1]
    h = 0.005  # Paso fino para precisión
    
    # Solución ODEINT (referencia)
    def deriv(y, t):
        S, E, I, R = y
        beta = beta0 * np.exp(-delta * t)
        dS = -beta * S * I / N
        dE = beta * S * I / N - kappa * E
        dI = kappa * E - gamma * I
        dR = gamma * I
        return [dS, dE, dI, dR]
    
    sol_ode = odeint(deriv, y0, t_data)
    S_ode = sol_ode[:, 0]
    
    print("\n" + "="*70)
    print("DIAGNÓSTICO: COMPARACIÓN DE SOLVERS")
    print("="*70)
    print(f"{'α':<8} {'Error S(t)':<15} {'Error Incidencia':<18} {'Converge a ODE?'}")
    print("-"*70)
    
    results = {}
    
    for alpha in alpha_values:
        # Solución fraccionaria
        t_fine, sol_frac = solver_fraccionario(
            lambda t, Y: seir_rhs_tvar(t, Y, beta0, delta, gamma, kappa, N),
            alpha, y0, T, h
        )
        
        # Interpolar en tiempos de observación
        interp = interp1d(t_fine, sol_frac, axis=0, kind='linear',
                          bounds_error=False, fill_value=(sol_frac[0], sol_frac[-1]))
        sol_frac_interp = interp(t_data)
        S_frac = sol_frac_interp[:, 0]
        
        # Calcular incidencia
        inc_frac = np.maximum(S_frac[:-1] - S_frac[1:], 0)
        inc_ode = np.maximum(S_ode[:-1] - S_ode[1:], 0)
        
        # Errores
        error_S = np.max(np.abs(S_frac - S_ode)) / np.max(S_ode) * 100
        error_inc = np.max(np.abs(inc_frac - inc_ode)) / (np.max(inc_ode) + 1e-8) * 100
        
        # Verificar si converge
        converge = error_S < 0.1 and error_inc < 0.1
        
        print(f"{alpha:<8.3f} {error_S:<14.2f}% {error_inc:<17.2f}% {converge}")
        
        results[alpha] = {
            'S_frac': S_frac,
            'inc_frac': inc_frac,
            'error_S': error_S,
            'error_inc': error_inc,
            'converge': converge
        }
    
    print("="*70)
    
    # Gráfico de diagnóstico
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Susceptibles
    ax1 = axes[0]
    ax1.plot(t_data, S_ode, 'k-', linewidth=2, label='ODEINT (referencia)')
    for alpha in alpha_values:
        if alpha in results:
            ax1.plot(t_data, results[alpha]['S_frac'], '--', 
                    label=f'α={alpha:.2f}', alpha=0.7)
    ax1.set_xlabel('Tiempo')
    ax1.set_ylabel('Susceptibles (S)')
    ax1.set_title('Comparación de S(t) entre solvers')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Incidencia
    ax2 = axes[1]
    ax2.plot(t_data[:-1], inc_ode, 'k-', linewidth=2, label='ODEINT (referencia)')
    for alpha in alpha_values:
        if alpha in results:
            ax2.plot(t_data[:-1], results[alpha]['inc_frac'], '--',
                    label=f'α={alpha:.2f}', alpha=0.7)
    ax2.set_xlabel('Tiempo')
    ax2.set_ylabel('Incidencia')
    ax2.set_title('Comparación de Incidencia entre solvers')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('diagnostic_solver_comparison.png', dpi=300)
    plt.close()
    
    return results