# run_seir_jalisco_betavar.py
# Ejecución del análisis SEIR fraccionario vs clásico con β(t) variable
# Versión enfocada en perfiles de verosimilitud

import os
import numpy as np
import matplotlib.pyplot as plt
from lmfit import fit_report
from seir_lib_mm import *

# Directorio y valores
OUTPUT_DIR = "outputs_seir_jalisco_betavar"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    # CONFIGURACIÓN INICIAL
    H = 0.05  # Paso de integración 
    N_TRUE = 8_800_000 * (1 - 0.824)  # Población susceptible efectiva
    print(f"Población susceptible efectiva: {N_TRUE:.0f}")
    print(f"Paso de integración (H): {H}")

    # Datos semanales observados (casos reportados)
    y_obs = np.array([172, 190, 235, 427, 497, 566, 753, 524, 520, 475, 
                      273, 309, 153, 269, 170, 223, 180, 137, 97, 91])
    t_data = np.arange(0, len(y_obs) + 1, 1)  # 0,1,2,...,20 (semanas)
    
    print(f"Número de observaciones: {len(y_obs)}")
    print(f"Rango de tiempo: {t_data[0]} a {t_data[-1]} semanas")

    #Verificar consistencia del solver
    print("\n" + "="*70)
    print("DIAGNÓSTICO DEL SOLVER")
    print("="*70)
    
    # Usar parámetros típicos para diagnóstico
    params_test = {
        "beta0": 2.5,
        "delta": 0.1,
        "gamma": GAMMA,
        "kappa": KAPPA,
        "N": N_TRUE,
        "I0": 185
    }
    
    print("\nComparando solvers para diferentes valores de α...")
    diagnostic_results = diagnostic_solver_comparison(t_data, params_test, 
                                                       alpha_values=[ 0.9, 0.95, 0.99, 1.0])
    
    # Verificar si α=1 es consistente con ODEINT
    if 1.0 in diagnostic_results:
        error_S = diagnostic_results[1.0]['error_S']
        error_inc = diagnostic_results[1.0]['error_inc']
        print(f"\n Error para α=1:")
        print(f"  Error en S(t): {error_S:.4f}%")
        print(f"  Error en incidencia: {error_inc:.4f}%")
      

    # AJUSTE DEL MODELO FRACCIONARIO PARA OBTENER PARÁMETROS MLE
    print('\n' + '='*70)
    print('AJUSTANDO MODELO FRACCIONARIO CON β(t) (α libre)')
    print('='*70)
    frac_result = fit_fractional_model(N_TRUE, t_data, y_obs, H)

    print('\n' + '='*70)
    print('AJUSTANDO MODELO ODEINT CLÁSICO CON β(t)')
    print('='*70)
    odeint_result = fit_classic_model(N_TRUE, t_data, y_obs)

    # REPORTE DE RESULTADOS
    print('\n--- REPORTE FRACCIONARIO CON β(t) ---')
    print(fit_report(frac_result))
    print_params_real(frac_result, "Fraccional β(t) (α libre)")

    print('\n--- REPORTE ODEINT CLÁSICO CON β(t) ---')
    print(fit_report(odeint_result))
    print_params_real(odeint_result, "ODEINT β(t) clásico")

    # PARÁMETROS PARA PERFILES
    
    # Parámetros MLE del modelo fraccionario
    frac_raw = {k: frac_result.params[k].value for k in frac_result.params}
    I0_frac = np.exp(frac_raw["log_I0"])
    mle_frac_real = {
        "beta0": np.exp(frac_raw["log_beta0"]),
        "delta": np.exp(frac_raw["log_delta"]),
        "theta": np.exp(frac_raw["log_theta"]),
        "p": np.exp(frac_raw["log_p"]),
        "I0": I0_frac,
        "S0": N_TRUE - E0_FIJO - I0_frac,
        "alpha": frac_raw["alpha"]
    }
    mle_frac_log = {
        "log_beta0": frac_raw["log_beta0"],
        "log_delta": frac_raw["log_delta"],
        "log_theta": frac_raw["log_theta"],
        "log_p": frac_raw["log_p"],
        "log_I0": frac_raw["log_I0"],
        "alpha": frac_raw["alpha"]
    }

    # Parámetros MLE de ODEINT (para inicialización)
    ode_raw = {k: odeint_result.params[k].value for k in odeint_result.params}
    mle_ode_log = {
        "log_beta0": ode_raw["log_beta0"],
        "log_delta": ode_raw["log_delta"],
        "log_theta": ode_raw["log_theta"],
        "log_p": ode_raw["log_p"],
        "log_I0": ode_raw["log_I0"],
    }

    # Parámetros fijos para perfiles
    fixed_params = {
        "gamma": frac_result.params["gamma"].value,
        "kappa": frac_result.params["kappa"].value,
        "N": N_TRUE,
    }

    # Límites para perfiles
    bounds_log = {
        "log_beta0": {"min": np.log(0.5), "max": np.log(10.0)},
        "log_delta": {"min": np.log(0.01), "max": np.log(0.5)},
        "log_theta": {"min": np.log(1), "max": np.log(100.0)},
        "log_p": {"min": np.log(0.01), "max": np.log(2.0)},
        "log_I0": {"min": np.log(1.0), "max": np.log(2000.0)},
        "alpha": {"min": 0.1, "max": 0.99}
    }

    # PERFILES BIVARIADOS - MODELO FRACCIONARIO β(t)
    print("PERFILES BIVARIADOS - MODELO FRACCIONARIO CON β(t)")
    
    n_fr = 25  # Número de puntos para perfiles bivariados (más fino)
    
    # Grillas para perfiles bivariados - centradas en los MLE
    beta0_log_biv = np.linspace(mle_frac_log["log_beta0"] - 0.5, 
                                mle_frac_log["log_beta0"] + 0.5, n_fr)
    delta_log_biv = np.linspace(mle_frac_log["log_delta"] - 0.5, 
                                mle_frac_log["log_delta"] + 0.5, n_fr)
    alpha_biv = np.linspace(0.7, 1.0, n_fr)
    p_log_biv = np.linspace(mle_frac_log["log_p"] - 0.5, 
                            mle_frac_log["log_p"] + 0.5, n_fr)
    I0_log_biv = np.linspace(mle_frac_log["log_I0"] - 0.8, 
                             mle_frac_log["log_I0"] + 0.8, n_fr)

    # Perfil α vs β₀
    print("\n1. Perfil bivariado: α vs β₀")
    Z_alpha_beta0 = bivariate_profile_frac_tvar(
        'alpha', alpha_biv, 'log_beta0', beta0_log_biv,
        mle_frac_log, fixed_params, bounds_log, t_data, y_obs, H, verbose=True)
    beta0_real_biv = np.exp(beta0_log_biv)
    X_ab, Y_ab = np.meshgrid(beta0_real_biv, alpha_biv)
    save_bivariate_contour(X_ab, Y_ab, Z_alpha_beta0,
                           mle_frac_real["beta0"], mle_frac_real["alpha"],
                           'β₀', 'α', 'perfil_beta0_alpha.png', OUTPUT_DIR)

    # Perfil α vs δ
    print("\n2. Perfil bivariado: α vs δ")
    Z_alpha_delta = bivariate_profile_frac_tvar(
        'alpha', alpha_biv, 'log_delta', delta_log_biv,
        mle_frac_log, fixed_params, bounds_log, t_data, y_obs, H, verbose=True)
    delta_real_biv = np.exp(delta_log_biv)
    X_ad, Y_ad = np.meshgrid(delta_real_biv, alpha_biv)
    save_bivariate_contour(X_ad, Y_ad, Z_alpha_delta,
                           mle_frac_real["delta"], mle_frac_real["alpha"],
                           'δ', 'α', 'perfil_delta_alpha.png', OUTPUT_DIR)

    # Perfil β₀ vs δ
    print("\n3. Perfil bivariado: β₀ vs δ")
    Z_beta0_delta = bivariate_profile_frac_tvar(
        'log_beta0', beta0_log_biv, 'log_delta', delta_log_biv,
        mle_frac_log, fixed_params, bounds_log, t_data, y_obs, H, verbose=True)
    X_bd, Y_bd = np.meshgrid(delta_real_biv, beta0_real_biv)
    save_bivariate_contour(X_bd, Y_bd, Z_beta0_delta,
                           mle_frac_real["delta"], mle_frac_real["beta0"],
                           'δ', 'β₀', 'perfil_beta0_delta.png', OUTPUT_DIR)

    # Perfil β₀ vs p
    print("\n4. Perfil bivariado: β₀ vs p")
    Z_beta0_p = bivariate_profile_frac_tvar(
        'log_beta0', beta0_log_biv, 'log_p', p_log_biv,
        mle_frac_log, fixed_params, bounds_log, t_data, y_obs, H, verbose=True)
    p_real_biv = np.exp(p_log_biv)
    X_bp, Y_bp = np.meshgrid(p_real_biv, beta0_real_biv)
    save_bivariate_contour(X_bp, Y_bp, Z_beta0_p,
                           mle_frac_real["p"], mle_frac_real["beta0"],
                           'p', 'β₀', 'perfil_beta0_p.png', OUTPUT_DIR)

    # Perfil p vs I₀ (MUESTRA LA COMPENSACIÓN)
    print("\n5. Perfil bivariado: p vs I₀ (¡compensación!)")
    Z_p_I0 = bivariate_profile_frac_tvar(
        'log_p', p_log_biv, 'log_I0', I0_log_biv,
        mle_frac_log, fixed_params, bounds_log, t_data, y_obs, H, verbose=True)
    I0_real_biv = np.exp(I0_log_biv)
    X_pi, Y_pi = np.meshgrid(I0_real_biv, p_real_biv)
    save_bivariate_contour(X_pi, Y_pi, Z_p_I0,
                           mle_frac_real["I0"], mle_frac_real["p"],
                           'I₀', 'p', 'perfil_I0_p.png', OUTPUT_DIR)

    # PERFILES UNIVARIADOS - MODELO FRACCIONARIO β(t)
    print("\n" + "=" * 70)
    print("PERFILES UNIVARIADOS - MODELO FRACCIONARIO CON β(t)")
    print("=" * 70)
    
    n_uni = 50  # Número de puntos para perfiles univariados
    
    # Perfil para alpha
    print("\n1. Perfil univariado: α")
    alpha_grid = np.linspace(0.5, 1.0, n_uni)
    logL_alpha = profile_univariate_frac_tvar('alpha', alpha_grid, mle_frac_log, 
                                              fixed_params, bounds_log, t_data, y_obs, H, verbose=True)
    alpha_lower, alpha_upper = get_confidence_interval(alpha_grid, logL_alpha, delta=1.92)
    if alpha_lower is not None:
        print(f"  IC 95% para α: [{alpha_lower:.4f}, {alpha_upper:.4f}]")
        print(f"  MLE α: {mle_frac_real['alpha']:.4f}")
    else:
        print("  No se pudo calcular IC para α")
    save_relative_profile_plot(alpha_grid, logL_alpha, mle_frac_real["alpha"], 
                               'α', 'profile_alpha.png', OUTPUT_DIR)

    # Perfil para β₀
    print("\n2. Perfil univariado: β₀")
    beta0_log_grid = np.linspace(mle_frac_log["log_beta0"] - 0.5, 
                                 mle_frac_log["log_beta0"] + 0.5, n_uni)
    logL_beta0 = profile_univariate_frac_tvar('log_beta0', beta0_log_grid, mle_frac_log,
                                              fixed_params, bounds_log, t_data, y_obs, H, verbose=True)
    beta0_lower_log, beta0_upper_log = get_confidence_interval(beta0_log_grid, logL_beta0, delta=1.92)
    if beta0_lower_log is not None:
        beta0_lower, beta0_upper = np.exp(beta0_lower_log), np.exp(beta0_upper_log)
        print(f"  IC 95% para β₀: [{beta0_lower:.4f}, {beta0_upper:.4f}]")
        print(f"  MLE β₀: {mle_frac_real['beta0']:.4f}")
    else:
        print("  No se pudo calcular IC para β₀")
    save_relative_profile_plot(beta0_log_grid, logL_beta0, mle_frac_log["log_beta0"],
                               'log(β₀)', 'profile_log_beta0.png', OUTPUT_DIR)

    # Perfil para δ
    print("\n3. Perfil univariado: δ")
    delta_log_grid = np.linspace(mle_frac_log["log_delta"] - 0.5, 
                                 mle_frac_log["log_delta"] + 0.5, n_uni)
    logL_delta = profile_univariate_frac_tvar('log_delta', delta_log_grid, mle_frac_log,
                                              fixed_params, bounds_log, t_data, y_obs, H, verbose=True)
    delta_lower_log, delta_upper_log = get_confidence_interval(delta_log_grid, logL_delta, delta=1.92)
    if delta_lower_log is not None:
        delta_lower, delta_upper = np.exp(delta_lower_log), np.exp(delta_upper_log)
        print(f"  IC 95% para δ: [{delta_lower:.4f}, {delta_upper:.4f}]")
        print(f"  MLE δ: {mle_frac_real['delta']:.4f}")
    else:
        print("  No se pudo calcular IC para δ")
    save_relative_profile_plot(delta_log_grid, logL_delta, mle_frac_log["log_delta"],
                               'log(δ)', 'profile_log_delta.png', OUTPUT_DIR)

    # Perfil para p
    print("\n4. Perfil univariado: p")
    p_log_grid = np.linspace(mle_frac_log["log_p"] - 0.5, 
                             mle_frac_log["log_p"] + 0.5, n_uni)
    logL_p = profile_univariate_frac_tvar('log_p', p_log_grid, mle_frac_log,
                                          fixed_params, bounds_log, t_data, y_obs, H, verbose=True)
    p_lower_log, p_upper_log = get_confidence_interval(p_log_grid, logL_p, delta=1.92)
    if p_lower_log is not None:
        p_lower, p_upper = np.exp(p_lower_log), np.exp(p_upper_log)
        print(f"  IC 95% para p: [{p_lower:.4f}, {p_upper:.4f}]")
        print(f"  MLE p: {mle_frac_real['p']:.4f}")
    else:
        print("  No se pudo calcular IC para p")
    save_relative_profile_plot(p_log_grid, logL_p, mle_frac_log["log_p"],
                               'log(p)', 'profile_log_p.png', OUTPUT_DIR)

    # Perfil para I₀
    print("\n5. Perfil univariado: I₀")
    I0_log_grid = np.linspace(mle_frac_log["log_I0"] - 1.0, 
                              mle_frac_log["log_I0"] + 1.0, n_uni)
    logL_I0 = profile_univariate_frac_tvar('log_I0', I0_log_grid, mle_frac_log,
                                           fixed_params, bounds_log, t_data, y_obs, H, verbose=True)
    I0_lower_log, I0_upper_log = get_confidence_interval(I0_log_grid, logL_I0, delta=1.92)
    if I0_lower_log is not None:
        I0_lower, I0_upper = np.exp(I0_lower_log), np.exp(I0_upper_log)
        print(f"  IC 95% para I₀: [{I0_lower:.1f}, {I0_upper:.1f}]")
        print(f"  MLE I₀: {mle_frac_real['I0']:.1f}")
    else:
        print("  No se pudo calcular IC para I₀")
    save_relative_profile_plot(I0_log_grid, logL_I0, mle_frac_log["log_I0"],
                               'log(I₀)', 'profile_log_I0.png', OUTPUT_DIR)

    # Perfil para θ
    print("\n6. Perfil univariado: θ")
    theta_log_grid = np.linspace(mle_frac_log["log_theta"] - 0.8, 
                                 mle_frac_log["log_theta"] + 0.8, n_uni)
    logL_theta = profile_univariate_frac_tvar('log_theta', theta_log_grid, mle_frac_log,
                                              fixed_params, bounds_log, t_data, y_obs, H, verbose=True)
    theta_lower_log, theta_upper_log = get_confidence_interval(theta_log_grid, logL_theta, delta=1.92)
    if theta_lower_log is not None:
        theta_lower, theta_upper = np.exp(theta_lower_log), np.exp(theta_upper_log)
        print(f"  IC 95% para θ: [{theta_lower:.2f}, {theta_upper:.2f}]")
        print(f"  MLE θ: {mle_frac_real['theta']:.2f}")
    else:
        print("  No se pudo calcular IC para θ")
    save_relative_profile_plot(theta_log_grid, logL_theta, mle_frac_log["log_theta"],
                               'log(θ)', 'profile_log_theta.png', OUTPUT_DIR)

    # RESUMEN 
    print("\n" + "=" * 70)
    print("RESUMEN DE RESULTADOS - MODELO CON β(t)")
    print("=" * 70)
    print(f"\nParámetros estimados (modelo fraccionario):")
    print(f"  β₀    = {mle_frac_real['beta0']:.4f}")
    print(f"  δ     = {mle_frac_real['delta']:.4f}")
    print(f"  α     = {mle_frac_real['alpha']:.4f}")
    print(f"  p     = {mle_frac_real['p']:.4f}")
    print(f"  I₀    = {mle_frac_real['I0']:.1f}")
    print(f"  θ     = {mle_frac_real['theta']:.2f}")
    
    # Calcular beta inicial y final
    beta_inicial = mle_frac_real['beta0']
    beta_final = mle_frac_real['beta0'] * np.exp(-mle_frac_real['delta'] * (len(y_obs) - 1))
    print(f"\nEvolución de β(t) = β₀ * exp(-δ * t):")
    print(f"  β(0)  = {beta_inicial:.4f}")
    print(f"  β({len(y_obs)-1}) = {beta_final:.4f}")
    print(f"  Reducción: {(1 - beta_final/beta_inicial)*100:.1f}%")
    
    # R₀ inicial y final
    R0_inicial = beta_inicial / GAMMA
    R0_final = beta_final / GAMMA
    print(f"\nNúmero reproductivo básico R₀:")
    print(f"  R₀(0)  = {R0_inicial:.3f}")
    print(f"  R₀({len(y_obs)-1}) = {R0_final:.3f}")

    print(f"\nTodos los resultados guardados en: {OUTPUT_DIR}")
    print("   Perfiles bivariados: perfil_*.png")
    print("   Perfiles univariados: profile_*.png")

if __name__ == "__main__":
    main()