# run_seir_jalisco.py
# Ejecución del análisis SEIR fraccionario vs clásico

import os
import numpy as np
import matplotlib.pyplot as plt
from lmfit import fit_report
from seir_lib_Sobreajuste import *

# Directorio y valores
OUTPUT_DIR = "outputs_seir_jalisco"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    H = 0.05
    N_TRUE = 8_800_000 * (1 - 0.824)  
    print(f"Población susceptible efectiva: {N_TRUE:.0f}")

    # Datos semanales observados
    y_obs = np.array([172, 190, 235, 427, 497, 566, 753, 524, 520, 475, 
                      273, 309, 153, 269, 170, 223, 180, 137, 97, 91])
    t_data = np.arange(0, len(y_obs) + 1, 1)   # 0,1,2,...,20

    # ---------- AJUSTES ----------
    print('\nAJUSTANDO MODELO FRACCIONARIO (α libre)')
    frac_result = fit_fractional_model(N_TRUE, t_data, y_obs, H)

    print('\nAJUSTANDO FRACCIONARIO α=1')
    frac_alpha1_result = fit_fractional_alpha1_model(N_TRUE, t_data, y_obs, H)

    print('\nAJUSTANDO MODELO CLÁSICO ODEINT')
    odeint_result = fit_classic_model(N_TRUE, t_data, y_obs)

    # Reportes
    print('\n--- REPORTE FRACCIONARIO ---')
    print(fit_report(frac_result))
    print_params_real(frac_result, "Fraccional (α libre)")

    print('\n--- REPORTE FRACCIONARIO α=1 ---')
    print(fit_report(frac_alpha1_result))
    print_params_real(frac_alpha1_result, "Fraccional (α=1)")

    print('\n--- REPORTE ODEINT CLÁSICO ---')
    print(fit_report(odeint_result))
    print_params_real(odeint_result, "ODEINT clásico")

    # Gráfica de comparación
    plot_comparison(frac_result, frac_alpha1_result, odeint_result, t_data, y_obs, H, OUTPUT_DIR, N_TRUE)

    # Número de parámetros libres
    k_frac = 5      # log_beta, log_theta, log_p, log_I0, alpha
    k_ord = 4       # log_beta, log_theta, log_p, log_I0

    logL_frac = -frac_result.fun
    logL_frac1 = -frac_alpha1_result.fun
    logL_ode = -odeint_result.fun

    n_data = len(y_obs)
    aic_frac, bic_frac = compute_aic_bic(logL_frac, k_frac, n_data)
    aic_frac1, bic_frac1 = compute_aic_bic(logL_frac1, k_ord, n_data)   
    aic_ode, bic_ode = compute_aic_bic(logL_ode, k_ord, n_data)

    print("\n CRITERIOS DE INFORMACIÓN") 
    print(f"{'Modelo':<25} {'logL':>10} {'k':>4} {'AIC':>10} {'BIC':>10}")
    print(f"{'Fraccional (α libre)':<25} {logL_frac:10.2f} {k_frac:4} {aic_frac:10.2f} {bic_frac:10.2f}")
    print(f"{'Fraccional (α=1)':<25} {logL_frac1:10.2f} {k_ord:4} {aic_frac1:10.2f} {bic_frac1:10.2f}")
    print(f"{'ODEINT clásico':<25} {logL_ode:10.2f} {k_ord:4} {aic_ode:10.2f} {bic_ode:10.2f}")

    # Prueba de hipótesis (Fraccional vs α=1)
    lrt = likelihood_ratio_test(frac_result, frac_alpha1_result)
    print("\n LIKELIHOOD RATIO TEST (Fraccional vs α=1)" ) 
    print(f"LRT = {lrt['lrt_stat']:.4f}, df = {lrt['df']}, p-value = {lrt['p_value']:.5f}")

    # ---------- PREPARAR PARÁMETROS MLE EN ESCALA LOG Y REAL ----------
    # Para el modelo fraccionario (el único con alpha libre)
    frac_raw = {k: frac_result.params[k].value for k in frac_result.params}
    I0_frac = np.exp(frac_raw["log_I0"])
    mle_frac_real = {
        "beta": np.exp(frac_raw["log_beta"]),
        "theta": np.exp(frac_raw["log_theta"]),
        "p": np.exp(frac_raw["log_p"]),
        "I0": I0_frac,
        "S0": N_TRUE - E0_FIJO - I0_frac,
        "alpha": frac_raw["alpha"]
    }
    mle_frac_log = {
        "log_beta": frac_raw["log_beta"],
        "log_theta": frac_raw["log_theta"],
        "log_p": frac_raw["log_p"],
        "log_I0": frac_raw["log_I0"],
        "alpha": frac_raw["alpha"]
    }

    # Parámetros fijos para perfiles (gamma, kappa, N)
    fixed_params = {
        "gamma": frac_result.params["gamma"].value,
        "kappa": frac_result.params["kappa"].value,
        "N": N_TRUE,
    }

    # Límites unificados para los perfiles
    bounds_log = {
        "log_beta": {"min": np.log(1.0), "max": np.log(30.0)},
        "log_theta": {"min": np.log(0.01), "max": np.log(100.0)},
        "log_p": {"min": np.log(1e-5), "max": np.log(1.0)},
        "log_I0": {"min": np.log(1.0), "max": np.log(50000.0)},
        "alpha": {"min": 0.01, "max": 0.99}
    }

    # Parámetros MLE para ODEINT
    ode_raw = {k: odeint_result.params[k].value for k in odeint_result.params}
    mle_ode_log = {
        "log_beta": ode_raw["log_beta"],
        "log_theta": ode_raw["log_theta"],
        "log_p": ode_raw["log_p"],
        "log_I0": ode_raw["log_I0"],
    }
    I0_ode = np.exp(mle_ode_log["log_I0"])
    mle_ode_real = {
        "beta": np.exp(mle_ode_log["log_beta"]),
        "theta": np.exp(mle_ode_log["log_theta"]),
        "p": np.exp(mle_ode_log["log_p"]),
        "I0": I0_ode,
        "S0": N_TRUE - E0_FIJO - I0_ode,
    }
    # ---------- PERFILES BIVARIADOS FRACCIONARIO ----------
    n_fr = 20
    beta_log_biv = np.linspace(mle_frac_log["log_beta"] - 0.5, mle_frac_log["log_beta"] + 0.5, n_fr)
    alpha_biv = np.linspace(mle_frac_log["alpha"] - 0.2, mle_frac_log["alpha"] + 0.2, n_fr)
    p_log_biv = np.linspace(mle_frac_log["log_p"] - 0.8, mle_frac_log["log_p"] + 0.7, n_fr)
    I0_log_biv = np.linspace(mle_frac_log["log_I0"] - 0.8, mle_frac_log["log_I0"] + 1, n_fr)

    print("\n--- Perfiles bivariados fraccionarios ---")
    print("Perfil α-β")
    Z_beta_alpha = bivariate_profile_log('log_beta', beta_log_biv, 'alpha', alpha_biv,
                                         mle_frac_log, fixed_params, t_data, y_obs, H, verbose=True)
    beta_real_biv = np.exp(beta_log_biv)
    X_ba, Y_ba = np.meshgrid(alpha_biv, beta_real_biv)
    save_bivariate_contour(X_ba, Y_ba, Z_beta_alpha,
                           mle_frac_real["alpha"], mle_frac_real["beta"],
                           'α', 'β', 'perfil_beta_alpha.png', OUTPUT_DIR)

    print("Perfil α-p")
    Z_alpha_p = bivariate_profile_log('alpha', alpha_biv, 'log_p', p_log_biv,
                                      mle_frac_log, fixed_params, t_data, y_obs, H, verbose=False)
    p_real_biv = np.exp(p_log_biv)
    X_ap, Y_ap = np.meshgrid(p_real_biv, alpha_biv)
    save_bivariate_contour(X_ap, Y_ap, Z_alpha_p,
                           mle_frac_real["p"], mle_frac_real["alpha"],
                           'p', 'α', 'profile_alpha_p.png', OUTPUT_DIR)

    print("Perfil β-p")
    Z_beta_p = bivariate_profile_log('log_beta', beta_log_biv, 'log_p', p_log_biv,
                                     mle_frac_log, fixed_params, t_data, y_obs, H, verbose=False)
    X_bp, Y_bp = np.meshgrid(p_real_biv, beta_real_biv)
    save_bivariate_contour(X_bp, Y_bp, Z_beta_p,
                           mle_frac_real["p"], mle_frac_real["beta"],
                           'p', 'β', 'profile_beta_p.png', OUTPUT_DIR)

    print("Perfil I₀-p")
    Z_I0_p = bivariate_profile_log('log_I0', I0_log_biv, 'log_p', p_log_biv,
                                   mle_frac_log, fixed_params, t_data, y_obs, H, verbose=False)
    I0_real_biv = np.exp(I0_log_biv)
    X_ip, Y_ip = np.meshgrid(p_real_biv, I0_real_biv)
    save_bivariate_contour(X_ip, Y_ip, Z_I0_p,
                           mle_frac_real["p"], mle_frac_real["I0"],
                           'p', 'I₀', 'profile_I0_p.png', OUTPUT_DIR)


    # ---------- PERFILES BIVARIADOS ODEINT ----------
    n_ode = 100
    beta_log_biv_ode = np.linspace(mle_ode_log["log_beta"] - 0.5, mle_ode_log["log_beta"] + 0.5, n_ode)
    p_real_biv_ode = np.linspace(0.004,0.007, n_ode)
    p_log_biv_ode = np.log(p_real_biv_ode)
    I0_real_biv_ode = np.linspace(20000,70000, n_ode)
    I0_log_biv_ode = np.log(I0_real_biv_ode)
    theta_log_biv_ode = np.linspace(mle_ode_log["log_theta"] - 1.0, mle_ode_log["log_theta"] + 1.0, n_ode)

    print("\n--- Perfiles bivariados ODEINT ---")

    print("Zode_PI0")
    Z_ode_pI = bivariate_profile_odeint_log('log_p', p_log_biv_ode, 'log_I0', I0_log_biv_ode,
                                            mle_ode_log, fixed_params, t_data, y_obs, 
                                            verbose=False, parallel= False)
    X_pI, Y_pI = np.meshgrid(I0_real_biv_ode,p_real_biv_ode)
    save_bivariate_contour(X_pI, Y_pI, Z_ode_pI,
                           mle_ode_real["I0"], mle_ode_real["p"],
                           'I₀', 'p', 'perfil_ode_p_I0.png', OUTPUT_DIR)
    print("Zode_bp")
    Z_ode_bp = bivariate_profile_odeint_log('log_beta', beta_log_biv_ode, 'log_p', p_log_biv_ode,
                                        mle_ode_log, fixed_params, t_data, y_obs,
                                        verbose=True, parallel=False) 
    beta_real_ode = np.exp(beta_log_biv_ode)
    p_real_ode = np.exp(p_log_biv_ode)
    X_bp, Y_bp = np.meshgrid(p_real_ode, beta_real_ode)
    save_bivariate_contour(X_bp, Y_bp, Z_ode_bp,
                           mle_ode_real["p"], mle_ode_real["beta"],
                           'p', 'β', 'perfil_ode_beta_p.png', OUTPUT_DIR)
    print("Zode_bI0")
    Z_ode_bI = bivariate_profile_odeint_log('log_beta', beta_log_biv_ode, 'log_I0', I0_log_biv_ode,
                                            mle_ode_log, fixed_params, t_data, y_obs, 
                                            verbose=True, parallel= False)
    I0_real_ode = np.exp(I0_log_biv_ode)
    X_bI, Y_bI = np.meshgrid(I0_real_ode, beta_real_ode)
    save_bivariate_contour(X_bI, Y_bI, Z_ode_bI,
                           mle_ode_real["I0"], mle_ode_real["beta"],
                           'I₀', 'β', 'perfil_ode_beta_I0.png', OUTPUT_DIR)
    
    print("Zode_PTHETA")
    Z_ode_ptheta = bivariate_profile_odeint_log('log_p', p_log_biv_ode, 'log_theta', theta_log_biv_ode,
                                                mle_ode_log, fixed_params, t_data, y_obs, 
                                                verbose=False, parallel= False)
    theta_real_ode = np.exp(theta_log_biv_ode)
    X_pt, Y_pt = np.meshgrid(theta_real_ode, p_real_ode)
    save_bivariate_contour(X_pt, Y_pt, Z_ode_ptheta,
                           mle_ode_real["theta"], mle_ode_real["p"],
                           'θ', 'p', 'perfil_ode_p_theta.png', OUTPUT_DIR)

    # ---------- PERFILES UNIVARIADOS FRACCIONARIO ----------
    n_uni = 45
    alpha_grid = np.linspace(0.01, 1, n_uni)
    beta_log_grid = np.linspace(mle_frac_log["log_beta"] - 0.3, mle_frac_log["log_beta"] + 0.3, n_uni)
    p_log_grid = np.linspace(mle_frac_log["log_p"] - 0.2, mle_frac_log["log_p"] + 0.4, n_uni)
    theta_log_grid = np.linspace(mle_frac_log["log_theta"] - 1.5, mle_frac_log["log_theta"] + 1.3, n_uni)
    I0_log_grid = np.linspace(mle_frac_log["log_I0"] - 1.5, mle_frac_log["log_I0"] + 1.2, n_uni)

    print("\nCalculando perfiles univariados")

    # Perfil para alpha
    logL_alpha = profile_univariate_log('alpha', alpha_grid, mle_frac_log, fixed_params, bounds_log, t_data, y_obs, H)
    alpha_lower, alpha_upper = get_confidence_interval(alpha_grid, logL_alpha, delta=1.92)
    if alpha_lower is not None:
        print(f"IC 95% para α: [{alpha_lower:.4f}, {alpha_upper:.4f}]")
    else:
        print("No se pudo calcular IC para α")
    save_relative_profile_plot(alpha_grid, logL_alpha, mle_frac_real["alpha"], 'α', 'profile_alpha.png', OUTPUT_DIR)

    # Perfil para beta (en escala original)
    logL_beta = profile_univariate_log('log_beta', beta_log_grid, mle_frac_log, fixed_params, bounds_log, t_data, y_obs, H)
    beta_lower_log, beta_upper_log = get_confidence_interval(beta_log_grid, logL_beta, delta=1.92)
    if beta_lower_log is not None:
        beta_lower, beta_upper = np.exp(beta_lower_log), np.exp(beta_upper_log)
        print(f"IC 95% para β: [{beta_lower:.4f}, {beta_upper:.4f}]")
    else:
        print("No se pudo calcular IC para β")
    save_relative_profile_plot(beta_log_grid, logL_beta, mle_frac_log["log_beta"], 'log(β)', 'profile_log_beta.png', OUTPUT_DIR)
    
    # Perfil para p
    logL_p = profile_univariate_log('log_p', p_log_grid, mle_frac_log, fixed_params, bounds_log, t_data, y_obs, H)
    p_lower_log, p_upper_log = get_confidence_interval(p_log_grid, logL_p, delta=1.92)
    if p_lower_log is not None:
        p_lower, p_upper = np.exp(p_lower_log), np.exp(p_upper_log)
        print(f"IC 95% para p: [{p_lower:.6f}, {p_upper:.6f}]")
    else:
        print("No se pudo calcular IC para p")
    save_relative_profile_plot(p_log_grid, logL_p, mle_frac_log["log_p"], 'log(p)', 'profile_log_p.png', OUTPUT_DIR)

    # Perfil para theta
    logL_theta = profile_univariate_log('log_theta', theta_log_grid, mle_frac_log, fixed_params, bounds_log, t_data, y_obs, H)
    theta_lower_log, theta_upper_log = get_confidence_interval(theta_log_grid, logL_theta, delta=1.92)
    if theta_lower_log is not None:
        theta_lower, theta_upper = np.exp(theta_lower_log), np.exp(theta_upper_log)
        print(f"IC 95% para θ: [{theta_lower:.4f}, {theta_upper:.4f}]")
    else:
        print("No se pudo calcular IC para θ")
    save_relative_profile_plot(theta_log_grid, logL_theta, mle_frac_log["log_theta"], 'log(θ)', 'profile_log_theta.png', OUTPUT_DIR)

    # Perfil para I0
    logL_I0 = profile_univariate_log('log_I0', I0_log_grid, mle_frac_log, fixed_params, bounds_log, t_data, y_obs, H)
    I0_lower_log, I0_upper_log = get_confidence_interval(I0_log_grid, logL_I0, delta=1.92)
    if I0_lower_log is not None:
        I0_lower, I0_upper = np.exp(I0_lower_log), np.exp(I0_upper_log)
        print(f"IC 95% para I₀: [{I0_lower:.1f}, {I0_upper:.1f}]")
    else:
        print("No se pudo calcular IC para I₀")
    save_relative_profile_plot(I0_log_grid, logL_I0, mle_frac_log["log_I0"], 'log(I₀)', 'profile_log_I0.png', OUTPUT_DIR)


    print(f"\nTodos los resultados guardados en: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()