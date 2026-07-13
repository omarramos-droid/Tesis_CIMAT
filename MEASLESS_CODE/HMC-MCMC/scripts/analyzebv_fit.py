# analyze_fit_beta_var.py
import sys, os
sys.path.append(os.path.abspath("../src"))

import numpy as np
from cmdstanpy import from_csv
from postprocess_betav import *

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Datos de entrenamiento (20 semanas) y validación
y_obs_train = np.array([172, 190, 235, 427, 497, 566, 753, 524, 520, 475, 
                        273, 309, 153, 269, 170, 223, 180, 137, 97, 91])
y_pred = np.array([101, 32, 81, 31, 26])   # semanas 21-24

# Cargar ajuste (debe apuntar a los CSVs generados por run_model_stan_beta_var.py)
fit = from_csv("../outputs/csv_beta_var")

# Extraer posteriores (8 parámetros principales)
posterior = extraer_posterior_combinado(fit, nombres=NOMBRES_PRIOR_POST)
cadenas = extraer_posterior_por_cadena(fit, nombres=NOMBRES_PRIOR_POST)

# Gráficos de diagnóstico
guardar_trazas(cadenas, "../outputs/figures", nombres=NOMBRES_PRIOR_POST)
guardar_autocorrelacion(posterior, "../outputs/figures", nombres=NOMBRES_PRIOR_POST)

# Ajuste a los datos de entrenamiento
graficar_ajuste_desde_stan(fit, y_obs_train, 
                           outdir=os.path.join(BASE_DIR, "outputs", "figures"))

# Prior vs posterior
guardar_prior_vs_posterior_beta_var(fit, 
                                     outdir=os.path.join(BASE_DIR, "outputs", "figures"))

# Perfil temporal (solo entrenamiento)
graficar_perfil_epidemia(fit, y_obs_train, 
                         outdir=os.path.join(BASE_DIR, "outputs", "figures"))

# Evolución de beta (para todo el período, incluye validación)
graficar_evolucion_beta(fit, outdir=os.path.join(BASE_DIR, "outputs", "figures"))

# ANÁLISIS PREDICTIVO 
metrics = analisis_predictivo(fit, y_obs_train, y_pred,
                              outdir=os.path.join(BASE_DIR, "outputs", "figures"))
print("\nMétricas predictivas:")
print(metrics)

# Resumen numérico completo
posterior_full = extraer_posterior_combinado(fit, nombres=NOMBRES_COMPLETOS)
df = resumen_numerico(posterior_full, nombres=NOMBRES_COMPLETOS)
df.to_csv(os.path.join(BASE_DIR, "outputs", "summaries", "resumen_beta_var.csv"))
print("\nResumen de parámetros:")
print(df)