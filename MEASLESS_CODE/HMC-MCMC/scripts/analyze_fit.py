# -*- coding: utf-8 -*-
#analyze_fit.py
"""
Análisis de cadenas MCMC para el modelo SEIR fraccionario
con subreporte (Jaslico).
"""
import sys, os
sys.path.append(os.path.abspath("../src"))

import numpy as np
from cmdstanpy import from_csv
from postprocess import *   # funciones 

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_CSV = os.path.join(BASE_DIR, "outputs_normal", "csv")
OUTPUT_FIG = os.path.join(BASE_DIR, "outputs_normal", "figures")
OUTPUT_SUM = os.path.join(BASE_DIR, "outputs_normal", "summaries")

# Datos observados (Jaslico)
y_obs = np.array([172, 190, 231, 425, 495,
                      563, 746, 520, 520, 474, 273, 310, 153, 269, 171, 223, 180, 138, 98, 70])

t_obs = np.arange(0, len(y_obs) + 1)   # semanas
N_TRUE = 8_800_000 * (1 - 0.824)  

# Cargar ajuste
fit = from_csv("../outputs_normal/csv")        # carpeta donde guardaste los CSV de Stan

# Extraer posteriores
posterior = extraer_posterior_combinado(fit)
cadenas = extraer_posterior_por_cadena(fit)

# Gráficos de diagnóstico
guardar_trazas(cadenas, "../outputs_normal/figures")
guardar_autocorrelacion(posterior, "../outputs_normal/figures")

# Ajuste a los datos
graficar_ajuste_desde_stan(
    fit=fit,
    y_obs=y_obs,
    outdir=os.path.join(BASE_DIR, "outputs_normal", "figures"),
    nsamples=40
)

# Prior vs posterior
guardar_prior_vs_posterior(
    fit=fit, 
    outdir=os.path.join(BASE_DIR, "outputs_normal", "figures")
)

#  Perfil temporal de la epidemia 
graficar_perfil_epidemia(
    fit=fit,
    y_obs=y_obs,
    outdir=os.path.join(BASE_DIR, "outputs_normal", "figures")
)
# Resumen numérico
df = resumen_numerico(posterior)
df.to_csv("../outputs_normal/summaries/resumen.csv")
print(df)