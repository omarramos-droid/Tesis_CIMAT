# -*- coding: utf-8 -*-
"""
Created on Tue May  5 16:01:27 2026

@author: dell
"""
import scipy.stats as stats  
import sys, os
sys.path.append(os.path.abspath("../src"))

from cmdstanpy import from_csv
from postprocess import *

y_obs = np.array([15, 47,108,177,332,481,587,645,
    351, 160,123, 132,52, 22, 12, 6, 1, 0, 0, 0])



t_obs = np.arange(0, len(y_obs)+1)
N_total = 20000

# Cargar resultados
fit = from_csv("../outputs/csv")

posterior = extraer_posterior_combinado(fit)
cadenas = extraer_posterior_por_cadena(fit)

# Guardar figuras
guardar_trazas(cadenas, "../outputs/figures")
guardar_autocorrelacion(posterior, "../outputs/figures")
graficar_ajuste_incidencia(fit,y_obs,t_obs, N_total,0,
    outdir="../outputs/figures",
    h_solver=0.01,
    nsamples=40)
#graficar_dispersiones(posterior, outdir="../outputs/figures")
guardar_prior_vs_posterior(fit, outdir="../outputs/figures")


df = resumen_numerico(posterior)
df.to_csv("../outputs/summaries/resumen.csv")

print(df)
