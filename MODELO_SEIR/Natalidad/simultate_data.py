# -*- coding: utf-8 -*-
"""
Simulación de datos para el modelo SEIR fraccionario con demografía.
Genera observaciones con ruido binomial negativo y guarda los datos.
"""

import numpy as np
import os
import json
import argparse

# Importar el solver unificado (nuevas funciones)
from solver_fraccionario import resolver_seirv, calcular_incidencia_diffS

# Argumentos de línea de comandos
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=12+1)
parser.add_argument("--h", type=float, default=0.01)
args = parser.parse_args()

np.random.seed(args.seed)

# Crear carpetas de salida
os.makedirs("data", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# Parámetros verdaderos del modelo
beta_true   = 1.6
sigma_true  = 1.5
gamma_true  = 0.9
mu_true     = 0.001      # 0.1% de mortalidad diaria 
Lambda_true = 25.0        
alpha_true  = 0.7
phi_true    = 30.0
N0_true     = 20000
I0_true     = 10
E0_true     = 0

# Configuración de la simulación
t_max       = 35        # tiempo máximo (días/semanas)
h_gen       = args.h    # paso de integración

# ------------------------------------------------------------
# Resolver el modelo
# ------------------------------------------------------------
t_data = np.arange(0, t_max + 1, 1)   # tiempos de observación (enteros)

resultado = resolver_seirv(
    beta=beta_true, sigma=sigma_true, gamma=gamma_true,
    mu=mu_true, Lambda=Lambda_true, alpha=alpha_true,
    N0=N0_true, I0=I0_true, E0=E0_true,
    T=t_data[-1], h=h_gen, t_observacion=t_data
)

# Calcular incidencia verdadera (diferencia de susceptibles)
incidencia_true = calcular_incidencia_diffS(resultado)
incidencia_true = np.clip(incidencia_true, 1e-8, None)

# ------------------------------------------------------------
# Generar observaciones con ruido binomial negativo
# ------------------------------------------------------------
mu = incidencia_true
p_nb = phi_true / (phi_true + mu)
y_obs = np.random.negative_binomial(phi_true, p_nb).astype(int)

# ------------------------------------------------------------
# Guardar resultados
# ------------------------------------------------------------
np.save(f"data/y_obs_seed_{args.seed}.npy", y_obs)
np.save(f"data/t_data.npy", t_data)
np.save(f"data/inc_true_seed_{args.seed}.npy", incidencia_true)
# Guardar en formato de texto (separado por comas)
# Preparar el diccionario con los nombres exactos del bloque 'data' de tu modelo Stan
stan_data = {
    "T": int(len(y_obs)),       # int<lower=1> T
    "y": y_obs.tolist(),        # array[T] int y
    "N0": float(N0_true),       # real N0
    "h": float(h_gen)           # real h
}

# Guardar como JSON (legible en bloc de notas y listo para Stan)
file_name = f"data/stan_data_seed_{args.seed}.json"
with open(file_name, "w") as f:
    json.dump(stan_data, f, indent=2)


params_true = {
    "beta": beta_true,
    "sigma": sigma_true,
    "gamma": gamma_true,
    "mu": mu_true,
    "Lambda": Lambda_true,
    "alpha": alpha_true,
    "phi": phi_true,
    "N0": N0_true,
    "I0": I0_true,
    "E0": E0_true,
    "h_gen": h_gen,
    "seed": args.seed
}
with open(f"data/params_true_seed_{args.seed}.json", "w") as f:
    json.dump(params_true, f, indent=2)

print(f"Simulación de datos completada con seed={args.seed}, h={h_gen}")
print("Parámetros verdaderos:")
for k, v in params_true.items():
    print(f"  {k}: {v}")
