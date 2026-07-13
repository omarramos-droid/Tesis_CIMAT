# run_model_stan_beta_var.py
import sys, os
import numpy as np
from cmdstanpy import CmdStanModel

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

for sub in ["csv", "figures", "summaries", "csv_beta_var"]:
    os.makedirs(os.path.join(BASE_DIR, "outputs", sub), exist_ok=True)

# Datos observados (Jalisco): primeras 20 semanas (entrenamiento)
y_obs_train = np.array([172, 190, 235, 427, 497, 566, 753, 524, 520, 475, 
                        273, 309, 153, 269, 170, 223, 180, 137, 97, 91])
# Datos de validación (5 semanas)
y_pred = np.array([101, 32, 81, 31, 26])

N_TRUE = 8_800_000 * (1 - 0.824)

stan_data = {
    "T": len(y_obs_train),
    "y": y_obs_train.tolist(),
    "T_new": len(y_pred),
    "y_new": y_pred.tolist(),
    "N0": N_TRUE,
    "h": 0.05
}

# Compilar modelo
stan_file = os.path.join(BASE_DIR, "stan", "seir_fracv.stan")
model = CmdStanModel(stan_file=stan_file)

def make_init():
    return {
       "log_beta0": np.log(5.0),
       "log_delta": np.log(0.02),
       "log_I0": np.log(3000),
       "log_p": np.log(0.01),
       "log_theta": np.log(20),
       "alpha": 0.9,
       "gamma": 0.8,
       "kappa": 0.5
    }

print("Iniciando el muestreo con CmdStanPy...")
fit = model.sample(
    data=stan_data,
    chains=2,
    parallel_chains=2,
    iter_warmup=1000,      # calentamiento
    iter_sampling=9000,   # Suficiente para una prueba
    adapt_delta=0.95,
    max_treedepth=12,
    seed=13,
    inits=make_init(),
    show_console=True
)

fit.save_csvfiles(dir=os.path.join(BASE_DIR, "outputs", "csv_beta_var"))
print("Muestreo terminado exitosamente.")