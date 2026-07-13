# run_model_stan_Jalisco.py
import sys, os
import numpy as np
from cmdstanpy import CmdStanModel

# Obtener directorio base 
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Nombre del modelo y carpeta de outputs
MODEL_NAME = "seir_frac"  # Beta constante
OUTPUT_DIR_NAME = "outputs_normal"

# Crear carpetas de salida 
output_dir = os.path.join(BASE_DIR, OUTPUT_DIR_NAME)
for sub in ["csv", "figures", "summaries"]:
    os.makedirs(os.path.join(output_dir, sub), exist_ok=True)

# Datos observados (Jalisco)
y_obs = np.array([172, 190, 231, 425, 495, 563, 746, 520, 520,
                  474, 273, 310, 153, 269, 171, 223, 180, 138, 98, 70])

# Población susceptible efectiva 
N_TRUE = 8_800_000 * (1 - 0.824)   

# Datos para Stan
stan_data = {
    "T": len(y_obs),
    "y": y_obs.tolist(),
    "N0": N_TRUE,         
    "h": 0.5
}

# Compilar modelo 
stan_file = os.path.join(BASE_DIR, "stan", f"{MODEL_NAME}.stan")
print(f"Compilando modelo: {stan_file}")
model = CmdStanModel(stan_file=stan_file)

# Initial point 
def make_init():
    return {
        "log_beta": np.log(10),     # Prior centrado en 10
        "log_I0": np.log(500),      # Prior centrado en 500
        "log_p": np.log(0.1),       # Prior centrado en 0.1
        "log_theta": np.log(30),    # Prior centrado en 30
        "alpha": 0.5,               # Prior Beta(1,1)
        "gamma": 0.8,               # Prior centrado en 0.8
        "kappa": 0.4                # Prior centrado en 0.4
    }

# Muestreo
print(f"\nEjecutando MCMC para {MODEL_NAME}...")
print(f"Output directory: {output_dir}")

csv_output_dir = os.path.join(output_dir, "csv")

fit = model.sample(
    data=stan_data,
    chains=1,
    parallel_chains=1,
    iter_warmup=1000,
    iter_sampling=9000,
    adapt_delta=0.95,
    max_treedepth=12,
    seed=1,
    inits=make_init(),
    show_console=True,
    output_dir=csv_output_dir,
    save_warmup=True  # Guardar warmup para diagnóstico
)

# Guardar información del muestreo
print(f"\nMuestreo completado:")
print(f"  - Iteraciones warmup: {fit.num_draws_warmup}")
print(f"  - Iteraciones sampling: {fit.num_draws_sampling}")
print(f"  - Total guardadas: {fit.num_draws_sampling + fit.num_draws_warmup}")

# Guardar resumen
summary = fit.summary()
summary.to_csv(os.path.join(output_dir, "summaries", "resumen_parametros.csv"))

# Guardar diagnóstico
with open(os.path.join(output_dir, "summaries", "diagnostico.txt"), 'w') as f:
    f.write(str(fit.diagnose()))
    f.write("\n\n")
    f.write(str(summary))

print("Done.")