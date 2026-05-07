import sys, os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, "src"))
import numpy as np
from cmdstanpy import CmdStanModel
os.makedirs(os.path.join(BASE_DIR, "outputs/csv"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "outputs/figures"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "outputs/summaries"), exist_ok=True)

y_obs = np.array([
    12,24,38,55,68,126,219,406,451,447, 782,1411,1180,
    918,1848,1256,1127,900,716,554,421,364,174,126,89,77,52,36,10,0
  ])

stan_data = {
    "T": len(y_obs),
    "y": y_obs.tolist(),
    "E0": 0,
    "N0": 20000,
    "h":  1 
}

model = CmdStanModel(stan_file=os.path.join(BASE_DIR, "stan/seir_frac.stan"))

fit = model.sample(
    data=stan_data,
    chains=2,
    parallel_chains=2,
    iter_warmup=2000,
    iter_sampling=2000,
    seed=13,
    show_console=True
)

fit.save_csvfiles(dir=os.path.join(BASE_DIR, "outputs/csv"))
print("Sampling terminado")
