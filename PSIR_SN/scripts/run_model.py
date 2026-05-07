import sys, os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, "src"))
import numpy as np
from cmdstanpy import CmdStanModel
os.makedirs(os.path.join(BASE_DIR, "outputs/csv"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "outputs/figures"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "outputs/summaries"), exist_ok=True)

y_obs = np.array([15, 47,108,177,332,481,587,645,
                  351,160,123,132,52,22,12,6,1,0,0,0])

stan_data = {
    "T": len(y_obs),
    "y": y_obs.tolist(),
    "N": 4500,
    "h":  1 
}

model = CmdStanModel(stan_file=os.path.join(BASE_DIR, "stan/sir_frac.stan"))

fit = model.sample(
    data=stan_data,
    chains=2,
    parallel_chains=2,
    iter_warmup=5000,
    iter_sampling=5000,
    seed=13,
    show_console=True
)

fit.save_csvfiles(dir=os.path.join(BASE_DIR, "outputs/csv"))
print("Sampling terminado")
