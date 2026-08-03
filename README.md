# Fractional SEIR Model for Measles in Jalisco, Mexico

[![Python Version](https://shields.io)](https://python.org)
[![Stan](https://shields.io)](https://mc-stan.org)
[![License: MIT](https://shields.io)](https://opensource.org)

This repository contains the source code, statistical models, and optimization pipelines used for parameter estimation and dynamic analysis of a **Fractional-Order SEIR Model** applied to real-world measles outbreak data from Jalisco, Mexico.

The project evaluates and compares two structural frameworks using Markov Chain Monte Carlo (MCMC) and Maximum Likelihood Estimation (MLE):
1. **Autonomous Framework:** Constant transmission rate (\(\beta = \text{constant}\)).
2. **Non-Autonomous Framework:** Time-dependent transmission rate governed by an exponential decay function (\(\beta(t) = \beta_0 e^{-\delta t}\)) to capture public health intervention dynamics and control efforts.

---

## 📈 Mathematical Framework: \(R(t)\)

In the non-autonomous extension, the static basic reproduction number (\(R_0\)) is replaced by a **time-dependent effective reproduction number \(R(t)\)**. This dynamic threshold evaluates how continuous mitigation strategies drive the system toward the disease-free equilibrium:

\[R(t) = R_0 e^{-\delta t}\]

Where \(R_0\) represents the initial fractional reproductive potential at \(t = 0\), and \(\delta\) is the attenuation parameter estimated from the Jalisco epidemiological data.

---

## 📂 Repository Structure

```text
.
├── likelihood/               # Optimization pipelines via Maximum Likelihood Estimation (MLE)
│   └── Autonomo/             # Autonomous model optimization scripts
├── outputs_beta_var/         # Results for the Non-Autonomous Model (Variable Beta)
│   ├── csv/                  # Raw MCMC sample outputs from Stan
│   ├── figures/              # Fitted trajectories and posterior distributions
│   └── summaries/            # Convergence diagnostic tables and parameter metrics
├── outputs_const/            # Results for the Autonomous Model (Constant Beta)
│   ├── csv/                  # Raw MCMC sample outputs from Stan
│   ├── figures/              # Trajectory fits and diagnostic plots
│   └── summaries/            # Parameter summary metrics
├── scripts/                  # Main execution wrappers and script entry points
│   ├── run_model_stan_Jalisco.py   # Runs MCMC for the Constant Beta model
│   ├── run_model_stan_beta_var.py  # Runs MCMC for the Variable Beta model
│   ├── analyze_fit.py              # Statistical evaluation for Constant Beta
│   └── analyzbv_fit.py             # Statistical evaluation for Variable Beta
├── src/                      # Post-processing helper modules
│   ├── postprocess.py              # Trajectory and chain analysis for Constant Beta
│   └── postprocess_betav.py        # Trajectory and chain analysis for Variable Beta
├── stan/                     # Bayesian inference models written in Stan
│   ├── seir_frac_const.stan        # Fractional SEIR model with constant beta
│   └── seir_fracv.stan             # Fractional SEIR model with time-varying beta
└── README.md                 # Project documentation
```

---

## ⚙️ Requirements & Dependencies

The codebase requires **Python 3.8+** along with a functioning C++ compiler setup configured for CmdStan.

### Core Libraries
* `cmdstanpy` ($\ge 1.0.0$) — Python interface to CmdStan
* `numpy` & `pandas` — Data manipulation and structuring
* `scipy` — Numerical routines and ODE components
* `matplotlib` — Visualization and plotting routines
* `statsmodels` — Time series and statistical diagnostics

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com
cd your-repo-name
```

### 2. Environment Setup
Create a virtual environment and install the required numerical computing stacks:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install numpy pandas scipy matplotlib statsmodels cmdstanpy
```

### 3. Install CmdStan Backend
`cmdstanpy` requires the underlying `CmdStan` C++ binaries. You can automatically compile and install them inside your environment by running:
```bash
python -c "import cmdstanpy; cmdstanpy.install_cmdstan()"
```

---

## 🏃 Execution Workflow

The repository is modularized into two separate execution branches depending on the structural model under study.

### Option A: Running the Autonomous Model ($\beta$ constant)
To perform Bayesian inference on the constant transmission setup, execute the following pipelines from the root directory:
```bash
# 1. Run MCMC sampling via Stan
python scripts/run_model_stan_Jalisco.py

# 2. Extract posteriors and generate plots
python scripts/analyze_fit.py
```

### Option B: Running the Non-Autonomous Extension ($\beta(t)$ variable)
To execute the time-dependent extension that accounts for continuous epidemic control and convergence to a stationary point:
```bash
# 1. Run MCMC sampling via Stan
python scripts/run_model_stan_beta_var.py

# 2. Extract trajectories and calculate R(t) metrics
python scripts/analyzbv_fit.py
```

All summary parameters, Markov chain traceplots, and population trajectories will be generated automatically and saved into their respective `outputs_const/` or `outputs_beta_var/` directories.


## Instalación

1. Clona el repositorio:
   ```bash
   git clone https://github.com/omarramos-droid/Tesis_CIMAT
   cd MEASLESS_CODE/HMC-MCMC
