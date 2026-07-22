# SEIR fraccionario para sarampión en Jalisco

Repositorio con los códigos utilizados para la estimación de parámetros y análisis del modelo SEIR fraccionario aplicado a datos de sarampión en Jalisco, México. El proyecto incluye implementaciones en **Stan** (MCMC) y **Python** (optimización y post‑procesamiento) de los modelos autónomo ($\beta$ constante) y no autónomo ($\beta(t) = \beta_0 e^{-\delta t}$).

## Estructura del repositorio
├── stan/ # Modelos en Stan

│ ├── seir_frac_const.stan # Beta constante

│ └── seir_fracv.stan # Beta variable

│
├── scripts/ # Scripts de ejecución y análisis

│ ├── run_model_stan_Jalisco.py # MCMC beta constante

│ ├── run_model_stan_beta_var.py# MCMC beta variable

│ ├── analyze_fit.py # Análisis beta constante

│ └── analyzbv_fit.py # Análisis beta variable
│
├── src/ # Módulos de post‑procesamiento
│ ├── postprocess.py # Funciones beta constante
│ └── postprocess_betav.py # Funciones beta variable
│
├── outputs_const/ # Resultados beta constante
│ ├── csv/ # Archivos CSV de Stan
│ ├── figures/ # Gráficos generados
│ └── summaries/ # Resúmenes numéricos
│
├── outputs_beta_var/ # Resultados beta variable
│ ├── csv/
│ ├── figures/
│ └── summaries/
│
├── likelihood/ # Códigos de optimización MLE
│ └── Autonomo/ # (ruta según estructura original)
│
└── README.md


## Requisitos

- **Python 3.8+**
- **cmdstanpy** (>= 1.0.0)
- **NumPy**, **Pandas**, **Matplotlib**, **SciPy**, **Statsmodels**
- **CmdStan** instalado y configurado (se recomienda la instalación automática con `cmdstanpy`)

## Instalación

1. Clona el repositorio:
   ```bash
   git clone https://github.com/omarramos-droid/Tesis_CIMAT
   cd MEASLESS_CODE/HMC-MCMC
