#!/bin/bash
#SBATCH --job-name=seir_frac
#SBATCH --output=../outputs/slurm_%j.out
#SBATCH --error=../outputs/slurm_%j.err
#SBATCH --time=48:00:00  
#SBATCH --partition=C1Mitad1

# Activar el entorno virtual
source ~/mi_entornoSTAN/bin/activate

# Asegurar que la carpeta outputs de los logs exista antes de escribir
mkdir -p ~/MEASLESS_CODE/HMC-MCMC/outputs

# Ir al directorio de trabajo
cd ~/MEASLESS_CODE/HMC-MCMC/scripts

# Ejecutar el script
python3 -u run_model_stan_beta_var.py
