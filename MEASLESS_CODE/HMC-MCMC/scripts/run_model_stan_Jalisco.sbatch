#!/bin/bash
#SBATCH --job-name=seir_frac
#SBATCH --output=../outputs/slurm_%j.out
#SBATCH --error=../outputs/slurm_%j.err
#SBATCH --time=20:00:00
#SBATCH --partition=C1Mitad1
# Activar el entorno virtual
source ~/mi_entornoSTAN/bin/activate

# Ir al directorio de trabajo
cd ~/PSEIR_CD/scripts

# Ejecutar el script
python3 -u  run_model_stan_Jalisco.py 2>&1

