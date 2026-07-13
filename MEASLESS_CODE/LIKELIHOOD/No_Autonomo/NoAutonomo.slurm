#!/bin/bash

#SBATCH --job-name=seir_fit
#SBATCH --output=seir_%j.log
#SBATCH --error=seir_%j.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --partition=C1Mitad1
#SBATCH --account=test


cd ~/MEASLESS_CODE/LIKELIHOOD/No_Autonomo
source ~/mi_entornoODEINT/bin/activate
python3  -u  run_seir_jalisco_mm.py
