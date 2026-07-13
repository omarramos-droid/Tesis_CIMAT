#!/bin/bash

#SBATCH --job-name=seir_fitN
#SBATCH --output=seir_%j.log
#SBATCH --error=seir_%j.err
#SBATCH --time=100:00:00
#SBATCH --ntasks=1
#SBATCH --partition=C1Mitad1
#SBATCH --account=test

cd ~/MEASLESS_CODE/LIKELIHOOD/Autonomo
source ~/mi_entornoODEINT/bin/activate
python3  -u run_seir_jalisco_Sobreajuste.py

