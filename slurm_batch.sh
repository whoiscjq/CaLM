#!/bin/bash
#SBATCH -p CAI
#SBATCH -N 1                       
#SBATCH --ntasks-per-node=1        
#SBATCH --gres=gpu:2               
#SBATCH -t 33-00:00:00             
#SBATCH --job-name=calm_run       
#SBATCH -o /mnt/petrelfs/chenjunqi/R1_like/CaLM/calm_log/output.%j.log    

python calm/run.py --models deepscaler_15b -p basic -t NDE-P_NDE-basic_EN -mcfg ./model_configs -o ./output_deepseek -l
