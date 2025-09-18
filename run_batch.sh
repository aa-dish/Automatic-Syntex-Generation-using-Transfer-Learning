#!/bin/bash
#SBATCH --mail-type=END,FAIL         # want a mail notification at end of job
#SBATCH -J  jupyter_lab     # name of the job
#SBATCH -o ./slurm_output/%x.%j.out     # Output: %j expands to jobid
#SBATCH -e ./slurm_output/%x.%j.err     # Error: %j expands to jobid
#SBATCH --time=8-00:00:00
#SBATCH --ntasks-per-node=5
#SBATCH -N 1
#SBATCH --mem-per-cpu=12G
#SBATCH --gres=gpu:V100:1
#SBATCH --partition=informatik-mind

module load anaconda3/latest
. $ANACONDA_HOME/etc/profile.d/conda.sh

conda activate env10

cd /scratch/guptad/hiwi/ || exit
jupyter lab --no-browser --ip "*" --notebook-dir /scratch/guptad/hiwi/avatars/src/notebooks/ --port 17039

conda deactivate 
