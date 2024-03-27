#!/bin/bash

#SBATCH --job-name=ANP
#SBATCH --account=cheme
#SBATCH --partition=ckpt
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --time=01:30:00
#SBATCH --mem=64G
#SBATCH --output=./slurm/anp.out
#SBATCH --error=./slurm/anp.err
#SBATCH --mail-user=kiranvad@uw.edu
#SBATCH --mail-type=END
#SBATCH --export=all
#SBATCH --exclusive

# usual sbatch commands
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "working directory = "$SLURM_SUBMIT_DIR
cd $SLURM_SUBMIT_DIR

eval "$(conda shell.bash hook)"
conda activate activephasemap
echo "python from the following environment"
which python
echo "working directory = "
pwd
ulimit -s unlimited

echo "Launch Python job"
python -u run.py  > ./slurm/anp.log
echo "All Done!"
exit