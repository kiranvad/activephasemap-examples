#!/bin/bash

#SBATCH --account=cheme
#SBATCH --partition=ckpt-all
#SBATCH --nodes=1
##SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --output=./slurm/%j.out
#SBATCH --error=./slurm/%j.err
#SBATCH --mail-user=kiranvad@uw.edu
#SBATCH --mail-type=END
#SBATCH --export=all
#SBATCH --exclusive
#SBATCH --requeue

# usual sbatch commands
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "working directory = "$SLURM_SUBMIT_DIR
cd $SLURM_SUBMIT_DIR

eval "$(conda shell.bash hook)"
conda activate activephasemap
echo "python from the following virtualenv"
which python
echo "working directory = "
pwd
ulimit -s unlimited

module load cuda
echo "Launch Python job"
python3 -u ./01-train_np.py
echo "All Done!"
exit