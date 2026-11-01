#!/bin/bash
# Template Slurm job for AIM on Pat. Copy into the run's folder, edit the marked lines, then:
#   sbatch -J J##_name aim_job.sbatch
# Partition (Part3 coding_notes §15): < 2 h -> CPU,CLUSTER ; longer -> CPU (CLUSTER jobs get preempted).
#SBATCH --partition=CPU,CLUSTER
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16          # run_parallel uses every CPU it is given
#SBATCH --mem=32G
#SBATCH --output=%x_%j.log          # %x = job name, %j = job id; lands in the submit folder

set -euo pipefail
source ~/aim-env/bin/activate
cd ~/aim
echo "job $SLURM_JOB_ID on $(hostname), commit $(git rev-parse --short HEAD), $(date)"
export SCRATCH="$SLURM_SUBMIT_DIR"   # run_parallel writes its trees/pickles here
export PYTHONUNBUFFERED=1

# ---- EDIT: the actual command ------------------------------------------------
python -u -m benchmark.step_recall --limit 300
# -------------------------------------------------------------------------------

echo "done $(date)"
