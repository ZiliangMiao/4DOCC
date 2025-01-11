#!/bin/bash
#
#SBATCH --get-user-env
#SBATCH --job-name=train100                   ## Job name
#SBATCH --output=%x_%j.out                 ## File that STOUT will be written (%j:Job ID, %t:Task ID)
#SBATCH --error=%x_%j.err                  ## File that STDERR will be written
# Uncomment these two lines if you want e-mail to be sent
#SBATCH --mail-type=ALL                    ## Email notification type: BEGIN,END,FAIL,ALL
#SBATCH --mail-user=miaozl@connect.hku.hk                ## Email that notifications will be sent to
# Partition and Nodes
#SBATCH --partition=l40s
#SBATCH --nodes=1                               ## Number of compute node(s)
#SBATCH --ntasks-per-gpu=1                     ## Number of process(es) per compute node
#SBATCH --time=7-00:00:00                         ## Runtime in D-HH:MM/HH:MM:SS/MM:SS
#SBATCH --cpus-per-gpu=48
#SBATCH --mem-per-cpu=3GB                              ## Total memory over all of the cores(in MB)
#SBATCH --gres=gpu:1

echo "Submission Directory : " $SLURM_SUBMIT_DIR
echo "Submission Host      : " $SLURM_SUBMIT_HOST
echo "Job User             : " $SLURM_JOB_USER
echo "Job ID               : " $SLURM_JOB_ID
echo "Job Name             : " $SLURM_JOB_NAME
echo "Queue                : " $SLURM_JOB_PARTITION
echo "Node(s) allocated    : " $SLURM_JOB_NODELIST
echo "Number of Node(s)    : " $SLURM_NNODES
echo "Number of CPU Task(s): " $SLURM_NTASKS
echo "Number of Process(s) : " $SLURM_NPROCS
echo "Task(s) per Node     : " $SLURM_TASKS_PER_NODE
echo "CPU(s) per Task      : " $SLURM_CPUS_PER_TASK
echo "Task ID              : " $SLURM_ARRAY_TASK_ID

echo ===========================================================
echo "Job Start  Time is `date "+%Y/%m/%d -- %H:%M:%S"`"

cd /scr/u/miaozl/Projects/4DOCC/;
OUTFILE=${SLURM_JOB_NAME} ${SLURM_JOB_ID}

source ~/.bashrc;
conda activate moco;

python mos4d_baseline_script.py --mode "train" --hpc True

mv ${OUTFILE} ${SLURM_SUBMIT_DIR}

echo "Job Finish Time is `date "+%Y/%m/%d -- %H:%M:%S"`"

exit 0
