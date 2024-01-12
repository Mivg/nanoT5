#!/bin/bash

# Set the environment variables
export JOB_NAME=${JOB_NAME:-nanoT5}
export OUT_DIR=${OUT_DIR:-/home/joberant/data_nobck/maorivgi/nanoT5}
export PARTITION=${PARTITION:-gpu-a100-killable}
export GPUS=${GPUS:-1}
export CONSTRAINTS=${CONSTRAINTS:-"a100"}
export DATADIR=${OUT_DIR:-/home/joberant/data_nobck/maorivgi/data/nanoT5}

# Check if OUT_DIR exists, create it if it doesn't
mkdir -p "${OUT_DIR}"/logs
mkdir -p "${OUT_DIR}"/hydra
mkdir -p "${OUT_DIR}"/cache
mkdir -p "${DATADIR}"

# Create the job script with SBATCH directives
cat <<EOF > temp_job_script.sh
#!/bin/bash

# SLURM configurations
#SBATCH --job-name=${JOB_NAME}
#SBATCH --nodes=1
#SBATCH --output=${OUT_DIR}/logs/%j.out
#SBATCH --error=${OUT_DIR}/logs/%j.out
#SBATCH --time=0-23:59:00
#SBATCH --partition=${PARTITION}
#SBATCH --cpus-per-task=20
#SBATCH --gpus=${GPUS}
#SBATCH --ntasks=1
#SBATCH --mem=128G
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=maorivgi
#SBATCH --signal=B:USR1@300
#SBATCH --constraint=${CONSTRAINTS}

source ~/.bashrc
conda activate nanoT5

# Environment variables
export XDG_CACHE_HOME=${OUT_DIR}/cache
export ALLENNLP_CACHE_ROOT=${OUT_DIR}/cache
export TORCH_HOME=${OUT_DIR}/cache
export HF_DATASETS_CACHE=${OUT_DIR}/cache
export HF_HUB_CACHE=${OUT_DIR}/cache
export HF_ASSETS_CACHE=${OUT_DIR}/cache # https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables
export HYDRA_FULL_ERROR=1


# Execution command
python -m nanoT5.main \
hydra.run.dir=${OUT_DIR}/hydra/\\\${now:%Y-%m-%d}/\\\${now:%H-%M-%S}-\\\${logging.neptune_creds.tags} \
checkpoint.every_steps=10000 \
eval.every_steps=5000 \
eval.steps=500 \
model.compile=true \
optim.batch_size=256 \
optim.grad_acc=4 \
optim.warmup_steps=100000 \
optim.total_steps=80000 \
logging.every_steps=10\

EOF

#data.data_dir=${DATADIR}/splits/default \  # those are only for FT
#data.task_dir=${DATADIR}/tasks       # those are only for FT

# Make the job script executable
chmod +x temp_job_script.sh

# Submit the job
sbatch temp_job_script.sh

# run it with JOB_NAME=MY_JOB OUT_DIR=MY_DIR PARTITION=my_partition GPUS=2 CONSTRAINTS=my_constraint ./submit_job.sh

# Note that by using klass: local_t5 we use this repo version of T5 and not the one one huggingface
# to resume a job, we will need to set checkpoint_path

# Resuming a training is with +accelerator.checkpoint_path=<checkpoint dir without trailing slash>
