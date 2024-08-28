#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:a100:1
#SBATCH --account=share-ie-idi
#SBATCH --time=05:00:00
#SBATCH --nodes=1
#SBATCH --mem=128000
#SBATCH --ntasks-per-node=1
#SBATCH --job-name="Pretrain"
#SBATCH --output=survival.txt
#SBATCH --mail-user=tong.yu@ntnu.no
#SBATCH --mail-type=ALL


WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "we are running from this directory: $SLURM_SUBMIT_DIR"
echo " the name of the job is: $SLURM_JOB_NAME"
echo "Th job ID is $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "We are using $SLURM_CPUS_ON_NODE cores"
echo "We are using $SLURM_CPUS_ON_NODE cores per node"
echo "Total of $SLURM_NTASKS cores"


module purge
module load Anaconda3/2023.09-0
source activate gunther
conda activate gunther

# python ./data/generate_negatives.py
# python ./data/generate_data4_pretrain.py
# python pretrain.py
# srun python push_model.py
# python ./experiments/SimSurv/run_survival.py


for i in {11..11}; do
   echo "Seed $i"
   # python ./experiments/TCGA_clf/run_classifier.py encoder.base.method=cnn experiment.seed="$i";
   # python run_survival.py experiment.seed="$i" encoder.base.method=lstm;
   # python run_survival.py experiment.seed="$i" encoder.base.method=cnn;
   # python run_survival.py experiment.seed="$i" encoder.base.method=average;
   python ./experiments/ClonalAE/run_attentive_autoencoders.py encoder.base.method=transformer experiment.seed="$i";
done


