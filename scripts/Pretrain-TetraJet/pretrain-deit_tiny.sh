SCRIPT_PATH=$(pwd)
DATA_PATH="dataset/imagenet"    # Dataset Path
WORK_PATH="../../"
PATH_TO_SAVE="."                 # NEED: Path to save checkpoints

MODEL_NAME="deit_tiny"          # Model name in ["deit_tiny", "deit_small", "deit_base"]
                                # NOTE: "deit_base" needs a different Learning-Rate & Batch-Size setting
OMP_NUM_THREADS=8

Experiment_NAME="TetraJet-MXFP4"
LOGS_NAME="logs_${Experiment_NAME}_${MODEL_NAME}"
SAVE_PATH="${PATH_TO_SAVE}/${Experiment_NAME}/${MODEL_NAME}"

TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
cd "$WORK_PATH"
mkdir -p $SCRIPT_PATH/${LOGS_NAME}

# nproc_per_node: how many gpus to run on
python -m torch.distributed.run --nproc_per_node=4 --master_port=29501 main.py \
    --model ${MODEL_NAME}_patch16_224 \
    --batch-size 256 \
    --tritonQ \
    --mxscale 1 \
    --data-path $DATA_PATH \
    --output_dir $SAVE_PATH \
    --row_blocksize 1 --column_blocksize 32 \
    --qchoice all --qlinear-all \
    --fabit 4 --fwbit 4 --babit 4 --bwbit 4 \
    --faexp 2 --fwexp 2 --baexp 2 --bwexp 2 \
    > $SCRIPT_PATH/${LOGS_NAME}/${TIMESTAMP}.log 2>&1
