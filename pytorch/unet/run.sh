#!/bin/bash

# Function to validate the IP address format
validate_ip() {
    local ip=$1
    if [[ $ip =~ ^([0-9]{1,3}\.){3}[0-9]{1,3}$ ]]; then
        for octet in $(echo "$ip" | tr '.' ' '); do
            if ((octet < 0 || octet > 255)); then
                return 1
            fi
        done
        return 0
    else
        return 1
    fi
}

LOG_DIR="./logs"

# Automatically determine the machine's IP address
DEFAULT_IP=$(hostname -I | awk '{print $1}') # First IP in the list
DEFAULT_IP=${DEFAULT_IP:-127.0.0.1}          # Fallback to 127.0.0.1 if no IP found

# Prompt for distributed training environment variables (with defaults)
read -p "Enter number of processes per node (nproc_per_node) [default: 1]: " NPROC_PER_NODE
NPROC_PER_NODE=${NPROC_PER_NODE:-1}

read -p "Enter number of nodes (nnodes) [default: 1]: " NNODES
NNODES=${NNODES:-1}

read -p "Enter node rank (node_rank) [default: 0]: " NODE_RANK
NODE_RANK=${NODE_RANK:-0}

# If master node (node_rank == 0), use the current machine's IP as default
if [[ $NODE_RANK -eq 0 ]]; then
    echo "Defaulting master address to this machine's IP: $DEFAULT_IP"
    MASTER_ADDR=$DEFAULT_IP
else
    read -p "Enter master address (master_addr) [default: 127.0.0.1]: " MASTER_ADDR
    MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
fi

# Validate master address IP
if ! validate_ip "$MASTER_ADDR"; then
    echo "Error: Invalid IP address format for master_addr: $MASTER_ADDR"
    exit 1
fi

read -p "Enter master port (master_port) [default: 29500]: " MASTER_PORT
MASTER_PORT=${MASTER_PORT:-29500}

# Prompt for training-specific arguments (with defaults)
read -p "Enter number of epochs (num_epochs) [default: 100]: " NUM_EPOCHS
NUM_EPOCHS=${NUM_EPOCHS:-100}

read -p "Enter batch size per process (batch_size) [default: 128]: " BATCH_SIZE
BATCH_SIZE=${BATCH_SIZE:-128}

read -p "Enter learning rate (learning_rate) [default: 0.001]: " LEARNING_RATE
LEARNING_RATE=${LEARNING_RATE:-0.001}

read -p "Enter random seed (random_seed) [default: 42]: " RANDOM_SEED
RANDOM_SEED=${RANDOM_SEED:-42}

read -p "Enter model directory (model_dir) [default: saved_models]: " MODEL_DIR
MODEL_DIR=${MODEL_DIR:-saved_models}

read -p "Enter model filename (model_filename) [default: model.pth]: " MODEL_FILENAME
MODEL_FILENAME=${MODEL_FILENAME:-model.pth}

read -p "Resume from a checkpoint? (yes or no) [default: no]: " RESUME_PROMPT
RESUME_PROMPT=${RESUME_PROMPT:-no}

# Determine if resume flag needs to be set
if [[ $RESUME_PROMPT == "yes" ]]; then
    RESUME="--resume"
else
    RESUME=""
fi


# Ensure necessary directories exist
if [[ ! -d "data" ]]; then
    echo "The 'data' directory does not exist. Please create it before running this script."
    exit 1
fi

if [[ ! -d "${MODEL_DIR}" ]]; then
    echo "The model directory '${MODEL_DIR}' does not exist. Please create it before running this script."
    exit 1
fi

if [[ ! -d "${LOG_DIR}" ]]; then
    echo "The logs directory '${LOG_DIR}' does not exist. Please create it before running this script."
    exit 1
fi


# Execute the distributed training with torchrun
torchrun --nproc_per_node=$NPROC_PER_NODE \
         --nnodes=$NNODES \
         --node_rank=$NODE_RANK \
         --master_addr=$MASTER_ADDR \
         --master_port=$MASTER_PORT \
         train.py \
         --num_epochs $NUM_EPOCHS \
         --batch_size $BATCH_SIZE \
         --learning_rate $LEARNING_RATE \
         --random_seed $RANDOM_SEED \
         --model_dir $MODEL_DIR \
         --model_filename $MODEL_FILENAME \
         $RESUME