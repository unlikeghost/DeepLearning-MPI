#!/bin/bash

# Solicitar entradas al usuario
read -p "Enter number of processes per node (nproc_per_node): " NPROC_PER_NODE
read -p "Enter number of nodes (nnodes): " NNODES
read -p "Enter node rank (node_rank): " NODE_RANK
read -p "Enter master address (master_addr): " MASTER_ADDR
read -p "Enter master port (master_port): " MASTER_PORT
read -p "Enter backend (e.g., nccl or gloo): " BACKEND
read -p "Enter script to run (e.g., hello_world.py): " SCRIPT


# Ejecución del comando torchrun
torchrun --nproc_per_node=$NPROC_PER_NODE \
         --nnodes=$NNODES \
         --node_rank=$NODE_RANK \
         --master_addr=$MASTER_ADDR \
         --master_port=$MASTER_PORT \
         $SCRIPT --backend $BACKEND
