# DDP Unet

This guide provides step-by-step instructions to set up and run the Distributed Data Parallel (DDP) Unet project using Docker or Conda.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Common Troubleshooting Problems (Docker)](#common-troubleshooting-problems-docker)
   - [Issue: PyTorch Does Not Detect the GPU](#issue-pytorch-does-not-detect-the-gpu)
3. [Running the Project Using Docker](#running-the-project-using-docker)
   - [Step 1: Build the Docker Image](#step-1-build-the-docker-image)
   - [Step 2: Verify the Docker Image](#step-2-verify-the-docker-image)
   - [Step 3: Run the Docker Container](#step-3-run-the-docker-container)
   - [Step 4: Activate Your Environment and Run the Script](#step-4-activate-your-environment-and-run-the-script)
   - [Explanation of Docker Parameters](#explanation-of-docker-parameters)
4. [Running the Project Using Conda](#running-the-project-using-conda)
   - [Step 1: Install Conda and Create Environment](#step-1-install-conda-and-create-environment)
   - [Step 2: Activate the Environment](#step-2-activate-the-environment)
   - [Step 3: Run the Training Script](#step-3-run-the-training-script)
5. [Running on Multiple Nodes](#running-on-multiple-nodes)
6. [Understanding the `run.sh` Script](#understanding-the-runsh-script)
7. [Project Directory Structure](#project-directory-structure)
8. [Additional Notes](#additional-notes)

---

## Prerequisites

Before proceeding, ensure you have read the root README file of the repository. Additionally, make sure you have:

- A working NVIDIA GPU with the latest drivers installed.
- Docker and NVIDIA Container Toolkit correctly set up (if using Docker).
- Conda installed (if using Conda for environment management).
- An SSH server installed and running for distributed training:

```bash
sudo apt install openssh-server
sudo systemctl enable ssh
sudo systemctl start ssh
```

## Common Troubleshooting Problems (Docker)

### Issue: PyTorch Does Not Detect the GPU

Even if your NVIDIA container is running and `nvidia-smi` shows the expected output inside the Docker image, PyTorch might not be able to detect the NVIDIA drivers.

#### Solution:

To fix this issue, modify the NVIDIA container runtime configuration.

Run the following command outside the Docker image:

```bash
sudo nano /etc/nvidia-container-runtime/config.toml
```

Locate the following line:

```
no-cgroups = true
```

Change it to:

```
no-cgroups = false
```

Save the file and restart the Docker daemon:

```bash
sudo systemctl restart docker
```

This should resolve the issue.

## Project Directory Structure

You must be inside the project folder before running the commands. Navigate to the correct directory:

```bash
cd pytorch/unet/
```

Expected directory structure:

```
└───pytorch
    ├───hello_world
    ├───resnet
->  └───unet
        ├───data
        │   ├───images
        │   └───masks
        ├───logs
        └───saved_models
```

Ensure all necessary directories exist before running the script.

## Running the Project Using Docker

Follow these steps to build and run the Docker image for the DDP Unet project.

### Step 1: Build the Docker Image

```bash
docker build -t {name_of_image} .
```

### Step 2: Verify the Docker Image

```bash
docker images
```

You should see `{name_of_image}` listed among the available images.

### Step 3: Run the Docker Container

```bash
docker run -it --rm --gpus all -v $(pwd)/data:/workspace/data -v $(pwd)/saved_models:/workspace/saved_models -v $(pwd)/logs:/workspace/logs -p 29500:29500 {name_of_image}
```

### Step 4: Activate Your Environment and Run the Script

Once inside the Docker container, activate your Conda environment and execute the `run.sh` script:

```bash
conda activate PytorchMPI
bash run.sh
```

### Explanation of Docker Parameters

- `-it` → Runs the container in interactive mode, allowing you to interact with the terminal.
- `--rm` → Automatically removes the container after it stops to free up space.
- `--gpus all` → Allocates all available GPUs to the container for training.
- `-v $(pwd)/data:/workspace/data` → Mounts the local `data` directory to `/workspace/data` inside the container.
- `-v $(pwd)/saved_models:/workspace/saved_models` → Mounts the local `saved_models` directory to `/workspace/saved_models` in the container.
- `-v $(pwd)/logs:/workspace/logs` → Mounts the local `logs` directory to `/workspace/logs` for storing logs.
- `-p 29500:29500` → Maps port 29500 from the container to the host machine for distributed communication.

## Running the Project Using Conda

If you prefer not to use Docker, you can set up and run the project using Conda.

### Step 1: Install Conda and Create Environment

Ensure you have Miniconda installed, then create a Conda environment using the provided `env.yml` file:

```bash
conda env create -f env.yml
```

### Step 2: Activate the Environment

Activate the Conda environment:

```bash
conda activate PytorchMPI
```

### Step 3: Run the Training Script

Execute the training script using the provided `run.sh` file:

```bash
bash run.sh
```

## Understanding the `run.sh` Script

The `run.sh` script automates the setup for distributed training using PyTorch's `torchrun`. It prompts the user for several input values:

- **Number of processes per node (`nproc_per_node`)**: Defines how many processes to run per GPU.
- **Number of nodes (`nnodes`)**: Specifies the total number of machines participating in training.
- **Node rank (`node_rank`)**: Determines the order of the machine in the distributed setup.
- **Master address (`master_addr`)**: Sets the IP address of the main node (default is the current machine's IP if rank is 0).
- **Master port (`master_port`)**: Defines the communication port for distributed training (default: `29500`).
- **Training hyperparameters**:
  - Number of epochs (`num_epochs`)
  - Batch size per process (`batch_size`)
  - Learning rate (`learning_rate`)
  - Random seed (`random_seed`)
  - Model directory (`model_dir`)
  - Model filename (`model_filename`)
- **Checkpointing**: Asks whether to resume training from a previous checkpoint.

The script ensures that necessary directories (`data`, `saved_models`, and `logs`) exist before executing the training command with `torchrun`.

## Running on Multiple Nodes

After completing the setup on the master node, repeat the entire process on additional nodes to configure a distributed training environment.

### Additional Notes:

- Ensure all nodes have the required NVIDIA drivers and Docker installed (if using Docker).
- Verify that the `run.sh` script is properly set up to handle multi-node training with DDP.
- Check that all nodes are in the same network and have proper SSH access.

---

By following these steps, you should be able to set up and run the DDP Unet project using either Docker or Conda. If you encounter any issues, refer to the troubleshooting section or seek additional resources.

