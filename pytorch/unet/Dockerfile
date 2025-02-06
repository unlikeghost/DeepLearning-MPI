# Use the Miniconda base image for Conda environment management
FROM continuumio/miniconda3:latest

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Update and install essential dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    curl \
    vim \
    wget \
    openssh-server \
    && rm -rf /var/lib/apt/lists/*

# Configure SSH for multi-node communication (for DDP)
RUN mkdir /var/run/sshd && \
    echo 'root:root' | chpasswd && \
    sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    sed -i 's/#PasswordAuthentication yes/PasswordAuthentication yes/' /etc/ssh/sshd_config

# Expose the default SSH and DDP communication ports
EXPOSE 22 29500

# Copy the Conda `env.yml` file into the container
COPY env.yml /opt/env.yml

# Create the Conda env and clean up
RUN conda env create -f /opt/env.yml && \
    conda clean -afy

# Activate the Conda environment by default
ENV PATH /opt/conda/envs/PytorchMPI/bin:$PATH
ENV CONDA_DEFAULT_ENV PytorchMPI

# Create necessary directories for saving models, logs, and data
RUN mkdir -p /workspace/saved_models /workspace/logs /workspace/data

# Set the working directory
WORKDIR /workspace

# Copy all project files into the container
COPY . /workspace/


# Make the `run.sh` script executable
RUN chmod +x /workspace/run.sh

# Start SSH service and run a default bash shell
#CMD service ssh start && bash
CMD /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate PytorchMPI && service ssh start && bash"
