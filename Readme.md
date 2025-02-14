# DDP Using Docker

This document provides a step-by-step guide to setting up your system and running Distributed Data Parallel (DDP) on Ubuntu.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installing NVIDIA Drivers](#installing-nvidia-drivers)
3. [Installing Docker](#installing-docker)
   - [Removing Conflicting Packages](#removing-conflicting-packages)
   - [Setting Up Docker Repository](#setting-up-docker-repository)
   - [Post-Installation Configuration](#post-installation-configuration)
4. [Configuring NVIDIA Containers](#configuring-nvidia-containers)
   - [Configuring the Production Repository](#configuring-the-production-repository)
   - [Updating the Package List](#updating-the-package-list)
   - [Installing the NVIDIA Container Toolkit](#installing-the-nvidia-container-toolkit)
   - [Configuring Docker](#configuring-docker)
   - [Rootless Mode (Recommended)](#rootless-mode-recommended)
   - [Testing NVIDIA Container and Docker](#testing-nvidia-container-and-docker)
5. [Configuring SSH for DDP](#configuring-ssh-for-ddp)
   - [Installing OpenSSH](#installing-openssh)
   - [Checking SSH Server Status](#checking-ssh-server-status)
   - [Allowing SSH Connections on Port 22](#allowing-ssh-connections-on-port-22)
6. [Using Conda or Other Python Environment Managers (Optional)](#using-conda-or-other-python-environment-managers-optional)
   - [Installing Miniconda](#installing-miniconda)
   - [Creating a Virtual Environment](#creating-a-virtual-environment)
   - [Installing Dependencies](#installing-dependencies)

---

## Prerequisites

Before starting, update your system:

```bash
sudo apt-get update
sudo apt-get upgrade
sudo reboot now
```

After rebooting, verify that NVIDIA drivers are installed:

```bash
nvidia-smi
```

If installed correctly, you should see output similar to:

```
Thu Feb 13 11:22:00 2025
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 528.24       Driver Version: 528.24       CUDA Version: 12.0     |
|-------------------------------+----------------------+----------------------+
| GPU Name            TCC/WDDM | Bus-Id        Disp.A | Volatile Uncorr. ECC  |
| Fan Temp Perf Pwr:Usage/Cap|         Memory-Usage | GPU-Util Compute M.   |
|                               |                      |               MIG M. |
|===============================+======================+======================|
| 0 NVIDIA RTX A100... WDDM  | 00000000:01:00.0 Off |                  N/A  |
| N/A   53C    P0     8W /  36W |      0MiB /  4096MiB |      0%      Default |
|                               |                      |                   N/A|
+-----------------------------------------------------------------------------+
```

## Installing NVIDIA Drivers

If `nvidia-smi` does not return the expected output, install the NVIDIA drivers. Refer to:

- [Installing NVIDIA Drivers on Ubuntu](https://ubuntu.com/server/docs/nvidia-drivers-installation)
- [CyberCiti Guide on Installing NVIDIA Drivers](https://www.cyberciti.biz/faq/ubuntu-linux-install-nvidia-driver-latest-proprietary-driver/)

To check available drivers:

```bash
sudo ubuntu-drivers list
```

Install the appropriate driver:

```bash
sudo ubuntu-drivers install
sudo reboot now
```

To install a specific version:

```bash
sudo apt-get install nvidia-driver-550
sudo reboot now
```

## Installing Docker

Docker requires a 64-bit Ubuntu version:

- Ubuntu 24.10 (Oracular)
- Ubuntu 24.04 (Noble, LTS)
- Ubuntu 22.04 (Jammy, LTS)
- Ubuntu 20.04 (Focal, LTS)

### Removing Conflicting Packages

```bash
for pkg in docker.io docker-doc docker-compose docker-compose-v2 podman-docker containerd runc; do sudo apt-get remove $pkg; done
```

### Setting Up Docker Repository

```bash
sudo apt-get update
sudo apt-get install ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
```

### Post-Installation Configuration

To run Docker without `sudo`:

```bash
sudo groupadd docker
sudo usermod -aG docker $USER
newgrp docker
```

## Configuring NVIDIA Containers

### Configuring the Production Repository

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
```

### Updating the Package List

```bash
sudo apt-get update
```

### Installing the NVIDIA Container Toolkit

```bash
sudo apt-get install -y nvidia-container-toolkit
```

### Configuring Docker

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### Rootless Mode (Recommended)

```bash
nvidia-ctk runtime configure --runtime=docker --config=$HOME/.config/docker/daemon.json
systemctl --user restart docker
```

### Testing NVIDIA Container and Docker

```bash
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
```

## Configuring SSH for DDP

### Installing OpenSSH

```bash
sudo apt install openssh-server
```

### Checking SSH Server Status

```bash
sudo systemctl status ssh
```

### Allowing SSH Connections on Port 22

```bash
sudo ufw allow ssh
```

## Using Conda or Other Python Environment Managers (Optional)

### Installing Miniconda

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
```
