# MSML605

## Table of Contents
- [Project Overview](#project-overview)
- [Setup Environment](#setup-environment)  
  - [Clone the repository](#clone-the-repository)  
  - [Setup Environment on Local CPU](#setup-environment-on-local-cpu)  
  - [Setup Environment on Local GPU](#setup-environment-on-local-gpu)  
- [Running the Code](#running-the-code)  
  - [CPU Mode](#cpu-mode)  
  - [GPU Mode](#gpu-mode)  
- [Using Docker](#using-docker)  
  - [CPU Image](#cpu-image)  
  - [GPU Image](#gpu-image)  
- [Running on HPC Cluster](#running-on-hpc-cluster)


## Project Overview
Implement a Convolutional Neural Network (CNN) in two hardware-optimized versions: CPU-optimized version and GPU-optimized version. Compare and explain how and why the code differs, and provide benchmarking results showing the hardware-specific optimization benefits.


## Setup Environment

### Clone the repository

   ```bash
   git clone https://github.com/AbhinavKumar333/MSML605.git
   cd MSML605/
   ```

### Setup Environment on Local CPU

1. **Create and activate a virtual environment**

   ```bash
   python3 -m venv cpu_venv
   source cpu_venv/bin/activate
   ```
2. **Install dependencies**

   ```bash
   pip install --upgrade pip
   pip install -r cpu_requirements.txt
   ```
3. **Deactivate the environment**
   ```bash
   deactivate
   ```

### Setup Environment on Local GPU

1. **Ensure CUDA 12.1** is installed on your system.

2. **Create and activate a virtual environment**

   ```bash
   python3 -m venv gpu_venv
   source gpu_venv/bin/activate
   ```

3. **Install GPU dependencies**

   ```bash
   pip install --upgrade pip
   pip install -r gpu_requirements.txt
   ```

4. **Deactivate the environment**
   ```bash
   deactivate
   ```

## Running the Code

Make sure to activate the corresponding environment when running CPU or GPU runs.

### CPU Mode

**Activate the virtual env**
```bash
source cpu_venv/bin/activate
```

**1. For full comparison of all the models (Resnet18 vs MobileNetv2)**
```bash
python driver.py --hardware CPU --mode full-compare
```

**2. For runnning only sweep (runs both Resnet and MobileNet)**
```bash
python driver.py --hardware CPU --mode sweep
```

**3. For running only tuning (runs for both Resnet and MobileNet)**
```bash
python driver.py --hardware CPU --mode tune
```

**4. For running single model (runs for either Resnet or MobileNet)**
```bash
python driver.py --hardware CPU --mode single --model resnet18 --subset True --dataset_size 5000 --batch_size 64 --learning_rate 0.001
```

### GPU Mode

**Activate the virtual env**
```bash
source gpu_venv/bin/activate
```

**1. For full comparison of all the models (Resnet18 vs MobileNetv2)**
```bash
python driver.py --hardware GPU --mode full-compare
```

**2. For runnning only sweep (runs both Resnet and MobileNet)**
```bash
python driver.py --hardware GPU --mode sweep
```

**3. For running only tuning (runs for both Resnet and MobileNet)**
```bash
python driver.py --hardware GPU --mode tune
```

**4. For running single model (runs for either Resnet or MobileNet)**
```bash
python driver.py --hardware GPU --mode single --model resnet18 --subset True --dataset_size 5000 --batch_size 64 --learning_rate 0.001
```

## Using Docker
The docker image runs the full comparison by default.

### CPU Image

```bash
# Pull and run CPU container
docker pull ak395/myapp:cpu-latest
docker run --rm -v /local/data:/data ak395/myapp:cpu-latest
```

### GPU Image

```bash
# Pull and run GPU container
docker pull ak395/myapp:gpu-latest
docker run --rm --gpus all -v /local/data:/data ak395/myapp:gpu-latest
```

## Running on HPC Cluster

1. **Clone repo and create venv** in your scratch area:

   ```bash
   git clone https://github.com/AbhinavKumar333/MSML605.git

   cd MSML605/

   python3 -m venv cpu_venv
   source cpu_env/bin/activate
   pip install --upgrade pip
   pip install -r cpu_requirements.txt
   deactivate

   python3 -m venv gpu_venv
   source gpu_env/bin/activate
   pip install --upgrade pip
   pip install -r gpu_requirements.txt
   deactivate
   ```
2. **Submit batch job scripts**

   ```bash
   sbatch run_cpu.sbatch
   sbatch run_gpu.sbatch
   ```
