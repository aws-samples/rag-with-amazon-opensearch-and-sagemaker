FROM ubuntu:latest

# Install necessary packages
RUN apt-get update && \
    apt-get install -y wget curl git \
        build-essential \
        automake libcurl4-openssl-dev \
        libxml2-dev pkg-config libssl-dev libfuse-dev parallel && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Miniforge
RUN wget --quiet https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /miniforge.sh && \
    bash /miniforge.sh -b -p /miniconda && \
    rm /miniforge.sh

WORKDIR /workspace
COPY . .

ENV PATH="/workspace:/miniconda/bin:${PATH}"

# Create Conda environment and clean up in one layer to reduce image size
RUN conda update conda -y && \
    conda install mamba -n base -c conda-forge && \
    conda config --set ssl_verify no && \
    mamba env create -f environment.yml && \
    mamba clean -afy

ENTRYPOINT ["mamba", "run", "--no-capture-output", "-n", "streamlit-env", "streamlit", "run", "app.py", "--server.address=0.0.0.0"]
