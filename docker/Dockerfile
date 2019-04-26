 # docker file from https://github.com/anibali/docker-pytorch
# installs fastai with all required deps +  opencv
FROM nvidia/cuda:10.0-base-ubuntu18.04

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    acl\
 && rm -rf /var/lib/apt/lists/*

# Create a working directory
RUN mkdir /app
WORKDIR /app

# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
 && chown -R user:user /app
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
#to allow install in /opt
RUN chmod 777 /opt
#default permissions for files created in /opt
RUN setfacl -R -d -m o:rwx /opt
USER user




# All users can use /home/user as their home directory
ENV HOME=/home/user
RUN chmod 777 /home/user


WORKDIR /home/user

# Install Miniconda
RUN curl -so /opt/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
 && chmod +x /opt/miniconda.sh \
 && /opt/miniconda.sh -b -p /opt/miniconda \
 && rm /opt/miniconda.sh
ENV PATH=/opt/miniconda/bin:$PATH
ENV CONDA_AUTO_UPDATE_CONDA=false

# Create a Python  environment
RUN /opt/miniconda/bin/conda install conda-build \
 && /opt/miniconda/bin/conda create -y --name fastai \
 && /opt/miniconda/bin/conda clean -ya
ENV CONDA_DEFAULT_ENV=fastai
ENV CONDA_PREFIX=/opt/miniconda/envs/$CONDA_DEFAULT_ENV
ENV PATH=$CONDA_PREFIX/bin:$PATH

# CUDA 10.0-specific steps
RUN conda install -y -c  pytorch\
	torchvision cudatoolkit=10.0\
 && conda clean -ya


# Install Requests, a Python library for making HTTP requests
RUN conda install -y requests \
 && conda clean -ya

#install fastai
RUN conda install -c fastai fastai\
	&& conda clean -a



# Install OpenCV3 Python bindings
RUN sudo apt-get update && sudo apt-get install -y --no-install-recommends \
    libgtk2.0-0 \
    libcanberra-gtk-module \
 && sudo rm -rf /var/lib/apt/lists/*
#RUN conda install -y -c conda-forge opencv=3.4.4 \
 #&& conda clean -ya

#install jupyter lab
RUN conda install -c conda-forge jupyterlab 

#install jupyterlab-git
RUN conda install -c conda-forge jupyterlab-git \
	&& conda clean -ya
RUN ls -l /opt && ls -l /opt/miniconda

RUN jupyter lab build

# Set the default command to jupyter
ENTRYPOINT jupyter lab --ip=0.0.0.0 --no-browser
