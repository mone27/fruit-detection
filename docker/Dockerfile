# docker file from https://github.com/anibali/docker-pytorch
# installs fastai with all required deps +  opencv + jupyterlab (with git extensions) 
FROM nvidia/cuda:10.0-base-ubuntu18.04

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    build-essential\
 && rm -rf /var/lib/apt/lists/*


# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' --shell /bin/bash user
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
#to allow install in /opt
RUN chmod 777 /opt

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

# Create a Python  environment
RUN /opt/miniconda/bin/conda install conda-build \
 && /opt/miniconda/bin/conda create -y --name fastai \
 && /opt/miniconda/bin/conda clean -ya
ENV CONDA_DEFAULT_ENV=fastai
ENV CONDA_PREFIX=/opt/miniconda/envs/$CONDA_DEFAULT_ENV
ENV PATH=$CONDA_PREFIX/bin:$PATH

#install pytorch with  CUDA 10.1
RUN conda install -y -c  pytorch\
	torchvision cudatoolkit=10.1\
 && conda clean -ya


#install from conda-forge: opencv, jupyterlab, nodejs an matplotlib widgets
RUN conda install -y -c conda-forge opencv jupyterlab nodejs ipympl \
&& conda clean -ya

RUN jupyter labextension install @jupyter-widgets/jupyterlab-manager\
    && jupyter labextension install jupyter-matplotlib
    
RUN jupyter labextension install @jupyterlab/git --no-build\
	&& pip install jupyterlab-git\
	&& jupyter serverextension enable --py jupyterlab_git

#install fastai
RUN conda install -y -c fastai fastai\
	&& conda clean -ya

#install ipywidgets becuause it get not installed by fastai: TODO look at this issue
RUN conda install ipywidgets\
	&& conda clean -ya
	
# rebuild jupyterlab for new installed extensions
RUN jupyter lab build
# enable container user to modify conda installation
#RUN chmod -R 777 /opt 
# Set the default command to jupyter
CMD jupyter lab --ip=0.0.0.0 --no-browser
