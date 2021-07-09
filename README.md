# Beetect

***Detecting and tracking honeybees***

Watch the [sample video](https://drive.google.com/file/d/1UqFB-nSua-if6lOqO-vTtg23VreHf8W6/view?usp=sharing) ran on the trained model.

## Recommended paths:

#### NVIDIA docker
follow: https://github.com/NVIDIA/nvidia-docker

#### Simple PyTorch GPU installation through NGC and Docker
follow: https://ngc.nvidia.com/catalog/containers/nvidia:pytorch

## Steps for installing Horovod
https://github.com/horovod/horovod

#### Install Open MPI
follow: https://www.open-mpi.org/software/ompi/v4.0/

For OS X:
https://stackoverflow.com/questions/42703861/how-to-use-mpi-on-mac-os-x
brew install openmpi
CFLAGS=-mmacosx-version-min=10.9 pip install horovod

#### Check that g++-4.9 or above is installed for PyTorch
For Linux:
conda install -c anaconda gxx_linux-64

#### Install Horovod pip package
MacOS:
CFLAGS=-mmacosx-version-min=10.9 pip install horovod

GPU:
follow https://github.com/horovod/horovod/blob/master/docs/gpus.rst
