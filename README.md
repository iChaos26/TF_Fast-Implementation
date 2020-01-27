# TF_Fast-Implementation - Simple framework to test the new version of Tensorflow, whose now is based on Keras API.

# Requeriments:
A prebuilt tensor ambient, with TF 2.0. The fastest way is to download docker prebuilt TensorFlow with gpu: https://www.tensorflow.org/install/docker

## TF Docker:
Install Docker and NVIDIA Docker support: 
```sh
$ docker pull tensorflow/tensorflow                     
$ docker pull tensorflow/tensorflow:devel-gpu           
$ docker pull tensorflow/tensorflow:latest-gpu-jupyter   
``` 

## Check GPU
```sh
$ lspci | grep -i nvidia
```
## Running TF Docker with GPU on Jupyter:
```sh
$ docker run -it --rm -v $(realpath ~/notebooks):/tf/notebooks -p 8888:8888 --gpus all tensorflow/tensorflow:latest-gpu-py3-jupyter
```
## Running TF Docker bash:
```sh
$ docker run -it --rm --gpus all tensorflow/tensorflow:latest-gpu-py3 python
```
