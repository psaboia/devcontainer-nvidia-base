# Dev Container NVIDIA based
That is an example of how to setup a NVIDIA DevContainer with GPU Support for Tensorflow/Keras, that follows the page [Setup a NVIDIA DevContainer with GPU Support for Tensorflow/Keras on Windows](https://alankrantas.medium.com/setup-a-nvidia-devcontainer-with-gpu-support-for-tensorflow-keras-on-windows-d00e6e204630).

## Prerequisites
- Docker engine (and setup .wslconfig to use more cores and memory than default)
- NVIDIA driver for the graphic card
- NVIDIA Container Toolkit (which is already included in Windows’ Docker Desktop; Linux users have to install it)
- VS Code with DevContainer extension installed

## Start the DevContainer
- Clone this repo.
- In VS Code press `Ctrl + Shift + P` to bring up the Command Palette. 
- Enter and find `Dev Containers: Reopen in Container`. 
- VS Code will starts to download the CUDA image, run the script and install everything, and finish opening the directory in DevContainer.
- The DevContainer would then run nvidia-smi to show what GPU can be seen by the container. Be noted that this works even without setting up cuDNN or any environment variables.

## Test with keras script for MNIST
The file `./src/train.py` is a short AutoKeras test script for you, which trains with the MNIST handwriting digit dataset with a pre-defined CNN model.
Open a new terminal and enter:
```bash
 python3 src/autokeras_script.py
``` 


## Setup details
### Dev Container definition
DevContainer definition `.devcontainer/devcontainer.json` uses the official CUDA developer image `nvidia/cuda:11.8.0-devel-ubuntu22.04` (not base or runtime), which supports AMD64 and ARM64 and have CUDA installed. 
It will run a script to install other stuff (including VS Code extensions) and finally run nvidia-smi after started up.

```json
{
  "name": "CUDA",
  "image": "nvidia/cuda:11.8.0-devel-ubuntu22.04", // https://hub.docker.com/r/nvidia/cuda/tags
  "runArgs": [
    "--gpus=all"
  ],
  "remoteEnv": {
    "PATH": "${containerEnv:PATH}:/usr/local/cuda/bin",
    "LD_LIBRARY_PATH": "$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64",
    "XLA_FLAGS": "--xla_gpu_cuda_data_dir=/usr/local/cuda"
  },
  "updateContentCommand": "bash .devcontainer/install-dev-tools.sh",
  "postCreateCommand": [
    "nvidia-smi"
  ],
  "customizations": {
    "vscode": {
      "extensions": [
        "ms-python.python",
        "ms-toolsai.jupyter",
        "ms-toolsai.vscode-jupyter-cell-tags",
        "ms-toolsai.jupyter-keymap",
        "ms-toolsai.jupyter-renderers",
        "ms-toolsai.vscode-jupyter-slideshow",
        "ms-python.vscode-pylance"
      ]
    }
  }
}
```

### Installing basic Linux tools, Python 3, Python packages and cuDNN
The script for installing basic Linux tools, Python 3, Python packages and cuDNN is `.devcontainer/install-dev-tools.sh`. Downloaded file will be removed so it won’t appear in your local directory.

```bash
# update system
apt-get update
apt-get upgrade -y
# install Linux tools and Python 3
apt-get install software-properties-common wget curl \
    python3-dev python3-pip python3-wheel python3-setuptools -y
# install Python packages
python3 -m pip install --upgrade pip
pip3 install --user -r .devcontainer/requirements.txt
# update CUDA Linux GPG repository key
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
dpkg -i cuda-keyring_1.0-1_all.deb
rm cuda-keyring_1.0-1_all.deb
# install cuDNN
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub
add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /" -y
apt-get update
apt-get install libcudnn8=8.9.0.*-1+cuda11.8
apt-get install libcudnn8-dev=8.9.0.*-1+cuda11.8
# install recommended packages
apt-get install zlib1g g++ freeglut3-dev \
    libx11-dev libxmu-dev libxi-dev libglu1-mesa libglu1-mesa-dev libfreeimage-dev -y
# clean up
pip3 cache purge
apt-get autoremove -y
apt-get clean
```

### Third party Python packages
The file `.devcontainer/requirements.txt` contains all third party Python packages you wish to install. Modify the list as you like.

```
numpy
scikit-learn
matplotlib
tensorflow
autokeras
ipykernel
regex
```


**Source**: [Setup a NVIDIA DevContainer with GPU Support for Tensorflow/Keras on Windows](https://alankrantas.medium.com/setup-a-nvidia-devcontainer-with-gpu-support-for-tensorflow-keras-on-windows-d00e6e204630)