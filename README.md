# TGSC_LSTM-TCN-GANs

## Model Structure

![figure2_a](https://user-images.githubusercontent.com/57590655/148371274-3a2b0d83-78b6-46d5-ae62-d5c67d791f46.jpg)
![figure2_b](https://user-images.githubusercontent.com/57590655/148371278-665a1d36-a663-41b8-b3c0-6d9890aca049.jpg)
FIGURE 1.  (a) The proposed 5G Traffic-GAN model architecture. (b) Temporal block structure of discriminator.

![figure3](https://user-images.githubusercontent.com/57590655/148371279-f4f2126b-36b6-4909-88fb-987043d520e9.jpg)
FIGURE 2. Discriminator network architecture

![figure4](https://user-images.githubusercontent.com/57590655/148371281-37c9d661-a22c-4f79-b817-5a8b4b7fced1.jpg)
FIGURE 3. Generator network architecture

## Getting Started

### Prerequisites

The following should be installed.

```
Window OS V11
WSL V2
Docker desktop for window
docker image : eungbean/deepo:lab
Nvidia Driver : 510.06
CUDA : 11.6
CUDNN : 7.6.5
```

### Device environment

I trained models in this computer environment.

```
CPU : Ryzen 7 1700
RAM : 16GB
GPU : Geforce GTX 1070 (6G)
```

Train time

```
1 epoch : 565 sec
665 epoch (best performance) : 104 hour
```

### Installing

You can install modules related to the current project with the following items.
```
pip install tensorboardX
```

## Running the tests

### Start Training

```
> cd your_path/TGSC_LSTM-TCN-GANs
> python3 main.py
```

