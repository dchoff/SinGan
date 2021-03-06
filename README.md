# SinGAN: Learning a Generative Model from a Single Natural Image

## NOTE
This repository is not the official implementation. The official implementation can be found here [SinGAN Official Pytorch implementation](https://github.com/tamarott/SinGAN).

This repository is based off the following implementation of SinGAN: [SinGAN FriedRonaldo](https://github.com/FriedRonaldo/SinGAN). The full pipeline gets hooked up to a trained CartoonGAN model, which is built off of the following repo: [CartoonGAN Yijunmaverick](https://github.com/Yijunmaverick/CartoonGAN-Test-Pytorch-Torch).

## SinGAN's Model and our Goal
Capturing the distribution of complex datasets is a task that is still considered a challenge despite continuous performance and quality increases experienced by GANs trained on visual data. In order to resolve this issue, traditional models focus on training the GAN on different input signals or training the model to do a specific task. SinGAN aims to resolve these issues: it is capable of generating new samples of arbitrary size and aspect ratio that look distinct from the original photo, but still conveys the same visual information and maintains the general global structures that the photo contains. This approach is not limited to texture images like some models are, and is unconditional (i.e. generates the samples via random noise). It does all this while maintaining good results, as seen by it easily fooling human vision in [Section 3.1 of their paper](https://arxiv.org/pdf/1905.01164.pdf). It accomplishes this but using a pyramidal pipeline of fully convolutional GANs, each of which attempt to learn the patch distribution of the image at a different scale, as seen below:

![structure](./src/structure.png)

However, the model itself injects noise into the outputted image, leading to noise being propagated through systems one may want to parse the output to, such as other GANs. One potential way of resolving this is the approach we took in this repository, where we perform denoising on the outputted images prior to inputting it into other models. However, while this greatly increases the results of the output in models like CartoonGAN, there are still signs of artifacting and noise propagation that could be improved upon. 


By manipulating the model, training it multiple times at different granularities, or fusing its outputs carefully, you can achieve the image modification results that are discussed in the official paper - however, this repository mainly focuses on the raw outputs of the model (similar to the [FriedRonaldo](https://github.com/FriedRonaldo/SinGAN) implementation) and aims to show a potential way of allowing the results to be used for other purposes where fine-grained quality greatly matters, e.g. data augmentation.

## General Deliverables
All of these can be found within the repo itself, but for sake of clarity as to where they are, see the following:
* [Example raw outputs](./code/results)
* [Example stylized outputs](./code/test_output/results)
* [Trained example SinGAN models](./code/logs)
* [Pre-trained CartoonGAN models](./code/cartoonGAN/pretrained_model)
* [Docker image](https://hub.docker.com/r/dchoff/singan)
* [Dockerfile](./Dockerfile)

## Input and Output
The input for our model is arbitrary sized images. To try and replicate the results of the paper, and to obtain better consistency, we trained and ran models that were trained on images from the official dataset used the authors of SinGAN. These input images were primarily PNG files, and typically had a resolution near 250x250.

The model outputs two things: the raw output of the SinGAN model, and stylized images from processing denoised versions of said images through CartoonGAN. The size of the outputted images are variable according to arguments, but due to hardware limitations and desiring consistency the results we discuss are primarily 250x250, similar to the majority of the results in the paper.


  
## Folder Hierarchy
  Images that you wish to train on should be placed in the trainPhoto and testPhoto directories, as seen below in the example hierarchy.

  * Directory hierarchy :
  ```
  Project
  |--- data
  |    |--- SinGANdata
  |         |--- trainPhoto
  |         |    |--- balloons.png
  |         |    |--- birds.png
  |         |    |--- ... other images used in the official paper
  |         |    
  |         |--- trainPhoto
  |         |    |--- yourPhoto.png
  |         |
  |         |--- testPhoto
  |              |--- yourPhoto.png
  |
  |--- code
       |--- cartoonGAN
       |    |--- pretrained_model
       |         |--- Hayao_net_G_float.pth
       |         |--- ...
       |
       |--- models
       |        |--- ...
       |
       |--- results
       |    |--- ...
       |
       |--- test_output
       |    |--- ...
       |
       |--- main.py 
       |--- train.py
       | ...
       
  ```
   
## How to Run via Docker

### Build Dockerfile
The Dockerfile can be found [here](./Dockerfile), and built using the following command:
```
docker build -t singan .
```

Alternatively, it is also pushed to [docker hub](https://hub.docker.com/r/dchoff/singan).

Note that as the repository requires GPU access, nvidia-docker should be installed for proper usage, and it does not support Windows/Mac native machines. If you are running a non-Linux machine, I recommend you refer to the "How to Run via Code" section below, which should be relatively simple to setup.

Note that the instructions below *may* not work, as I don't have access to a Linux machine and thus was not able to get Docker running with GPU access.

If you are on a Linux machine, and still wish to use Docker, then you should make sure that nvidia-docker is installed, as described on their official page [here](https://github.com/NVIDIA/nvidia-docker). Once this is done, you'll want to assign the docker a single GPU and use the image for testing, e.g.:
```
docker run --gpus 1 -ti --name sg dchoff/singan
```

Which will assign the container a single gpu and put you in an interactive shell mode. You should then be able to run the code using the following commands:
```
conda activate singan
cd SinGan/code/
python main.py --gpu 0 --img_to_use 0 --img_size_max 250 --gantype zerogp --validation --load_model SinGAN_2020-03-02_20-39-53
```

which will activate the singan environment which has the relevant dependencies, navigate you to the code directory of this repo, and then run the code (as described in the section below on How to run via Code).

Once you're done running the code, exit the interactive shell and use the following command to get the container ID:
```
> docker ps -a
CONTAINER ID        IMAGE               COMMAND             CREATED              STATUS                     PORTS               NAMES
a9e23438b1a0        dchoff/singan      "/bin/bash"         About a minute ago   Exited (0) 3 seconds ago 
```

You'll then want to use `docker cp` to copy the output images to wherever you're able to open and view them. For pulling both the raw and stylized outputs of a session, e.g. SinGAN_2020-03-02_20-39-53, you would use the following:
```
docker cp a9e23438b1a0:/SinGan/code/results/SinGAN_2020-03-02_20-39-53 ./SinGAN_2020-03-02_20-39-53
docker cp a9e23438b1a0:/SinGan/code/test_output/results/SinGAN_2020-03-02_20-39-53 ./SinGAN_2020-03-02_20-39-53/stylized

```

You should now have the images in an accessible location where you can review the results.


## How to Run via Code
### Conda Environment
#### 1. Install Conda if not already installed
See [here](https://docs.anaconda.com/anaconda/install/) for instructions on installation

#### 2. Create the conda environment from the provided YML file
```
conda env create -f environment.yml
```
#### 3. Activate the environment
```
conda activate singan
```

#### 3a. Install remaining packages via pip
For some reason, the YML file doesn't always install the pip packages listed in the YML file. So run the following to make sure you have all the other packages you need:
```
pip install -r requirements.txt
```

#### 4. Run the code as desired, as seen in the following sections

### Arguments
   * gantype
       * Loss type of GANs. You can choose among "wgangp, zerogp, lsgan". Recommended to use "zerogp" to reduce run variance.
   * model_name
       * Prefix of the directory of log files.
   * workers
       * Workers to use for loading dataset.
   * img_size_max
       * Size of largest image. = Finest
   * img_size_min
       * Size of smallest image. = Coarsest
   * img_to_use
       * Index of the image to use. If you do not change, it will be sampled randomly.
   * load_model
       * Directory of the model to load.
   * validation
       * Validation mode
   * gpu
       * GPU number to use. You should set. Unless it utilizes all the available GPUs.

SinGAN uses only one image to train and test. Therefore multi-gpus mode is not supported.
   
### Train
Use NS loss with zero-cented GP and 0-th gpu. The train image will be selected randomly. It will generate (1052, 1052) images at last.
```
python main.py --gpu 0 --gantype zerogp --img_size_max 250
```


Use WGAN-GP loss to train and 0-th gpu.
```
python main.py --gpu 0 --img_to_use 0 --img_size_max 250 --gantype wgangp
```


### Test trained model, e.g. SinGAN_2020-03-02_20-39-53
Note that when using this option, you should take care to choose the `img_to_use` flag to be the indexed photo of the photo in `trainData` unless you desire to input a custom image into the model purposes other than generating variations of the training image. If you are unsure what image it was trained on, you can check the log folder for the `record.txt` file, which will specify whih IMGTOUSE was chosen. So long as files have not been moved around in the trainPhoto and testPhoto folders, this should be the proper index to use.
```
python main.py --gpu 0 --img_to_use 0 --img_size_max 250 --gantype zerogp --validation --load_model SinGAN_2020-03-02_20-39-53
```


### Manually run CartoonGAN on a trained model's results, e.g. SinGAN_2020-03-02_20-39-53
```
python ./cartoonGAN/test.py --input_dir ./results/SinGAN_2020-03-02_20-39-53 --gpu 0 --mod_name SinGAN_2020-03-02_20-39-53
```

## Provided Models
One of the benefits of this model is that a trained model is rather small. We've provided several different models, which can be found [here](./code/logs). These contain snapshots of codebase at the time and a text file called `record.txt` which contains the GAN type which was used and the index of the image trained upon.

Results from having run these models can be found [here](./code/results). The stylized results can be found [here](./code/test_output/results). The pre-trained models for CartoonGAN can similarly be found [here](./code/cartoonGAN/pretrained_model).

Please note that if you wish to reobtain the results for a provided model, you should empty out the results that are already in that model's results directory, as the code will not attempt to overwrite images that are already there. Additionally, the outputted generated images are in the format GEN_x_y.png (or GEN_x_y.jpg if stylized), where x refers to the current scale and y refers to the current iteration in that scale. Thus, higher x's are demonstrable of the final results of the models (e.g. GEN_8_43.png), while y is indicative of more variability.

## Results

By now, you have either trained your new model or have ran a pretrained model and want to see the results from it.
If you have trained a new model, the saved model can be found under `logs`, as seen below. Images that were output during the training/validation process can be found under `results/SinGAN_2020-03-02_20-39-53` for raw output, and `test_output/results/SinGAN_2020-03-02_20-39-53` for the images that have been denoised and processed through CartoonGAN.
```
  Project
  |--- data
  |    |--- SinGANdata
  |         |--- paper_photos
  |         |--- trainPhoto
  |         |--- testPhoto
  |
  |--- code
       |--- cartoonGAN
       |    |--- pretrained_model
       |         |--- Hayao_net_G_float.pth
       |         |--- ...
       |
       |---datasets
       |--- logs
       |    |---SinGAN_2020-03-02_20-39-53
       |    |--- ...
       |
       |--- models
       |        |--- generator.py
       |        |--- ...
       |
       |--- results
       |    |--- SinGAN_2020-03-02_20-39-53
       |         |--- Gen_0_0.png
       |         |--- ... more images from trained model
       |
       |--- test_output
       |    |--- denoised_images
       |    |--- results
       |         |--- SinGAN_2020-03-02_20-39-53
       |              |--- Hayao
       |              |--- Hosoda
       |              |--- Paprika
       |              |--- Shinkai
       |
       |--- main.py 
       |--- train.py
       | ...
       
```

Raw results tend to look rather satisfying to the human eye, as seen in the table below. However, SinGAN injects a lot of noise into the image. If you use only the raw result, you will obtain a rather poor stylization of the image, as seen in the second column. The final column shows the result after denoising, showing that some form of post-processing pipeline, or an active noise reduction algorithm during the training, should be used when wanting to use the output for things like data augmentation. Note that the images shown below were taken from different trained models. While active noise reduction seems like it would be rather desirable, this could potentially harm the results of the model itself, as the patch distributions may not be learned as well due to such a change.

As seen, there is room for further improvements before the output of SinGAN is completely viable for purposes which require high fidelity, such as data augmentation. While the results were improved upon, there are still obvious signs of noise propagation and artifacting that need to be resolved. One thing that needs to be considering is the fact that denoising algorithms often inject forms of noise into the image as a result of the denoising process, which should be considered when decided how to further improve these results. The specific denoising algorithm used, non-local means denoising, injects white noise into the image (i.e. Gaussian noise) which should have little impact on the image. Additionally passes of denoising could be done as an attempt to further improve the results, but this would mean further white noise injection, and potentially losing clarity of small structural information of specific color information that could be important in specific use cases.

Raw Output             |  Hayao-stylized Raw Output    |  Hayao-stylized Denoised
:-------------------------:|:-------------------------:|:-------------------------:
![](./src/birds.png)  |  ![](./src/Hayao_birds_noisy.jpg) | ![](./src/Hayao_birds_denoised.jpg)


