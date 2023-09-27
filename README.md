# [ICCV 2023] FlipNeRF: Flipped Reflection Rays for Few-shot Novel View Synthesis

## [Project Page](https://shawn615.github.io/flipnerf/) | [arXiv](https://arxiv.org/abs/2306.17723)

This repository contains the code release for the ICCV 2023 project [FlipNeRF: Flipped Reflection Rays for Few-shot Novel View Synthesis](https://shawn615.github.io/flipnerf/).
The code is based on [MixNeRF implementation](https://github.com/shawn615/MixNeRF).
Contact [Seunghyeon Seo](https://shawn615.github.io/) if you have any questions. :)

![Teaser Image](teaser.png)

## About FlipNeRF

Neural Radiance Field (NeRF) has been a mainstream in novel view synthesis with its remarkable quality of rendered images and simple architecture. Although NeRF has been developed in various directions improving continuously its performance, the necessity of a dense set of multi-view images still exists as a stumbling block to progress for practical application. In this work, we propose FlipNeRF, a novel regularization method for few-shot novel view synthesis by utilizing our proposed flipped reflection rays. The flipped reflection rays are explicitly derived from the input ray directions and estimated normal vectors, and play a role of effective additional training rays while enabling to estimate more accurate surface normals and learn the 3D geometry effectively. Since the surface normal and the scene depth are both derived from the estimated densities along a ray, the accurate surface normal leads to more exact depth estimation, which is a key factor for few-shot novel view synthesis. Furthermore, with our proposed Uncertainty-aware Emptiness Loss and Bottleneck Feature Consistency Loss, FlipNeRF is able to estimate more reliable outputs with reducing floating artifacts effectively across the different scene structures, and enhance the feature-level consistency between the pair of the rays cast toward the photo-consistent pixels without any additional feature extractor, respectively. Our FlipNeRF achieves the SOTA performance on the multiple benchmarks across all the scenarios.

**TL;DR:** We utilize the flipped reflection rays as additional training resources for the few-shot novel view synthesis, leading to more accurate surface normal estimation.

## Installation

We recommend to use [Anaconda](https://www.anaconda.com/products/individual) to set up the environment. First, create a new `flipnerf` environment: 

```conda create -n flipnerf python=3.6.15```

Next, activate the environment:

```conda activate flipnerf```

You can then install the dependencies:

```pip install -r requirements.txt```

Finally, install jaxlib with the appropriate CUDA version, e.g. if you have CUDA 11.0:

```
pip install --upgrade pip
pip install --upgrade jaxlib==0.1.68+cuda110 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

Note: If you run into problems installing jax, please see [the official documentation](https://github.com/google/jax#pip-installation-gpu-cuda) for additional help.

## Data

Please follow [RegNeRF's data preparation instructions](https://github.com/google-research/google-research/tree/master/regnerf) to prepare the DTU and LLFF datasets.

## Running the code

### Training an new model

For training a new model from scratch, you need to first need to define your CUDA devices. For example, when having access to 4 GPUs, you can run

```export CUDA_VISIBLE_DEVICES=0,1,2,3```

and then you can start the training process by calling

```python train.py --gin_configs configs/{CONFIG} ```

where you replace `{CONFIG}` with the config you want to use. For example, for running an experiment on the Blender dataset with 4 input views, you would choose the config `blender4.gin`. In the config files, you might need to adjust the `Config.data_dir` argument pointing to your dataset location. For the DTU dataset, you might further need to adjust the `Config.dtu_mask_path` argument.

Once the training process is started, you can monitor the progress via the tensorboard by calling
```
tensorboard --logdir={LOGDIR}
```
and then opening [localhost:6006](http://localhost:6006/) in your browser. `{LOGDIR}` is the path you indicated in your config file for the `Config.checkpoint_dir` argument. 

### Rendering test images

You can render and evaluate test images by running

```python eval.py --gin_configs configs/{CONFIG} ```

where you replace `{CONFIG}` with the config you want to use. Similarly, you can render a camera trajectory (which we used for our videos) by running

```python render.py --gin_configs configs/{CONFIG} ```


### Using a pre-trained model

You can find our pre-trained models, split into the 6 zip folders for the 6 different experimental setups, here: https://drive.google.com/drive/folders/1lpI8GV3-31VTbX8Vb-YKwQ82vZlW-Uv7?usp=sharing

After downloading the checkpoints, you need to change the `Config.checkpoint_dir` argument in the respective config file accordingly to use the pre-trained model. You can then render test images or camera trajectories as indicated above.

## Citation

If you find our work useful, please consider citing
```
@InProceedings{Seo_2023_ICCV,
    author    = {Seo, Seunghyeon and Chang, Yeonjin and Kwak, Nojun},
    title     = {FlipNeRF: Flipped Reflection Rays for Few-shot Novel View Synthesis},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {22883-22893}
}

@InProceedings{Seo_2023_CVPR,
    author    = {Seo, Seunghyeon and Han, Donghoon and Chang, Yeonjin and Kwak, Nojun},
    title     = {MixNeRF: Modeling a Ray With Mixture Density for Novel View Synthesis From Sparse Inputs},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {20659-20668}
}
```
*The template is borrowed from the [RegNeRF repository](https://github.com/google-research/google-research/tree/master/regnerf).
