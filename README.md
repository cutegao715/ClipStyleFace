# Disentangled Text-Driven Stylization of 3D Faces via Directional CLIP Losses

![](img/figure1.jpg)

## Description
The repository implements the **Disentangled Text-Driven Stylization of 3D Faces via Directional CLIP Losses** paper.
This paper proposes ClipStyleFace, a text-driven approach for 3D face stylization that leverages CLIP
(Contrastive Language-Image Pre-training) knowledge to create variations in both geometric and texture structures. 
ClipStyleFace consists of three modules: a deformable surface model for geometry deformation, 
a compact parameter space for texture transformation, and directional CLIP losses for semantic alignment
and domain correction.

## Getting Started

For all the methods described in the paper, it is required to have:

- Anaconda
- PyTorch >=1.7.1
- Packages from requirements.txt

### Notes

Here, the code relies on the [Rosinality](https://github.com/rosinality/stylegan2-pytorch/) pytorch implementation of StyleGAN2.
Some parts of the StyleGAN implementation were modified, so that the whole implementation is native pytorch. 

In addition to the requirements mentioned before, a pretrained StyleGAN2 generator will attempt to be downloaded with script *download.py*.

### Dependencies

Our model built on pytorch 1.9.0+ cu111, python 3.8, pytorch3d 0.7.1

### Pre-trained weights
Follow [[DECA](https://github.com/yfeng95/DECA)] to download DECA pre-trained weights. Put them in the 'data' folder.

Download the pretrained models from https://pan.baidu.com/s/1LseMReNHke2g8GpfiUvJNA?pwd=ns9r ns9r 
and put them in the 'pretrained/StyleGAN2'

### Data preparation
You can download the face dataset from [[FFHQ](https://github.com/NVlabs/ffhq-dataset)]
or generate your own dataset from the pretrained StyleGANs.
Then use [[DECA](https://github.com/yfeng95/DECA)] to extract their 3D obj. files  with dense verts.
Use psp encoder to extract the latent codes
Put them into datasets/images; datasets/latents; datasets/shapes; datasets/texture

## Model training

Here, we provide the code for the training.

In general training could be launched by following command

```
python main.py exp.config=td_single_ffhq.yaml
```
Our model needs two stages of training;
first save the images generated from the early stages
## Acknowledgements

For functions or scripts that are based on external sources, we acknowledge the origin individually in each file.
Our code is inspired by the following code repositories. Please make sure to refer to their and our license terms before downloading the pre-trained weights.

1. [[DECA](https://github.com/yfeng95/DECA)]
2. [[MICA](https://github.com/Zielon/MICA)]
3. [[FLAME](https://github.com/soubhiksanyal/FLAME_PyTorch)]
4. [[Hyperdomainnet](https://github.com/MACderRu/HyperDomainNet)]
# ClipStyleFace
