# MobileGen
A repository that deploys a diffusion model (DDPM) on mobile devices with sped-up inference and low memory consumption.

## Goals
- The objective of this project (MobileGen) is to generate images from the CIFAR 10 64x64 dataset given a label to condition the Denoising Diffusion Probabilistic Model  (DDPM).

- An instance of the DDPM model is initialized on a GPU from the cloud server and inference is performed before forwarding the resulting image to a mobile device.

- The main aim here is to make complex architectures such as  DDPMs accessible to a wider demographic on handy devices such as smartphones to harness the power of diffusion models.

## Optimization techniques
1. Quantization Aware Training
     The model was trained with intermittent shifts to the float16 weight space for the weight tensors involved in training the UNet to make the     
     backpropagation easier.
2. Knowledge Distillation
     The distillation process ensures that the student model gets some level of guidance in terms of training a smaller and more compressed student model. 
3. Pruning @ Inference
      Pruning ensures that certain connections (weights) of layers such as convolutions and Linear layers from our UNet model are sufficiently         
      compressed (pruned) using the L1 unstructed method available through PyTorch.

The link to model weights, and the TFLite model is available and can be accessed [here](https://drive.google.com/drive/folders/1OWXpgdDkLas5HAaJOBuOcPVtY0nZbU4l?usp=drive_link
).

The link to the CIFAR 64x64 dataset is available [here](https://www.kaggle.com/datasets/joaopauloschuler/cifar10-64x64-resized-via-cai-super-resolution).

## DDPM Results
 - Version 1
   
![image](https://github.com/01pooja10/MobileGen/assets/66198904/5dc4801e-e20a-4445-ba7c-439288b16df7)

- Version 2
  
![image](https://github.com/01pooja10/MobileGen/assets/66198904/825901ec-deeb-42a4-adf1-58c7712c5092)

## Inference Results

![image](https://github.com/01pooja10/MobileGen/assets/66198904/4bf03a29-80f3-46d9-979a-53da39062272)

## Mobile Deployment Diagram
The flowchart below depicts the overall flow of data (images and labels) between the model and our mobile application.

![image](https://github.com/01pooja10/MobileGen/assets/66198904/0c04808c-b6f2-4e02-805a-0acb96c61804)


## Contributors


<table align="center">
<tr align="center">


<td width:25%>
Pooja Ravi

<p align="center">
<img src = "https://avatars3.githubusercontent.com/u/66198904?s=460&u=06bd3edde2858507e8c42569d76d61b3491243ad&v=4"  height="120" alt="Ansh Sharma">
</p>
<p align="center">
<a href = "https://github.com/01pooja10"><img src = "http://www.iconninja.com/files/241/825/211/round-collaboration-social-github-code-circle-network-icon.svg" width="36" height = "36"/></a>
<a href = "https://www.linkedin.com/in/pooja-ravi-9b88861b2/">
<img src = "http://www.iconninja.com/files/863/607/751/network-linkedin-social-connection-circular-circle-media-icon.svg" width="36" height="36"/>
</a>
</p>
</td>


<td width:25%>
Aneri Gandhi

<p align="center">
<img src = "https://avatars.githubusercontent.com/u/40311799?v=4"  height="120" alt="Aneri Gandhi">
</p>
<p align="center">
<a href = "https://github.com/amg10"><img src = "http://www.iconninja.com/files/241/825/211/round-collaboration-social-github-code-circle-network-icon.svg" width="36" height = "36"/></a>
<a href = "https://www.linkedin.com/in/aneri-manoj-gandhi/">
<img src = "http://www.iconninja.com/files/863/607/751/network-linkedin-social-connection-circular-circle-media-icon.svg" width="36" height="36"/>
</a>
</p>
</td>


<td width:25%>
Sruthi Srinivasan

<p align="center">
<img src = "https://avatars.githubusercontent.com/u/63657533?v=4"  height="120" alt="Aneri Gandhi">
</p>
<p align="center">
<a href = "https://github.com/sruthi0107"><img src = "http://www.iconninja.com/files/241/825/211/round-collaboration-social-github-code-circle-network-icon.svg" width="36" height = "36"/></a>
<a href = "https://www.linkedin.com/in/sruthi-srinivasan/">
<img src = "http://www.iconninja.com/files/863/607/751/network-linkedin-social-connection-circular-circle-media-icon.svg" width="36" height="36"/>
</a>
</p>
</td>


</table>


## License
MIT © Pooja Ravi

This project is licensed under the MIT License - see the [License](LICENSE) file for details

[![License](http://img.shields.io/:license-mit-blue.svg?style=flat-square)](http://badges.mit-license.org)
