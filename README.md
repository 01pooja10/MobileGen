# MobileGen
A repository that deploys a diffusion model (DDPM) on mobile devices with sped-up inference and low memory consumption.

##Goals
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
      compressed (pruned) using L1 unstructed method available through PyTorch.


