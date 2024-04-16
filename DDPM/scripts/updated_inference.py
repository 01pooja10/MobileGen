import warnings
warnings.filterwarnings("ignore")

import argparse, logging, copy
from types import SimpleNamespace
from contextlib import nullcontext

import torch
from torch import optim
import torch.nn as nn
from torch.nn.utils import prune
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity
from fastprogress import progress_bar

import wandb
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt

from torch.fft import fft2, ifft2
from fvcore.nn import FlopCountAnalysis

from modules import UNet_conditional, EMA
from filtering import ImageFilter
from unet_compressed_v1 import UNet_conditional_student_v1
from unet_compressed_v2 import UNet_conditional_student_v2

class Diffusion:
	def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=256, num_classes=10, c_in=3, c_out=3, device="cuda", version=1, **kwargs):
		self.noise_steps = noise_steps
		self.beta_start = beta_start
		self.beta_end = beta_end

		self.beta = self.prepare_noise_schedule().to(device)
		self.alpha = 1. - self.beta
		self.alpha_hat = torch.cumprod(self.alpha, dim=0)
  
		self.version = version
		
		self.img_size = img_size
		if version==1:
			self.model = UNet_conditional_student_v1(c_in, c_out, num_classes=num_classes,**kwargs).to(device)
		elif version==2:
			self.model = UNet_conditional_student_v2(c_in, c_out, num_classes=num_classes,**kwargs).to(device)
		self.ema_model = copy.deepcopy(self.model).eval().requires_grad_(False)
		self.device = device
		self.c_in = c_in
		self.num_classes = num_classes

	def alpha_beta(self):
		return (self.alpha, self.beta, self.alpha_hat)

	def prepare_noise_schedule(self):
		return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

	def sample_timesteps(self, n):
		return torch.randint(low=1, high=self.noise_steps, size=(n,))

	def noise_images(self, x, t):
		"Add noise to images at instant t"
		sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
		sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
		Ɛ = torch.randn_like(x)
		return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ


	#new pruning function - modified
	def pruning(self, model):

		mod = copy.deepcopy(model).to(self.device)
		l1_prune = torch.nn.utils.prune.L1Unstructured(0.3)

		for name,module in mod.named_modules():

			if isinstance(module,nn.Linear):
				l1_prune.apply(module,name='weight',amount=0.3)

			elif isinstance(module,nn.Conv2d):
				l1_prune.apply(module,name='weight',amount=0.3)

		return mod


	def find_weights(self,weights,bits):

		min_wt, max_wt = weights.min(), weights.max()
		qmin, qmax = -2.0 ** (bits - 1), 2.0 ** (bits - 1) - 1

		scale = (max_wt - min_wt) / (qmax - qmin)

		zero_point = qmin - min_wt / scale

		quantized_weights = torch.round(weights / scale + zero_point)
		quantized_weights = torch.clamp(quantized_weights, qmin, qmax).int()

		return quantized_weights

	#new manual quantization function
	def quantize_eval(self,model):

		for param_name, param in model.named_parameters():

			if 'weight' in param_name:
				new_name = param_name.replace(".", "_")
				quantized_weights = self.find_weights(param.data, bits=8)
				setattr(model, new_name, param)

		return model


	@torch.inference_mode()
	def sample(self, model, labels, cfg_scale=0):
		n = len(labels)
		logging.info(f"Sampling {n} new images....")
		model.eval()
		model = self.quantize_eval(model)
		#model = self.pruning(model)


		with torch.inference_mode():
			x = torch.randn((n, self.c_in, self.img_size, self.img_size)).to(self.device)
			for i in progress_bar(reversed(range(1, self.noise_steps)), total=self.noise_steps-1, leave=False):
				t = (torch.ones(n) * i).long().to(self.device)
				predicted_noise = model(x, t, labels)
				if cfg_scale > 0:
					uncond_predicted_noise = model(x, t, None)
					predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
				alpha = self.alpha[t][:, None, None, None]
				alpha_hat = self.alpha_hat[t][:, None, None, None]
				beta = self.beta[t][:, None, None, None]
				if i > 1:
					noise = torch.randn_like(x)
				else:
					noise = torch.zeros_like(x)
				x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
		x = (x.clamp(-1, 1) + 1) / 2
		x = (x * 255).type(torch.uint8)
		return x

	#new plot images function integrated with image filters
	def plot_images(self,images):

		ilist = []
		filter_instance = ImageFilter(filter_type='weighted', weight = [0.25, 0.25, 0.25, 0.25])

		for i in images.cpu():
			image = cv2.cvtColor(i.permute(1,2,0).numpy(),cv2.COLOR_RGBA2RGB)
			filtered_image = filter_instance.apply_filter(image)
			ilist.append(filtered_image)

		ilist = torch.Tensor(np.array(ilist))
		return ilist

	#new method to analyze time
	def analyze_time(self, model, inp, t, y):
		with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
			with record_function("model_inference"):
				model(x=inp,t=t,y=y)
		print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

	def count_parameters(self, model):
		return sum(p.numel() for p in model.parameters())

	def count_flops(self, model, inputs, t, y):
		flops = FlopCountAnalysis(model, (inputs, t, y))
		return flops.total()

	def model_size(self, model):
		param_size = 0
		buffer_size = 0
		for param in model.parameters():
			param_size += param.nelement() * param.element_size()

		for buffer in model.buffers():
			buffer_size += buffer.nelement() * buffer.element_size()

		size_all_mb = (param_size + buffer_size) / 1024**2
		print('Size: {:.3f} MB'.format(size_all_mb))

		return size_all_mb

def infer(obj_class, version):
	n = 1
	device = "cuda"
	
	if version==1:
		model = UNet_conditional_student_v1(num_classes=10, remove_deep_conv=True).to(device)
		ckpt = torch.load("/content/drive/MyDrive/csc2231/models/v1/ema_student_ckpt.pt")
	elif version==2:
		model = UNet_conditional_student_v2(num_classes=10, remove_deep_conv=False).to(device)
		ckpt = torch.load("/content/drive/MyDrive/csc2231/models/v2/ema_student_ckpt.pt")
	model = model.eval().requires_grad_(False)
	model.load_state_dict(ckpt)
	
	diffusion = Diffusion(img_size=64, device=device, version=version)
	y = torch.Tensor([obj_class] * n).long().to(device)
	x = diffusion.sample(model, y, cfg_scale=3)
	generated_img = x[0].cpu().permute(1, 2, 0)
	return generated_img
