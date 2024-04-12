import os
import torch
from tqdm import tqdm
import torch.nn as nn
from torch import optim
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

class Diffusion:
    def __init__(self, beta_s = 1e-4, beta_e = 0.02, noise_steps = 1000, img_dim = 64, device="cuda") -> None:
        self.beta_s = beta_s
        self.beta_e = beta_e
        self.nsteps = noise_steps
        self.img_size = img_dim
        self.device = device
        #;locals
        self.beta = self.Scheduler().to(self.device)
        self.alpha = 1 - self.beta
        self.alpha_prod = torch.cumprod(self.alpha, dim=0)
        
    def Scheduler(self):
        return torch.linspace(self.beta_s, self.beta_e, self.nsteps)
    
    def AddNoise(self, x, t):
        term1 = torch.sqrt(self.alpha_prod[t])[:,None,None,None]
        term2 = torch.sqrt(1 - self.alpha_prod[t])[:,None,None,None]
        eps = torch.randn_like(x)
        res =  term1 * x + term2 * eps
        return res, eps
    
    def Timesteps(self,n):
        return torch.randint(low=1, high=self.nsteps,size=(n,))
    
    def Sampler(self, model, n, label):
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.nsteps)),position=0):
                t = (torch.ones(n)*i).long().to(self.device)
                e_pred = model(x, t, label)
                alpha = self.alpha[t][:,None,None,None]
                alpha_prod = self.alpha_prod[t][:,None,None,None]
                beta = self.beta[t][:,None,None,None]
                
                if i>1:
                    z = torch.randn_like(x)
                else:
                    z = torch.zeros_like(x)
                    
                final_x = 1 / torch.sqrt(alpha) * (x - ((1-alpha) / (torch.sqrt(1-alpha_prod)))*e_pred) + torch.sqrt(beta) * z
                
        model.train()
        
        final_x = (final_x.clamp(-1,1) + 1) / 2
        final_x = (final_x*255).type(torch.uint8)
        return final_x
