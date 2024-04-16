import torch
import torch.nn as nn
from tqdm import tqdm
from torch.nn.utils import prune
from torch.quantization import QuantStub, DeQuantStub

from unet import UNet
from ddpm import Diffusion

class Inference():
    
    def __init__(self, mpath, n, device='cuda', to_prune=False, to_quant=False):
        super(Inference,self).__init__()
        
        self.mpath = mpath
        self.n = n
        self.device = device
        self.to_prune = to_prune
        self.to_quant = to_quant
        
        self.diffu = Diffusion()
        
        self.beta = self.diffu.Scheduler().to(self.device)
        self.alpha = 1 - self.beta
        self.alpha_prod = torch.cumprod(self.alpha, dim=0)
        
    def load_weights(self,cond=True):
        
        '''
        cond = conditional or unconditional image generation
        
        '''
        
        if cond:
            model = UNet(nc=10,time=64)
        else:
            model = UNet(time=64)
        
        to_load = torch.load(self.mpath)
        model.load_state_dict(to_load)
        
        return model
        
    def prune(self,model):
            
        mod1 = copy.deepcopy(model).to(self.device)
        #print(mod1.named_modules)
        
        for name,module in mod1.named_modules():
           
            if isinstance(module,nn.Linear):
                prune.l1_unstructured(module,name='weight',amount = 0.4)

            elif isinstance(module,nn.Conv2d):
                prune.l1_unstructured(module,name='weight',amount = 0.3)
                
        return mod1
    
    def quantized(self,x,t,model):
        
        '''
        x = input
        t = timestep
        model = model (UNet)
        
        '''
        
        xquant = QuantStub()
        x = xquant(x)
        t = xquant(t)
        out = model(x,t)
        dquant = DeQuantStub()
      
        return dquant(out)
    
    def initialize_model(self):
        
        if self.to_prune:
            model = self.prune(self.load_weights(self.mpath))
        else:
            model = self.load_weights(self.mpath)
        
        model.eval()
        
        return model
    
    def sample(self,n=64,nsteps=500,im_size=64):
        
        '''
        n = batch size
        nsteps = noise steps
        im_size = size of the image (hxw)
        
        '''
        
        model = self.initialize_model()
        
        with torch.no_grad():
            x = torch.randn((n, 3, im_size, im_size)).to(self.device)
            
            for i in tqdm(reversed(range(1, nsteps)),position=0):
                t = (torch.ones(n)*i).long().to(self.device)
                
                if self.to_quant:
                    e_pred = self.quantized(x=x,t=t,model=model)
                else:
                    e_pred = model(x, t)
                    
                alpha = self.alpha[t][:,None,None,None]
                alpha_prod = self.alpha_prod[t][:,None,None,None]
                beta = self.beta[t][:,None,None,None]
                
                if i>1:
                    z = torch.randn_like(x)
                else:
                    z = torch.zeros_like(x)
                    
                final_x = 1/torch.sqrt(alpha) * (x - ((1-alpha) / torch.sqrt(1-alpha_prod))*e_pred) + torch.sqrt(beta) * z
                
        
        final_x = (final_x.clamp(-1,1) + 1) / 2
        final_x = (final_x*255).type(torch.uint8)
        return final_x
