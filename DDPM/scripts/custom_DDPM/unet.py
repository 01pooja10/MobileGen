import torch
import torch.nn as nn
import torch.nn.functional as F


class EMA:
    
    def __init__(self, beta):
		super(EMA,self).__init__()
        self.beta = beta
        self.step = 0
    
    def reset_parameters(self,ema_mod,mod):
        ema_mod.load_state_dict(mod.state_dict())
        
    def step_ema(self,ema_mod,mod,start=2000):
        if self.step<start:
            self.reset_parameters(ema_mod,mod)
            self.step += 1
            return
        self.update_model(ema_mod=ema_mod,mod=mod)
        self.step += 1
    
    def update_model(self,ema_mod,mod):
        for mod_p,ema_p in zip(mod.parameters(),ema_mod.parameters()):
            new_w,old_w =  mod_p.data, ema_p.data
            ema_p.data = self.update_avg(old_w,new_w)
    
    def update_avg(self,orig,new):
        out = orig * self.beta + (1+self.beta) * new
        
    

class DoubleConv(nn.Module):
    def __init__(self,in_ch,out_ch,mid_ch=None,resid=False):
        super(DoubleConv,self).__init__()
        
        self.res = resid
        
        if not mid_ch:
            mid_ch = out_ch
            
        self.dconv = nn.Sequential(nn.Conv2d(in_ch,mid_ch,kernel_size=3,padding=1,bias=False),
                                   nn.GroupNorm(1,mid_ch),
                                   nn.GELU(),
                                   nn.Conv2d(mid_ch,out_ch,kernel_size=3,padding=1,bias=False),
                                   nn.GroupNorm(1,out_ch))
        
    def forward(self,x):
        if self.res:
            return F.gelu(x + self.dconv(x))
        else:
            return self.dconv(x)
            

class SelfAttention(nn.Module):
    def __init__(self,ch,heads=2):
        super(SelfAttention,self).__init__()
        
        self.channels = ch
        #self.size = size
        
        self.multihead = nn.MultiheadAttention(ch,heads,batch_first=True)
        self.norm = nn.LayerNorm([ch])
        self.ffs = nn.Sequential(nn.LayerNorm([ch]),
                                 nn.Linear(ch,ch),
                                 nn.GELU(),
                                 nn.Linear(ch,ch))
        
    def forward(self,x):
       
        #print('inner x',x.size())
        
        bs,chn,h,w = x.size()
        x = x.reshape(bs,chn,h*w)
        
        x = x.transpose(1,2)
        
        normx = self.norm(x)
        
        attn,_ = self.multihead(normx,normx,normx)
        attn = attn + x
        #print('attn',attn.size())
        
        attn = self.ffs(attn) + attn
        attn = attn.transpose(1,2)
        
        attn = attn.reshape(bs,chn,h,w)
        
        return attn
    

class Down(nn.Module):
    def __init__(self,in_ch,out_ch,edim=256):
        super(Down,self).__init__()
        self.mconv = nn.Sequential(nn.MaxPool2d(2),
                                   DoubleConv(in_ch,in_ch,resid=True),
                                   DoubleConv(in_ch,out_ch))
        
        self.embed = nn.Sequential(nn.SiLU(),
                                   nn.Linear(edim, out_ch))
        
    def forward(self, x, t):
        x = self.mconv(x)
        
        emb = self.embed(t)[:,:,None,None].repeat(1,1,x.shape[-2],x.shape[-1])
        #print(x.size(),emb.size())
        out = x + emb
        return out
    
class Up(nn.Module):
    def __init__(self,in_ch,out_ch,edim=256):
        super(Up,self).__init__()
        
        self.up = nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)
        self.conv = nn.Sequential(DoubleConv(in_ch, in_ch, resid=True),
                                  DoubleConv(in_ch, out_ch, in_ch//2))
        
        self.embed = nn.Sequential(nn.SiLU(),
                                   nn.Linear(edim, out_ch))
        
    def forward(self,x,skip,t):
        x = self.up(x)
        x = torch.cat([skip,x],dim=1)
        x = self.conv(x)
        emb = self.embed(t)[:,:,None,None]
        #.repeat(1,1,x.shape[-2],x.shape[-1])
        return x + emb
    

class UNet(nn.Module):
    def __init__(self,cin=3,cout=3,nc=None,time=256):
        super(UNet,self).__init__()
        
        self.time = time
        self.edim = time
        
        self.dc = DoubleConv(cin,64)
        
        self.d1 = Down(64,128,edim=self.edim)
        self.a1 = SelfAttention(ch=128)
        
        self.d2 = Down(128,256,edim=self.edim)
        self.a2 = SelfAttention(ch=256)
        
        self.d3 = Down(256,256,edim=self.edim)
        self.a3 = SelfAttention(ch=256)
        
        self.bn1 = DoubleConv(256,512)
        self.bn2 = DoubleConv(512,512)
        self.bn3 = DoubleConv(512,256)
        
        self.u1 = Up(512,128,edim=self.edim)
        self.a4 = SelfAttention(ch=128)
        
        self.u2 = Up(256,64,edim=self.edim)
        self.a5 = SelfAttention(ch=64)
        
        self.u3 = Up(128,64,edim=self.edim)
        self.a6 = SelfAttention(ch=64)
        
        self.outc = nn.Conv2d(64,cout,kernel_size=1)
        
        if nc is not None:
            self.label_emb = nn.Embedding(nc,self.time)
        
    def posencode(self,t,chn):
        device1 = 'cuda' if torch.cuda.is_available() else 'cpu'
        inv_freq = 1.0/(10000 ** (torch.arange(0,chn,2,device=device1).float()/chn))
        
        enc_a = torch.sin(t.repeat(1,chn//2)*inv_freq)
        enc_b = torch.cos(t.repeat(1,chn//2)*inv_freq)
        
        pos = torch.cat([enc_a,enc_b],dim=-1)
        return pos
    
    def forward(self,x,t,y=None):
        
        t = t.unsqueeze(-1).type(torch.float)
        t = self.posencode(t,self.time)
        
        if y is not None:
            t += self.label_emb(y)
            
        x1 = self.dc(x)
        #print('x1',x1.size())
        
        x2 = self.d1(x1,t)
        
        #print('x2',x2.size())
        x2 = self.a1(x2)
        #print('x2 pt 2',x2.size())
        
        x3 = self.d2(x2,t)
        x3 = self.a2(x3)
        x4 = self.d3(x3,t)
        x4 = self.a3(x4)
        
        x4 = self.bn1(x4)
        x4 = self.bn2(x4)
        x4 = self.bn3(x4)
        
        #print(x4.size())
        #print(x3.size())
        
        x = self.u1(x4,x3,t)
        x = self.a4(x)
        x = self.u2(x,x2,t)
        x = self.a5(x)
        x = self.u3(x,x1,t)
        x = self.a6(x)
        
        out = self.outc(x)
        
        return out
        
