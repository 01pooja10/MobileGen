import time
import torch
import torch.nn as nn
import torch.nn.functional as F


def one_param(m):
    "get model first parameter"
    return next(iter(m.parameters()))

class SelfAttention(nn.Module):
    def __init__(self, channels):
        super(SelfAttention, self).__init__()
        self.channels = channels        
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        size = x.shape[-1]
        x = x.view(-1, self.channels, size * size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, size, size)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            #nn.InstanceNorm2d(mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
            #nn.InstanceNorm2d(out_channels),
        )

    def forward(self, x):
        if self.residual:
            
            out = F.gelu(x + self.double_conv(x))
            
            return out
        else:
            return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class UNet_student_v2(nn.Module):
    def __init__(self, c_in=3, c_out=3, time_dim=256, remove_deep_conv=False):
        super().__init__()
        self.time_dim = time_dim
        self.remove_deep_conv = remove_deep_conv
        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128)
        self.sa1 = SelfAttention(128)
        self.down2 = Down(128, 256)
        self.sa2 = SelfAttention(256)
        
        self.down3 = Down(256, 256)
        self.sa3 = SelfAttention(256)


        if remove_deep_conv:
            self.bot1 = DoubleConv(256, 256)
            self.bot3 = DoubleConv(256, 256)
        else:
            self.bot1 = DoubleConv(256, 256)
            #self.bot2 = DoubleConv(512, 256)
            self.bot3 = DoubleConv(256,128)

        self.up1 = Up(512, 128)
        self.sa4 = SelfAttention(128)
        
        self.up2 = Up(256, 64)
        self.sa5 = SelfAttention(64)
        
        self.up3 = Up(128, 64)
        self.sa6 = SelfAttention(64)
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=one_param(self).device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def unet_forward(self, x, t):
        #print("Input size:",x.size())
        x1 = self.inc(x)
        #print("Input size after inc1 :",x1.size())
        x2 = self.down1(x1, t)
        #print("Input size after down1 :",x2.size())
        x2 = self.sa1(x2)
        #print("Input size after sa1 :",x2.size())
        x3 = self.down2(x2, t)
        #print("Input size after down2 :",x3.size())
        x3 = self.sa2(x3)
        #print("Input size after sa2 :",x3.size())
        
        x4 = self.bot1(x3)
        #print("Input size after bottle1 :",x4.size())
        
        #x4 = self.bot2(x4)
        #print("Input size after bottle2 :",x4.size())
        
        x4 = self.bot3(x4)
        #print("Input size after bottle3 :",x4.size())

      
        x = self.up2(x4, x2, t)
        #print("Input size after up2 :",x.size())
        x = self.sa5(x)
        #print("Input size after sa5 :",x.size())
        x = self.up3(x, x1, t)
        #print("Input size after up3 :",x.size())
        x = self.sa6(x)
        #print("Input size after sa6 :",x.size())
        output = self.outc(x)
        #print("Ouput size :",output.size())
        return output
    
    def forward(self, x, t):
        t = t.unsqueeze(-1)
        t = self.pos_encoding(t, self.time_dim)
        return self.unet_forward(x, t)
    


class UNet_conditional_student_v2(UNet_student_v2):
    def __init__(self, c_in=3, c_out=3, time_dim=256, num_classes=None, **kwargs):
        super().__init__(c_in, c_out, time_dim, **kwargs)
        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_dim)

    def forward(self, x, t, y=None):
        t = t.unsqueeze(-1)
        t = self.pos_encoding(t, self.time_dim)

        if y is not None:
            t += self.label_emb(y)

        return self.unet_forward(x, t)


'''
def test_time():
    
    device = 'cuda'
    model = UNet().to(device)
    test_arr = torch.randn((1,3,64,64)).to(device)
    ttime = torch.randint(low=1, high=64,size=(1,)).to(device)
    
    st1 = time.process_time()
    out1 = model(test_arr,ttime)
    end1 = time.process_time()
    
    print("Time taken:", (end1-st1))
    
#test_time()
'''
